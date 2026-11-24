"""Generate widened miniMobileNetv2 variants (32x32 input) in the Deeploy/QuantLib integerized
format, with random int8 weights and calibrated per-channel requant params, plus bit-exact
golden inputs/outputs. Purpose: check deployability on Siracusa before training.

Usage: python gen_widened_mmnv2.py <width_mult> <num_classes> <out_dir> [--seed N]
"""
import argparse, copy, os, sys
import numpy as np, onnx
from onnx import numpy_helper as nh
sys.path.insert(0, os.path.dirname(__file__))
import int_ref

REF = os.path.expanduser("~/Deeploy/DeeployTest/Tests/Models/miniMobileNetv2/network.onnx")
TARGET_STD = 32.0  # aim per-channel int8 activation std

p = argparse.ArgumentParser()
p.add_argument("width", type=int); p.add_argument("classes", type=int); p.add_argument("out")
p.add_argument("--relu", action="store_true", help="unsigned (ReLU) outputs after stem/expand/dw, as in MobileNetV2")
p.add_argument("--gap", action="store_true", help="global avg pool before classifier (instead of Flatten)")
p.add_argument("--seed", type=int, default=0); p.add_argument("--res", type=int, default=32)
a = p.parse_args()
rng = np.random.default_rng(a.seed)

m = copy.deepcopy(onnx.load(REF)); g = m.graph
inits = {i.name: i for i in g.initializer}
consumers = {n.name: n for n in g.node}

def set_init(name, arr):
    inits[name].CopyFrom(nh.from_array(arr.astype(np.float32), name))

# 1) widen: every conv out-channel and in-channel (except the image's 3) times width
spatial = a.res // 4  # two stride-2 layers
for n in g.node:
    if n.op_type == "Conv":
        w = nh.to_array(inits[n.input[1]])
        co, ci, kh, kw = w.shape
        groups = [x for x in n.attribute if x.name == "group"][0]
        co *= a.width
        if groups.i > 1:
            groups.i = co
        elif ci != 3:
            ci *= a.width
        lim = np.abs(w).max()
        set_init(n.input[1], rng.integers(-lim, lim + 1, size=(co, ci, kh, kw)))
        for r in g.node:  # resize the following per-channel requant params
            if r.op_type == "RequantShift" and r.input[0] == n.output[0]:
                set_init(r.input[1], np.ones((co, 1, 1))); set_init(r.input[2], np.zeros((co, 1, 1)))
    elif n.op_type == "Gemm":
        w = nh.to_array(inits[n.input[1]])
        fin = w.shape[1] // 16 * a.width * spatial * spatial  # ref: 32ch*4*4
        set_init(n.input[1], rng.integers(-6, 7, size=(a.classes, fin)))
        set_init(n.input[2], rng.integers(-512, 513, size=(a.classes,)))

if a.gap:
    # Global average pool expressed with already-supported ops: depthwise Conv (kernel = HxW,
    # all-ones weights) + per-channel RequantShift with mul/div = 1/(H*W) -> exact rounded mean.
    # (Deeploy's integer ReduceMean path is untested on Siracusa and crashes in lowering/binding.)
    C = 32 * a.width
    fl = [n for n in g.node if n.op_type == "Flatten"][0]
    rq_last = [n for n in g.node if n.output[0] == fl.input[0]][0]
    div = float(int_ref._attr(rq_last, "div"))
    g.initializer.extend([
        nh.from_array(np.ones((C, 1, spatial, spatial), np.float32), "gap.weight"),
        nh.from_array(np.full((C, 1, 1), div / spatial**2, np.float32), "gap_rq.mul"),
        nh.from_array(np.zeros((C, 1, 1), np.float32), "gap_rq.add")])
    for i in g.initializer[-3:]:
        inits[i.name] = i
    conv = onnx.helper.make_node("Conv", [fl.input[0], "gap.weight"], ["gap_sum"], name="Conv_gap",
                                 kernel_shape=[spatial, spatial], strides=[1, 1], pads=[0, 0, 0, 0],
                                 dilations=[1, 1], group=C)
    rq = copy.deepcopy(rq_last); rq.name = "RequantShift_gap"
    rq.input[:] = ["gap_sum", "gap_rq.mul", "gap_rq.add"]; rq.output[:] = ["gap_out"]
    fl.input[0] = "gap_out"
    idx = list(g.node).index(fl)
    g.node.insert(idx, rq); g.node.insert(idx, conv)
    for n in g.node:
        if n.op_type == "Gemm":
            set_init(n.input[1], rng.integers(-32, 33, size=(a.classes, C)))

# 2) fix I/O shapes, drop stale value_info
g.input[0].type.tensor_type.shape.dim[2].dim_value = a.res
g.input[0].type.tensor_type.shape.dim[3].dim_value = a.res
g.output[0].type.tensor_type.shape.dim[1].dim_value = a.classes
del g.value_info[:]

RELU_RQS = {"RequantShift_1", "RequantShift_3", "RequantShift_5", "RequantShift_10", "RequantShift_12",
            "RequantShift_19", "RequantShift_21"}
if a.relu:
    for n in g.node:
        if n.name in RELU_RQS:
            for at in n.attribute:
                if at.name == "signed":
                    at.t.CopyFrom(nh.from_array(np.array(0.0, dtype=np.float32)))

# 3) calibrate requant mul/add layer-by-layer on a batch of random images
x = {"input": rng.integers(0, 256, size=(16, 3, a.res, a.res))}
div_of = lambda n: float(int_ref._attr(n, "div"))
for n in g.node:
    if n.op_type != "RequantShift" or n.name == "RequantShift_gap":
        continue
    _, env = int_ref.run(m, x, return_all=True)  # recompute with params calibrated so far
    pre = env[n.input[0]]
    mul_shape = nh.to_array(inits[n.input[1]]).shape
    div = div_of(n)
    if len(mul_shape) == 3:  # per-channel
        std = pre.transpose(1, 0, 2, 3).reshape(pre.shape[1], -1).std(1) + 1e-6
        mean = pre.transpose(1, 0, 2, 3).reshape(pre.shape[1], -1).mean(1)
        mul = np.clip(np.round(div * TARGET_STD / std), 1, 2**15)
        target = 40.0 if (a.relu and n.name in RELU_RQS) else 0.0
        add = np.round((target - mean * mul / div + rng.normal(0, 0.3, size=mul.shape)) * div)
        set_init(n.input[1], mul.reshape(mul_shape)); set_init(n.input[2], add.reshape(mul_shape))
    else:  # scalar (residual-branch and classifier requants)
        mul = np.clip(np.round(div * TARGET_STD / (pre.std() + 1e-6)), 1, 2**15)
        if "ADDITION" in n.input[1] and "acts" in n.input[1]:
            mul = div / 2  # keep both residual inputs at half-scale, like the reference
        set_init(n.input[1], np.array(mul)); set_init(n.input[2], np.array(0.0))

# (onnx checker rejects the custom RequantShift op, as for the reference model)
# rebuild value_info (Deeploy needs intermediate shapes; onnx can't infer the custom op)
_, env1 = int_ref.run(m, {"input": np.zeros((1, 3, a.res, a.res))}, return_all=True)
io = {i.name for i in g.input} | {o.name for o in g.output} | set(inits)
for n in g.node:
    for o in n.output:
        if o not in io:
            g.value_info.append(onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, env1[o].shape))
os.makedirs(a.out, exist_ok=True)
onnx.save(m, os.path.join(a.out, "network.onnx"))

inp = rng.integers(0, 256, size=(1, 3, a.res, a.res)).astype(np.int64)
outs, env = int_ref.run(m, {"input": inp}, return_all=True)
np.savez(os.path.join(a.out, "inputs.npz"), input=inp)
np.savez(os.path.join(a.out, "outputs.npz"), output=outs["output"].astype(np.int64))

# report
_, envb = int_ref.run(m, x, return_all=True)
params = sum(nh.to_array(i).size for i in g.initializer if i.name.endswith(".weight"))
print(f"width={a.width} classes={a.classes} weights={params}")
for n in g.node:
    if n.op_type == "RequantShift":
        y = envb[n.output[0]]
        print(f"  {n.name:18s} {str(y.shape[1:]):16s} std={y.std():5.1f} sat={np.mean((y >= 127) | (y <= -128))*100:4.1f}% min={y.min():4.0f} zero={np.mean(y == 0)*100:4.1f}%")
print("  output:", outs["output"].astype(int).ravel()[:10])
