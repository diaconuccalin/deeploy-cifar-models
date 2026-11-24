"""Build the deployable Deeploy test folder for a trained int8 MiniMobileNetV2.

Takes the integer parameters from integerize.py and writes them into the graph that was
validated on Siracusa (models_test/miniMobileNetv2_relu_gap_w<W>_c<C>), then
 - checks the integer test-set accuracy with the bit-exact reference executor (int_ref),
 - writes network.onnx, inputs.npz (a real CIFAR test image, uint8 values) and outputs.npz.

Usage (base env, onnx available): python build_mmnv2_onnx.py <int_params.npz> <out_dir> [--image-idx 0]
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np
import onnx
from onnx import numpy_helper as nh

sys.path.insert(0, os.path.dirname(__file__))
import int_ref  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("params")
p.add_argument("out")
p.add_argument("--image-idx", type=int, default=0)
a = p.parse_args()

P = dict(np.load(a.params))
W, C = int(P["meta.width"]), int(P["meta.classes"])
tmpl = os.path.expanduser(f"~/tallinn/models_test/miniMobileNetv2_relu_gap_w{W}_c{C}/network.onnx")
m = copy.deepcopy(onnx.load(tmpl))
g = m.graph
inits = {i.name: i for i in g.initializer}
nodes = list(g.node)


def set_init(name, arr):
    inits[name].CopyFrom(nh.from_array(np.asarray(arr, dtype=np.float32).reshape(nh.to_array(inits[name]).shape)
                                       if np.asarray(arr).size == nh.to_array(inits[name]).size else
                                       np.asarray(arr, dtype=np.float32), name))


def set_attr(node, name, val):
    for at in node.attribute:
        if at.name == name:
            old = nh.to_array(at.t)
            at.t.CopyFrom(nh.from_array(np.full(old.shape, val, dtype=np.float32)))
            return
    raise KeyError(name)


def rqs_of(conv):
    return next(n for n in nodes if n.op_type == "RequantShift" and n.input[0] == conv.output[0])


convs = [n for n in nodes if n.op_type == "Conv"]
names = ["stem", "b1.expand", "b1.dw", "b1.project", "b2.expand", "b2.dw", "b2.project", "b3.expand", "b3.dw",
         "b3.project"]
assert len(convs) == 11 and convs[-1].name == "Conv_gap"
for conv, nm in zip(convs[:10], names):
    r = rqs_of(conv)
    set_init(conv.input[1], P[f"{nm}.weight"])
    set_init(r.input[1], P[f"{nm}.mul"].reshape(-1, 1, 1))
    set_init(r.input[2], P[f"{nm}.add"].reshape(-1, 1, 1))
    set_attr(r, "div", float(P[f"{nm}.div"]))
    set_attr(r, "signed", float(P[f"{nm}.signed"]))

byname = {n.name: n for n in nodes}
for rq, nm in [("RequantShift_8", "b2.skip"), ("RequantShift_15", "b2.branch"), ("RequantShift_17", "b2.add"),
               ("RequantShift_26", "fc")]:
    r = byname[rq]
    set_init(r.input[1], P[f"{nm}.mul"])
    set_init(r.input[2], P[f"{nm}.add"])
    set_attr(r, "div", float(P[f"{nm}.div"]))
# sanity: residual wiring of the template (RQS_8 on block-1 output, RQS_15 on block-2 project output)
assert byname["RequantShift_8"].input[0] == rqs_of(convs[3]).output[0]
assert byname["RequantShift_15"].input[0] == rqs_of(convs[6]).output[0]

# GAP: ones-kernel DW conv + RQS mul = div / (H*W)
rg = byname["RequantShift_gap"]
hw = int(np.prod(nh.to_array(inits["gap.weight"]).shape[2:]))
D = 14
set_init(rg.input[1], np.full(nh.to_array(inits[rg.input[1]]).shape, 2**D / hw))
set_init(rg.input[2], np.zeros(nh.to_array(inits[rg.input[2]]).shape))
set_attr(rg, "div", float(2**D))

gemm = byname["Gemm_25"]
set_init(gemm.input[1], P["fc.weight"])
set_init(gemm.input[2], P["fc.bias"])

# ---- integer test accuracy (bit-exact with Deeploy) and golden I/O
ds = "cifar-100-python" if C == 100 else "cifar-10-batches-py"
root = os.path.expanduser("~/tallinn/data/" + ds)
d = pickle.load(open(os.path.join(root, "test" if C == 100 else "test_batch"), "rb"), encoding="latin1")
X = np.asarray(d["data"], dtype=np.int64).reshape(-1, 3, 32, 32)
Y = np.asarray(d["fine_labels" if C == 100 else "labels"])
correct, ties = 0, 0
for i in range(0, len(X), 500):
    out = int_ref.run(m, {"input": X[i:i + 500]})["output"]
    correct += (out.argmax(1) == Y[i:i + 500]).sum()
    top = out.max(1, keepdims=True)
    ties += ((out == top).sum(1) > 1).sum()
acc = 100.0 * correct / len(X)
print(f"integer test accuracy: {acc:.2f}%  (argmax ties: {ties})")

os.makedirs(a.out, exist_ok=True)
onnx.save(m, os.path.join(a.out, "network.onnx"))
inp = X[a.image_idx:a.image_idx + 1]
out = int_ref.run(m, {"input": inp})["output"].astype(np.int64)
np.savez(os.path.join(a.out, "inputs.npz"), input=inp)
np.savez(os.path.join(a.out, "outputs.npz"), output=out)
with open(os.path.join(a.out, "accuracy.txt"), "w") as f:
    f.write(f"integer (bit-exact) CIFAR-{C} test accuracy: {acc:.2f}%\n")
print(f"saved to {a.out}; test image {a.image_idx}: label={Y[a.image_idx]} pred={out.argmax()}")
