"""Build the deployable Deeploy test folder for a trained FP32 CCT_2_32_32_128.

Writes the trained weights (from export_cct_npz.py) into the reference graph
DeeployTest/Tests/Models/CCT/FP32/CCT_2_32_32_128 (same operators/structure, own bias per Gemm),
checks the graph against the PyTorch logits on the whole test set with a float reference executor,
and writes network.onnx, inputs.npz (a normalized CIFAR test image) and outputs.npz.

Usage (base env): python build_cct_onnx.py <weights.npz> <out_dir> [--image-idx 0]
"""
import argparse
import copy
import math
import os
import pickle

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper as nh

REF = os.path.expanduser("~/Deeploy/DeeployTest/Tests/Models/CCT/FP32/CCT_2_32_32_128/network.onnx")
MEAN = {10: (0.4914, 0.4822, 0.4465), 100: (0.5071, 0.4865, 0.4409)}
STD = {10: (0.2470, 0.2435, 0.2616), 100: (0.2673, 0.2564, 0.2762)}

p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("out")
p.add_argument("--image-idx", type=int, default=0)
a = p.parse_args()

Wt = dict(np.load(a.weights))
C = Wt["fc.weight"].shape[0]
m = copy.deepcopy(onnx.load(REF))
g = m.graph
inits = {i.name: i for i in g.initializer}


def put(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if name in inits:
        inits[name].CopyFrom(nh.from_array(arr, name))
    else:
        g.initializer.append(nh.from_array(arr, name))
        inits[name] = g.initializer[-1]


put("tokenizer_conv_layers_0_0_weight", Wt["tokenizer.0.weight"])
put("tokenizer_conv_layers_1_0_weight", Wt["tokenizer.3.weight"])
put("classifier_positional_emb", Wt["positional_emb"])
for i in range(2):
    for ln in ("pre_norm", "norm1"):
        put(f"classifier_blocks_{i}_{ln}_weight", Wt[f"blocks.{i}.{ln}.weight"])
        put(f"classifier_blocks_{i}_{ln}_bias", Wt[f"blocks.{i}.{ln}.bias"])
put("classifier_norm_weight", Wt["norm.weight"])
put("classifier_norm_bias", Wt["norm.bias"])
put("classifier_fc_weight", Wt["fc.weight"])
put("classifier_fc_bias", Wt["fc.bias"])

for n in g.node:
    if n.op_type == "MatMul" and n.input[1] in inits:
        if "attention_pool" in n.name:
            put(n.input[1], Wt["attention_pool.weight"].T)  # its bias cancels inside the softmax
            continue
        blk = int(n.name.split("blocks_")[1].split("_")[0])
        proj = n.name.split("self_attn_")[1].split("_MatMul")[0]  # q_proj / k_proj / v_proj
        put(n.input[1], Wt[f"blocks.{blk}.self_attn.{proj}.weight"].T)
    elif n.op_type == "Gemm" and "blocks_" in n.name:
        blk = int(n.name.split("blocks_")[1].split("_")[0])
        mod = ("self_attn.proj" if "self_attn_proj" in n.name else "linear1" if "linear1" in n.name else "linear2")
        W = Wt[f"blocks.{blk}.{mod}.weight"].T
        put(n.input[1], W[None])  # (1, in, out), used as A @ B
        put(n.input[1].replace("_reshaped", ""), W)  # unused 2D original kept consistent
        bname = f"classifier_blocks_{blk}_{mod.replace('.', '_')}_bias"
        put(bname, Wt[f"blocks.{blk}.{mod}.bias"])
        n.input[2] = bname

# the formerly shared bias name is now block 0's proj bias; unused 2D weight originals are kept
# (they were present in the reference graph as well)
for vi in list(g.value_info) + list(g.output):
    if vi.name == "output":
        vi.type.tensor_type.shape.dim[1].dim_value = C


# ---- float reference executor for the ops in this graph
def attr(n, k, d=None):
    for x in n.attribute:
        if x.name == k:
            return onnx.helper.get_attribute_value(x)
    return d


def run(model, x):
    env = {i.name: torch.from_numpy(nh.to_array(i).copy()) for i in model.graph.initializer}
    env["input"] = x
    pending = list(model.graph.node)  # the reference graph is not stored in topological order
    order = []
    while pending:
        ready = [n for n in pending if all(k in env or k in {o for d in order for o in d.output} for k in n.input)]
        assert ready, "graph has a cycle or missing input"
        order += ready
        pending = [n for n in pending if n not in ready]
    for n in order:
        i = [env[k] for k in n.input]
        t = n.op_type
        if t == "Conv":
            y = F.conv2d(i[0], i[1], i[2] if len(i) > 2 else None, attr(n, "strides"), attr(n, "pads")[:2])
        elif t == "Relu":
            y = F.relu(i[0])
        elif t == "MaxPool":
            y = F.max_pool2d(i[0], attr(n, "kernel_shape"), attr(n, "strides"), attr(n, "pads")[:2])
        elif t == "Reshape":
            shp = [int(v) for v in i[1]]
            y = i[0].reshape([i[0].shape[0]] + shp[1:])  # batch-generalized
        elif t == "Transpose":
            y = i[0].permute(attr(n, "perm"))
        elif t == "Add":
            y = i[0] + i[1]
        elif t == "Mul":
            y = i[0] * i[1]
        elif t == "LayerNormalization":
            y = F.layer_norm(i[0], i[0].shape[-1:], i[1], i[2], attr(n, "epsilon"))
        elif t == "MatMul":
            y = i[0] @ i[1]
        elif t == "Softmax":
            y = i[0].softmax(attr(n, "axis"))
        elif t == "Gelu":
            y = F.gelu(i[0], approximate="tanh")
        elif t == "Squeeze":
            y = i[0].squeeze(attr(n, "axes")[0])
        elif t == "Gemm":
            B = i[1].transpose(-1, -2) if attr(n, "transB", 0) else i[1]
            y = i[0] @ B + i[2]
        else:
            raise NotImplementedError(t)
        env[n.output[0]] = y
    return env["output"]


ds = "cifar-100-python" if C == 100 else "cifar-10-batches-py"
d = pickle.load(open(os.path.expanduser(f"~/tallinn/data/{ds}/" + ("test" if C == 100 else "test_batch")), "rb"),
                encoding="latin1")
X = np.asarray(d["data"], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
X = (X - np.array(MEAN[C], np.float32).reshape(1, 3, 1, 1)) / np.array(STD[C], np.float32).reshape(1, 3, 1, 1)
Y = np.asarray(d["fine_labels" if C == 100 else "labels"])
ref_logits = Wt["_test_logits"]
correct, maxdiff, agree = 0, 0.0, 0
with torch.no_grad():
    for s in range(0, len(X), 500):
        out = run(m, torch.from_numpy(X[s:s + 500])).numpy()
        correct += (out.argmax(1) == Y[s:s + 500]).sum()
        agree += (out.argmax(1) == ref_logits[s:s + 500].argmax(1)).sum()
        maxdiff = max(maxdiff, float(np.abs(out - ref_logits[s:s + 500]).max()))
acc = 100.0 * correct / len(X)
print(f"ONNX graph test accuracy: {acc:.2f}%  | argmax agreement with PyTorch: {100 * agree / len(X):.2f}%"
      f" | max |logit diff|: {maxdiff:.2e}")

os.makedirs(a.out, exist_ok=True)
onnx.save(m, os.path.join(a.out, "network.onnx"))
inp = X[a.image_idx:a.image_idx + 1]
with torch.no_grad():
    out = run(m, torch.from_numpy(inp)).numpy()
np.savez(os.path.join(a.out, "inputs.npz"), input=inp)
np.savez(os.path.join(a.out, "outputs.npz"), output=out.astype(np.float32))
with open(os.path.join(a.out, "accuracy.txt"), "w") as f:
    f.write(f"FP32 CIFAR-{C} test accuracy: {acc:.2f}%\n"
            f"input: RGB image /255, normalized with mean={MEAN[C]} std={STD[C]}, NCHW float32\n")
print(f"saved to {a.out}; test image {a.image_idx}: label={Y[a.image_idx]} pred={out.argmax()}")
