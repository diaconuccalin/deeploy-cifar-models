"""Bit-exact numpy/torch reference executor for QuantLib-style integerized ONNX graphs
(Conv / Gemm / Add / Flatten / RequantShift), as consumed by Deeploy."""
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper as nh


def _attr(n, name, default=None):
    for a in n.attribute:
        if a.name == name:
            v = onnx.helper.get_attribute_value(a)
            return nh.to_array(v) if isinstance(v, onnx.TensorProto) else v
    return default


def run(model, inputs, return_all=False):
    g = model.graph
    env = {i.name: nh.to_array(i).astype(np.float64) for i in g.initializer}
    env.update({k: np.asarray(v, dtype=np.float64) for k, v in inputs.items()})
    for n in g.node:
        x = [env[i] for i in n.input]
        t = n.op_type
        if t == "Conv":
            y = F.conv2d(torch.from_numpy(x[0]), torch.from_numpy(x[1]),
                         torch.from_numpy(x[2]) if len(x) > 2 else None,
                         stride=_attr(n, "strides"), padding=_attr(n, "pads")[:2],
                         dilation=_attr(n, "dilations"), groups=_attr(n, "group", 1)).numpy()
        elif t == "RequantShift":
            div = float(_attr(n, "div")); lv = float(_attr(n, "n_levels_out", _attr(n, "n_levels")))
            signed = bool(_attr(n, "signed"))
            mul, add = x[1], x[2]
            if mul.ndim and x[0].ndim == 2 and mul.ndim > 1:
                mul, add = mul.reshape(1, -1), add.reshape(1, -1)
            y = np.floor((x[0] * mul + add + div // 2) / div)  # Deeploy rounds: += 1 << (log2D-1)
            lo, hi = (-lv / 2, lv / 2 - 1) if signed else (0, lv - 1)
            y = np.clip(y, lo, hi)
        elif t == "Add":
            y = x[0] + x[1]
        elif t == "ReduceMean":  # Deeploy PULP integer mean, keepdims=0, power-of-2 length
            axes = tuple(int(v) for v in (x[1] if len(x) > 1 else _attr(n, "axes"))); assert _attr(n, "keepdims", 1) == 0
            L = int(np.prod([x[0].shape[i] for i in axes])); sh = int(np.log2(L)); assert 2**sh == L
            y = np.floor((x[0].sum(axis=axes) + (1 << (sh - 1))) / L)
        elif t == "Flatten":
            y = x[0].reshape(x[0].shape[0], -1)
        elif t == "Gemm":
            w = x[1].T if _attr(n, "transB", 0) else x[1]
            y = x[0] @ w + (x[2] if len(x) > 2 else 0)
        else:
            raise NotImplementedError(t)
        assert np.all(y == np.round(y)), f"non-integer at {n.name}"
        env[n.output[0]] = y
    outs = {o.name: env[o.name] for o in g.output}
    return (outs, env) if return_all else outs
