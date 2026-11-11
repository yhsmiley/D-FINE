import os
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import YAMLConfig


class DeepStreamOutput(nn.Module):
    def __init__(self, img_size, use_focal_loss):
        super().__init__()
        self.img_size = img_size
        self.use_focal_loss = use_focal_loss

    def forward(self, x):
        boxes = x["pred_boxes"]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype, device=boxes.device
        )
        boxes @= convert_matrix
        boxes *= torch.as_tensor([[*self.img_size]]).flip(1).tile([1, 2]).unsqueeze(1)
        scores = F.sigmoid(x["pred_logits"]) if self.use_focal_loss else F.softmax(x["pred_logits"])[:, :, :-1]
        scores, labels = torch.max(scores, dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


# A checkpoint trained with a wider class head than the config (e.g. 80 outputs for 2 real classes) is cut down to the
# config's num_classes. This keeps classes 0..N-1 and assumes the labels used in training were exactly those.
# denoising_class_embed has num_classes + 1 rows, its last row is the padding index, so that row is kept as the last.
def slice_class_head(state, model_state):
    state = dict(state)
    for k, v in state.items():
        want = model_state.get(k)
        if want is None or v.shape == want.shape or "score_head" not in k and "denoising_class_embed" not in k:
            continue
        if v.shape[1:] != want.shape[1:] or v.shape[0] < want.shape[0]:
            continue
        if "denoising_class_embed" in k:
            idx = list(range(want.shape[0] - 1)) + [v.shape[0] - 1]
            state[k] = v[idx]
        else:
            state[k] = v[: want.shape[0]]
        print(f"Sliced {k}: {tuple(v.shape)} -> {tuple(want.shape)}")
    return state


def dfine_export(weights, cfg_file, device, slice_classes=False):
    cfg = YAMLConfig(cfg_file, resume=weights)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    if slice_classes:
        state = slice_class_head(state, cfg.model.state_dict())
    cfg.model.load_state_dict(state)
    return cfg.model.deploy().to(device), cfg.postprocessor.use_focal_loss


# Post-process the ONNX (opset 11) so the TensorRT 7 ONNX parser (JetPack 4.5.1 / DeepStream 5.1) accepts it
# and computes the right values. Every rewrite is mathematically equivalent to the original graph.
#
# - Constant nodes (an export without --simplify keeps them) become initializers, so the patterns below match them
# - Pad / Clip with an empty optional input: drop it (Pad) or use the largest value of the data type (Clip), TensorRT 7
#   needs constants
# - Gather with a scalar index: use a [1] index plus Squeeze, TensorRT 7 keeps the gathered axis for 0-d indices;
#   negative indices are resolved to idx + dim since TensorRT 7 Gather returns zeros for them
# - MatMul with a 1-D constant right operand: use a [K,1] operand plus Squeeze, TensorRT 7 returns rank 0
# - Split with several outputs: one Slice per output, TensorRT 7 gives every slice layer of a Split the node name and
#   the build fails with "repeated layer name"
# Keep this function in sync with utils/trt7_onnx_fix.py in DeepStream-Yolo (the standalone version).
def trt7_onnx_fix(src, dst):
    import numpy as np
    from onnx import helper, numpy_helper, shape_inference
    m = onnx.load(src)
    opset = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=0)
    if opset > 12:
        raise SystemExit("opset %d: the Squeeze rewrite needs opset 12 or lower (opset 11 is the tested one)" % opset)
    g = m.graph
    kept, n_const = [], 0
    outputs = set(o.name for o in g.output)
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] not in outputs and any(a.name == "value" for a in n.attribute):
            t = onnx.TensorProto()
            t.CopyFrom(next(a.t for a in n.attribute if a.name == "value"))
            t.name = n.output[0]
            g.initializer.append(t)
            n_const += 1
        else:
            kept.append(n)
    if n_const:
        del g.node[:]
        g.node.extend(kept)
    inferred = shape_inference.infer_shapes(m).graph
    vi_shape, vi_type = {}, {}
    for v in inferred.value_info:
        vi_type[v.name] = v.type.tensor_type.elem_type
        if v.type.tensor_type.HasField("shape"):    # no shape field means unknown, not a scalar
            vi_shape[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    for v in g.input:
        if v.type.tensor_type.HasField("shape"):
            vi_shape.setdefault(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim])
        vi_type.setdefault(v.name, v.type.tensor_type.elem_type)
    init = {i.name: i for i in g.initializer}
    for i in g.initializer:
        vi_shape.setdefault(i.name, list(i.dims))
        vi_type.setdefault(i.name, i.data_type)

    def np_dtype(name):
        et = vi_type.get(name)
        if et is None:
            skipped["clip_type_unknown"] += 1
            return np.dtype(np.float32)
        if hasattr(helper, "tensor_dtype_to_np_dtype"):
            return np.dtype(helper.tensor_dtype_to_np_dtype(et))
        return np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[et])

    def clip_limit(dt, kind):
        # float32 keeps the plain names; other types get a suffix
        name = "trt7_clip_%s%s" % (kind, "" if dt == np.float32 else "_" + dt.name)
        if name not in init:
            info = np.finfo(dt) if dt.kind == "f" else np.iinfo(dt)
            value = info.max if kind == "max" else info.min if dt.kind != "f" else -info.max
            t = numpy_helper.from_array(np.array(value, dtype=dt), name)
            g.initializer.append(t)
            init[name] = t
        return name

    new_nodes = []
    count = {"constant": n_const, "pad": 0, "clip": 0, "gather": 0, "matmul": 0, "split": 0}
    skipped = {"gather_not_constant": 0, "matmul_unknown_rank": 0, "clip_type_unknown": 0, "split_unknown_shape": 0}
    for n in g.node:
        if n.op_type == "Pad" and len(n.input) == 3 and n.input[2] == "":
            del n.input[2]
            count["pad"] += 1
        elif n.op_type == "Clip" and len(n.input) >= 2 and (n.input[1] == "" or len(n.input) == 2 or n.input[2] == ""):
            dt = np_dtype(n.input[0])
            if n.input[1] == "":
                n.input[1] = clip_limit(dt, "min")
            if len(n.input) == 2:
                n.input.append("")
            if n.input[2] == "":
                n.input[2] = clip_limit(dt, "max")
            count["clip"] += 1
        elif n.op_type == "Gather" and n.input[1] in init and len(init[n.input[1]].dims) == 0:
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            idx = numpy_helper.to_array(init[n.input[1]]).reshape(1)
            if idx[0] < 0:
                # TensorRT 7 Gather does not support negative indices
                dims = vi_shape.get(n.input[0])
                if dims is None or not -len(dims) <= axis < len(dims) or dims[axis] <= 0:
                    raise SystemExit("cannot resolve the negative Gather index of %s: the size of axis %d of %s is "
                                     "unknown or dynamic. An ONNX exported without --simplify often has such shape "
                                     "tensors, export with --simplify" % (n.name, axis, n.input[0]))
                idx = idx + dims[axis]
            # the same scalar can be shared by Gathers whose resolved value differs, so reuse a name only if equal
            base = n.input[1] + "_trt7_1d"
            idx_name, k = base, 0
            while idx_name in init and not np.array_equal(numpy_helper.to_array(init[idx_name]), idx):
                k += 1
                idx_name = "%s_%d" % (base, k)
            if idx_name not in init:
                t = numpy_helper.from_array(idx, idx_name)
                g.initializer.append(t)
                init[idx_name] = t
            n.input[1] = idx_name
            out = n.output[0]
            tmp = out + "_trt7_pre"
            n.output[0] = tmp
            new_nodes.append(n)
            new_nodes.append(helper.make_node("Squeeze", [tmp], [out], name=(n.name or out) + "_trt7_squeeze", axes=[axis]))
            count["gather"] += 1
            continue
        elif n.op_type == "MatMul" and n.input[1] in init and len(init[n.input[1]].dims) == 1:
            if n.input[0] not in vi_shape:
                skipped["matmul_unknown_rank"] += 1
            else:
                a_rank = len(vi_shape[n.input[0]])
                col_name = n.input[1] + "_trt7_col"
                if col_name not in init:
                    t = numpy_helper.from_array(numpy_helper.to_array(init[n.input[1]]).reshape(-1, 1), col_name)
                    g.initializer.append(t)
                    init[col_name] = t
                n.input[1] = col_name
                out = n.output[0]
                tmp = out + "_trt7_pre"
                n.output[0] = tmp
                new_nodes.append(n)
                new_nodes.append(helper.make_node("Squeeze", [tmp], [out], name=(n.name or out) + "_trt7_squeeze",
                                                  axes=[a_rank - 1]))
                count["matmul"] += 1
                continue
        elif n.op_type == "Split" and len(n.output) > 1:
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            sizes = next((list(a.ints) for a in n.attribute if a.name == "split"), None)
            dims = vi_shape.get(n.input[0])
            if len(n.input) > 1 or dims is None or not -len(dims) <= axis < len(dims) or \
                    (sizes is None and dims[axis] <= 0):
                skipped["split_unknown_shape"] += 1
            else:
                axis %= len(dims)
                if sizes is None:    # equal split
                    sizes = [dims[axis] // len(n.output)] * len(n.output)
                start = 0
                for k, (out, size) in enumerate(zip(n.output, sizes)):
                    names = [(n.name or n.output[0]) + "_trt7_slice%d_%s" % (k, x) for x in ("starts", "ends", "axes")]
                    for nm, val in zip(names, ([start], [start + size], [axis])):
                        t = numpy_helper.from_array(np.array(val, dtype=np.int64), nm)
                        g.initializer.append(t)
                        init[nm] = t
                    new_nodes.append(helper.make_node("Slice", [n.input[0]] + names, [out],
                                                      name="%s_trt7_slice%d" % (n.name or n.output[0], k)))
                    start += size
                count["split"] += 1
                continue
        elif n.op_type == "Gather" and n.input[1] not in init:
            skipped["gather_not_constant"] += 1
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    # drop value_info: shapes changed in the middle
    del g.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, dst + ".tmp")    # write then rename, so dst is never left half written
    os.replace(dst + ".tmp", dst)
    print("saved", dst, count)
    if skipped["clip_type_unknown"]:
        print("WARNING: the data type of %d Clip inputs is unknown, float32 limits were used" % skipped["clip_type_unknown"])
    if skipped["matmul_unknown_rank"]:
        print("WARNING: %d MatMul with a 1-D constant operand were not rewritten (unknown input rank)"
              % skipped["matmul_unknown_rank"])
    if skipped["split_unknown_shape"]:
        print("WARNING: %d Split nodes were not rewritten (unknown input shape or Split-13 style input), TensorRT 7 "
              "may fail with 'repeated layer name'" % skipped["split_unknown_shape"])
    if skipped["gather_not_constant"]:
        print("note: %d Gather nodes have a non-constant index and were left as they are (only matters if the index "
              "is a scalar)" % skipped["gather_not_constant"])


def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f"\nStarting: {args.weights}")

    print("Opening D-FINE model")

    device = torch.device("cpu")
    model, use_focal_loss = dfine_export(args.weights, args.config, device, args.slice_classes)

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    model = nn.Sequential(model, DeepStreamOutput(img_size, use_focal_loss))

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"
    # the unpatched export goes here; onnx_output_file is only written once the TensorRT 7 fix has succeeded
    raw_file = onnx_output_file + ".raw"

    dynamic_axes = {
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }

    print("Exporting the model to ONNX")
    torch.onnx.export(
        model,
        onnx_input_im,
        raw_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print("Simplifying the ONNX model")
        import onnxslim
        model_onnx = onnx.load(raw_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, raw_file)

    print("Applying TensorRT 7 fixes")
    try:
        trt7_onnx_fix(raw_file, onnx_output_file)
    except SystemExit:
        print(f"\nTensorRT 7 fix failed, {onnx_output_file} was not written. The unpatched export is kept as {raw_file}")
        raise
    os.remove(raw_file)

    print(f"Done: {onnx_output_file}\n")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DeepStream D-FINE conversion (TensorRT 7)")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pth) file path (required)")
    parser.add_argument("-c", "--config", required=True, type=str, help="Input YAML (.yml) file path (required)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] (default [640])")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--slice-classes", action="store_true",
                        help="Cut a wider class head in the checkpoint down to num_classes of the config")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if not os.path.isfile(args.config):
        raise RuntimeError("Invalid config file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set dynamic batch-size and static batch-size at same time")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
