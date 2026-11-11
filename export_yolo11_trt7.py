import os
from copy import deepcopy

import onnx
import torch
import torch.nn as nn

# the TensorRT 7 ONNX post-processing is shared with the D-FINE export script
from export_dfine_trt7 import suppress_warnings, trt7_onnx_fix


class DeepStreamOutput(nn.Module):
    def forward(self, x):
        # x: [batch, 4 + num_classes, anchors], boxes are cx, cy, w, h
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype, device=boxes.device
        )
        boxes @= convert_matrix
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def yolo11_export(weights, device):
    from ultralytics.nn.modules import C2f, Detect
    try:
        from ultralytics.nn.tasks import load_checkpoint
        model, _ = load_checkpoint(weights, device=device, inplace=True, fuse=True)
    except ImportError:    # older ultralytics only has attempt_load_weights
        from ultralytics.nn.tasks import attempt_load_weights
        model = attempt_load_weights(weights, device=device, inplace=True, fuse=True)
    model = deepcopy(model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for m in model.modules():
        if isinstance(m, Detect):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model


def main(args):
    suppress_warnings()

    print(f"\nStarting: {args.weights}")

    print("Opening YOLO11 model")

    device = torch.device("cpu")
    model = yolo11_export(args.weights, device)

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    model = nn.Sequential(model, DeepStreamOutput())

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
    parser = argparse.ArgumentParser(description="DeepStream YOLO11 conversion (TensorRT 7)")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path (required)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] (default [640])")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set dynamic batch-size and static batch-size at same time")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
