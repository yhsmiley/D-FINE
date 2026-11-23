"""
Check an exported ONNX against the original PyTorch weights on a folder of images.

Reference = PyTorch model built with the ORIGINAL cross attention (grid_sample), not the "gather" replacement,
so the check also covers the opset 11 sampler and every rewrite in trt7_onnx_fix.

Both sides get the same input: the image resized to --size (no letterbox) and scaled to 0..1.
Detections above --score-thr are compared by label and IoU, not by row, because the query order can differ.

Exit code 0 = pass, 1 = fail.

Example (inside the docker image):
  python3 verify_onnx.py --onnx weights/dfine_n_coco.onnx --weights weights/dfine_n_coco.pth \
      -c configs/dfine/custom/dfine_hgnetv2_n_sits.yml -s 640 --images coco128/images/train2017
  python3 verify_onnx.py --model yolo11 --onnx weights/maritime/YOLO11n_all_maritime.onnx \
      --weights weights/maritime/YOLO11n_all_maritime.pt -s 1280 --images <folder>
"""

import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# onnxruntime 1.24.4 fuses Pad(0) into a following MaxPool, which is wrong (a MaxPool ignores its own padding, a Pad
# of 0 does not). The HGNetv2 stem has exactly this pattern, so the default level changes some boxes. The exported
# file is fine, so the check runs with the optimizer off unless asked otherwise.
ORT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def build_dfine_reference(weights, cfg_file, slice_classes):
    from export_dfine_trt7 import DeepStreamOutput, slice_class_head
    from src.core import YAMLConfig

    cfg = YAMLConfig(cfg_file, resume=weights)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["DFINETransformer"]["cross_attn_method"] = "default"    # the original grid_sample path
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    if slice_classes:
        state = slice_class_head(state, cfg.model.state_dict())
    cfg.model.load_state_dict(state)
    return cfg.model.deploy(), cfg.postprocessor.use_focal_loss, DeepStreamOutput


def build_reference(args):
    device = torch.device("cpu")
    if args.model == "dfine":
        model, use_focal_loss, head = build_dfine_reference(args.weights, args.config, args.slice_classes)
        return nn.Sequential(model, head(args.img_size, use_focal_loss)).to(device).eval()
    from export_yolo11_trt7 import DeepStreamOutput, yolo11_export
    return nn.Sequential(yolo11_export(args.weights, device), DeepStreamOutput()).eval()


def load_image(path, img_size):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((img_size[1], img_size[0]))
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def iou_one_to_many(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / (union + 1e-9)


def match(src, dst, iou_thr, margin, thr):
    """For each det in src with score >= thr + margin, find the best same-label det in dst (dst already >= thr).
    Returns (n_checked, n_missing, ious, score_diffs). Dets inside the margin are skipped: a score of 0.3004 vs
    0.2996 flips across the threshold without being a real difference."""
    checked = missing = 0
    ious, sdiffs = [], []
    for d in src:
        if d[4] < thr + margin:
            continue
        checked += 1
        cand = dst[dst[:, 5] == d[5]]
        if len(cand) == 0:
            missing += 1
            continue
        v = iou_one_to_many(d[:4], cand[:, :4])
        j = int(v.argmax())
        ious.append(float(v[j]))
        sdiffs.append(abs(float(d[4]) - float(cand[j, 4])))
        if v[j] < iou_thr:
            missing += 1
    return checked, missing, ious, sdiffs


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Compare an exported ONNX with the PyTorch weights on real images")
    p.add_argument("--onnx", required=True, help="Exported ONNX file")
    p.add_argument("-w", "--weights", required=True, help="PyTorch weights (.pth for dfine, .pt for yolo11)")
    p.add_argument("-c", "--config", help="D-FINE YAML config (required for --model dfine)")
    p.add_argument("--model", choices=["dfine", "yolo11"], default="dfine")
    p.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Input size [H,W] (default [640])")
    p.add_argument("--images", required=True, help="Folder of images (searched recursively)")
    p.add_argument("--max-images", type=int, default=0, help="Use only the first N images (0 = all)")
    p.add_argument("--slice-classes", action="store_true", help="Same flag as the export, for a wider class head")
    p.add_argument("--score-thr", type=float, default=0.3, help="Compare detections above this score")
    p.add_argument("--score-margin", type=float, default=0.01, help="Ignore unmatched dets this close to the threshold")
    p.add_argument("--iou-thr", type=float, default=0.9, help="Min IoU for two detections to count as the same")
    p.add_argument("--max-missing", type=int, default=0, help="Allowed unmatched detections in total (default 0)")
    p.add_argument("--ort-opt", choices=list(ORT_LEVELS), default="disable",
                   help="onnxruntime graph optimization level (default disable, see the note at the top)")
    args = p.parse_args()
    if args.model == "dfine" and not args.config:
        p.error("--config is required for --model dfine")
    for path in (args.onnx, args.weights):
        if not os.path.isfile(path):
            p.error("file not found: %s" % path)
    args.img_size = args.size * 2 if len(args.size) == 1 else args.size
    return args


def main():
    args = parse_args()
    files = sorted(f for f in glob.glob(os.path.join(args.images, "**", "*"), recursive=True)
                   if f.lower().endswith(IMAGE_EXTS))
    if args.max_images:
        files = files[: args.max_images]
    if not files:
        raise SystemExit("no images found in %s" % args.images)

    print("Loading the PyTorch reference")
    reference = build_reference(args)

    so = ort.SessionOptions()
    so.graph_optimization_level = ORT_LEVELS[args.ort_opt]
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    onnx_hw = [d for d in inp.shape[2:]]
    if all(isinstance(d, int) for d in onnx_hw) and list(onnx_hw) != list(args.img_size):
        raise SystemExit("the ONNX input is %s but --size is %s" % (inp.shape, args.img_size))

    thr, margin = args.score_thr, args.score_margin
    n_ref = n_onnx = n_missing = n_extra = n_checked_ref = n_checked_onnx = count_diff = 0
    ious, sdiffs = [], []
    for i, f in enumerate(files, 1):
        x = load_image(f, args.img_size)
        with torch.no_grad():
            ref = reference(x).numpy()[0]
        out = sess.run(None, {inp.name: x.numpy()})[0][0]
        ref, out = ref[ref[:, 4] >= thr], out[out[:, 4] >= thr]
        n_ref += len(ref)
        n_onnx += len(out)
        count_diff += len(ref) != len(out)
        c, m, io, sd = match(ref, out, args.iou_thr, margin, thr)       # reference dets the ONNX lost
        n_checked_ref += c
        n_missing += m
        ious += io
        sdiffs += sd
        c, m, _, _ = match(out, ref, args.iou_thr, margin, thr)         # ONNX dets the reference does not have
        n_checked_onnx += c
        n_extra += m
        if i % 20 == 0 or i == len(files):
            print("  %d/%d images" % (i, len(files)), flush=True)

    print("\nimages: %d, score >= %.2f, onnxruntime optimization: %s" % (len(files), thr, args.ort_opt))
    print("detections: torch %d, onnx %d, images with a different count: %d" % (n_ref, n_onnx, count_diff))
    print("torch dets missing in onnx: %d / %d, onnx dets missing in torch: %d / %d"
          % (n_missing, n_checked_ref, n_extra, n_checked_onnx))
    if ious:
        print("matched IoU: min %.4f, median %.4f; score abs diff: max %.4f, median %.5f"
              % (min(ious), float(np.median(ious)), max(sdiffs), float(np.median(sdiffs))))

    failed = n_missing + n_extra > args.max_missing
    print("\n%s (allowed unmatched: %d)" % ("FAIL" if failed else "PASS", args.max_missing))
    if failed and args.ort_opt != "disable":
        print("note: onnxruntime optimizations are on; re-run with --ort-opt disable to rule out the Pad/MaxPool fusion bug")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
