"""
Accuracy against ground truth for the PyTorch model and its exported ONNX, side by side.

Both models run on the same input as verify_onnx.py (image resized to --size, scaled to 0..1). Boxes are scaled back
to the original image size and scored with one of two evaluators (--eval):
  fdet  (default) F1 and F2 scores (macro) from the fdet-api fork. They are computed on the detections kept by
                  --min-score (default 0.3), so use the score you deploy with.
  coco            COCO mAP from faster-coco-eval (a pycocotools replacement already used by this repo), on all
                  detections down to --min-score (default 0.001).
The PyTorch side uses the original grid_sample path, so a drop in the ONNX column is caused by the export.

Class mapping:
  --label-map coco80   (default) the model has the 80 COCO classes. A model label is turned into a class name with
                       the repo's COCO tables and matched BY NAME to the categories of the ground truth file, so the
                       ground truth can use any category ids.
  --label-map identity the model label (plus --category-offset) is used as the ground truth category id. Use this for
                       a custom model such as the maritime one.

Run it on images the model has not seen. coco128 is a subset of COCO train2017, which the COCO checkpoints were trained
on, so the absolute mAP there is optimistic. The comparison between PyTorch and ONNX is what it is good for.

Example (inside the docker image):
  python3 eval_map.py --onnx weights/dfine_n_coco.onnx -w weights/dfine_n_coco.pth \
      -c configs/dfine/custom/dfine_hgnetv2_n_sits.yml -s 640 \
      --images coco128/images --gt coco128/labels.json
"""

import contextlib
import io
import os

import numpy as np
import onnxruntime as ort
import torch

from verify_onnx import ORT_LEVELS, build_reference, load_image


def build_label_to_category(args, gt):
    if args.label_map == "identity":
        return lambda label: int(label) + args.category_offset
    from src.data.dataset import mscoco_category2name, mscoco_label2category
    name_to_id = {c["name"]: c["id"] for c in gt.dataset["categories"]}
    missing = sorted(n for n in mscoco_category2name.values() if n not in name_to_id)
    if missing:
        print("note: %d COCO classes are not in the ground truth, detections of them are ignored" % len(missing))
    return lambda label: name_to_id.get(mscoco_category2name[mscoco_label2category[int(label)]])


def to_results(det, image_id, orig_w, orig_h, img_size, label_to_category):
    sx, sy = orig_w / img_size[1], orig_h / img_size[0]
    out = []
    for x1, y1, x2, y2, score, label in det:
        cat = label_to_category(label)
        if cat is None:
            continue
        out.append({"image_id": image_id, "category_id": cat, "score": float(score),
                    "bbox": [float(x1 * sx), float(y1 * sy), float((x2 - x1) * sx), float((y2 - y1) * sy)]})
    return out


def coco_eval(gt, results, image_ids):
    from faster_coco_eval import COCOeval_faster
    if not results:
        return [0.0] * 12
    dt = gt.loadRes(results)
    ev = COCOeval_faster(gt, dt, iouType="bbox")
    ev.params.imgIds = image_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(s) for s in ev.stats]


def fdet_eval(true_path, preds, image_ids=None):
    """F-beta scores (macro) from the fdet-api fork. preds is a list of COCO result dicts or a json path.
    Returns 10 values, in the order of FDET_ROWS. The scores are computed on exactly the detections given, there is
    no confidence cutoff of its own, so filter them to the score you deploy with first."""
    from fdet_api.coco import COCO
    from fdet_api.cocoeval import COCOeval
    if not preds:
        return [0.0] * 10
    cocoGt = COCO(true_path)  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(preds)  # initialize COCO prediction api
    ev = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize COCO evaluation api
    if image_ids is not None:
        ev.params.imgIds = image_ids
    ev.evaluate()
    ev.accumulateFBeta()
    ev.summarizeFBetaScores(average='macro')
    return [float(s) for s in ev.fstats]    # summarizeFBetaScores fills fstats, not stats


COCO_ROWS = ((0, "mAP 0.50:0.95"), (1, "AP50"), (2, "AP75"), (3, "AP small"), (4, "AP medium"), (5, "AP large"),
             (8, "AR 100"))
FDET_ROWS = ((0, "F1 IoU 0.50"), (1, "F1 IoU 0.75"), (2, "F1 small"), (3, "F1 medium"), (4, "F1 large"),
             (5, "F2 IoU 0.50"), (6, "F2 IoU 0.75"), (7, "F2 small"), (8, "F2 medium"), (9, "F2 large"))


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="COCO mAP of the PyTorch model and its ONNX against ground truth")
    p.add_argument("--onnx", required=True, help="Exported ONNX file")
    p.add_argument("-w", "--weights", required=True, help="PyTorch weights (.pth for dfine, .pt for yolo11)")
    p.add_argument("-c", "--config", help="D-FINE YAML config (required for --model dfine)")
    p.add_argument("--model", choices=["dfine", "yolo11"], default="dfine")
    p.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Input size [H,W] (default [640])")
    p.add_argument("--images", required=True, help="Folder the file_name entries of the ground truth are relative to")
    p.add_argument("--gt", required=True, help="Ground truth, COCO format JSON")
    p.add_argument("--max-images", type=int, default=0, help="Use only the first N images (0 = all)")
    p.add_argument("--slice-classes", action="store_true", help="Same flag as the export, for a wider class head")
    p.add_argument("--label-map", choices=["coco80", "identity"], default="coco80")
    p.add_argument("--category-offset", type=int, default=0, help="With --label-map identity: category id = label + N")
    p.add_argument("--eval", choices=["fdet", "coco"], default="fdet",
                   help="fdet = F1/F2 scores from the fdet-api fork (default), coco = COCO mAP from faster-coco-eval")
    p.add_argument("--min-score", type=float, default=None,
                   help="Drop detections below this score (default 0.3 for --eval fdet, 0.001 for --eval coco)")
    p.add_argument("--ort-opt", choices=list(ORT_LEVELS), default="disable",
                   help="onnxruntime graph optimization level (default disable, see verify_onnx.py)")
    args = p.parse_args()
    if args.model == "dfine" and not args.config:
        p.error("--config is required for --model dfine")
    for path in (args.onnx, args.weights, args.gt):
        if not os.path.isfile(path):
            p.error("file not found: %s" % path)
    args.img_size = args.size * 2 if len(args.size) == 1 else args.size
    if args.min_score is None:
        # mAP needs the full low-score tail, F-scores are only meaningful at the score you deploy with
        args.min_score = 0.3 if args.eval == "fdet" else 0.001
    return args


def main():
    args = parse_args()
    from faster_coco_eval import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.gt)
    images = sorted(gt.dataset["images"], key=lambda i: i["file_name"])
    if args.max_images:
        images = images[: args.max_images]
    image_ids = [i["id"] for i in images]
    label_to_category = build_label_to_category(args, gt)

    print("Loading the PyTorch reference")
    reference = build_reference(args)
    so = ort.SessionOptions()
    so.graph_optimization_level = ORT_LEVELS[args.ort_opt]
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    res_torch, res_onnx = [], []
    for n, im in enumerate(images, 1):
        path = os.path.join(args.images, im["file_name"])
        x = load_image(path, args.img_size)
        with torch.no_grad():
            d_torch = reference(x).numpy()[0]
        d_onnx = sess.run(None, {in_name: x.numpy()})[0][0]
        for d, res in ((d_torch, res_torch), (d_onnx, res_onnx)):
            d = d[d[:, 4] >= args.min_score]
            res += to_results(d, im["id"], im["width"], im["height"], args.img_size, label_to_category)
        if n % 20 == 0 or n == len(images):
            print("  %d/%d images" % (n, len(images)), flush=True)

    if args.eval == "coco":
        s_torch = coco_eval(gt, res_torch, image_ids)
        s_onnx = coco_eval(gt, res_onnx, image_ids)
        rows = COCO_ROWS
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            s_torch = fdet_eval(args.gt, res_torch, image_ids)
            s_onnx = fdet_eval(args.gt, res_onnx, image_ids)
        rows = FDET_ROWS
    n_gt = sum(1 for a in gt.dataset["annotations"] if a["image_id"] in set(image_ids))
    print("\nimages: %d, ground truth boxes: %d, onnxruntime optimization: %s" % (len(images), n_gt, args.ort_opt))
    print("evaluation: %s, detections kept (score >= %g): torch %d, onnx %d"
          % (args.eval, args.min_score, len(res_torch), len(res_onnx)))
    print("\n%-22s %10s %10s %10s" % ("", "PyTorch", "ONNX", "diff"))
    for i, name in rows:
        print("%-22s %10.4f %10.4f %+10.4f" % (name, s_torch[i], s_onnx[i], s_onnx[i] - s_torch[i]))


if __name__ == "__main__":
    main()
