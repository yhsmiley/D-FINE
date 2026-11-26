1. Build docker with `docker build -t dfine .`
1. Run docker with `bash run_docker.sh`
1. Reference commands \
    For bb, copy reference commands in `configs/dfine/custom/dfine_hgnetv2_n_sits.yml` \
    For maritime, copy reference commands in `configs/dfine/custom/dfine_hgnetv2_n_maritime.yml`
1. Verify the onnx against the original weights (needs a folder of real images, e.g. coco128) \
    Expect `PASS`, matched IoU min about 1.0 and score diff about 0. \
    Keep the default `--ort-opt disable`. The onnx file is not affected.
1. Check accuracy against ground truth (optional, needs images and a COCO format json) \
    `python3 eval_map.py --onnx <name>.onnx -w <weights> -c <config> -s <size> --images <image root> --gt <labels.json>` \
    Prints the scores for PyTorch and the onnx side by side, they should be equal to about 3 decimals. \
    `--eval fdet` (default) prints F1 and F2 scores from the fdet-api fork, on the detections above `--min-score` \
    (default 0.3, set it to the score you deploy with). `--eval coco` prints COCO mAP instead. \
    For a COCO model the classes are matched by name. For a custom model such as maritime, add `--label-map identity`.
1. Copy the onnx into sits
