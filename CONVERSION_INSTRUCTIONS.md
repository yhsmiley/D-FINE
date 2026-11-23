1. Build docker with `docker build -t dfine .`
1. Run docker with `bash run_docker.sh`
1. Reference commands \
    For bb, copy reference commands in `configs/dfine/custom/dfine_hgnetv2_n_sits.yml` \
    For maritime, copy reference commands in `configs/dfine/custom/dfine_hgnetv2_n_maritime.yml`
1. Verify the onnx against the original weights (needs a folder of real images, e.g. coco128) \
    Expect `PASS`, matched IoU min about 1.0 and score diff about 0. \
    Keep the default `--ort-opt disable`. The onnx file is not affected.
1. Copy the onnx into sits
