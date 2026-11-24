# Techniques used to export D-FINE and YOLO11 to ONNX opset 11

Target: an ONNX file at opset 11 that the TensorRT 7 ONNX parser (JetPack 4.5.1 / DeepStream 5.1) accepts and that
computes the same values as the PyTorch model. Both models output `[x1, y1, x2, y2, score, label]` per detection.

Upstream D-FINE exports with opset 16 or higher, because it uses `F.grid_sample`. The changes below are what was needed
to get below that, and then to get past the TensorRT 7 parser.

## Summary

| Problem | Technique | Where |
|---|---|---|
| `grid_sample` does not export below opset 16 | Replace it with a bilinear sampler built from Floor, Pad and GatherElements (`cross_attn_method: gather`) | `src/zoo/dfine/utils.py` |
| fp16 loses integer precision in the flat index `y*w+x` (above 2048) | Cast to int64 before building the index | `src/zoo/dfine/utils.py` |
| ONNX Clip only accepts integer inputs from opset 12 (`discrete` method) | Clamp before the int64 cast | `src/zoo/dfine/utils.py` |
| Batch size baked into the graph by tracing | Always call `anchors.repeat(batch, 1, 1)`, no Python `if` on the batch | `src/zoo/dfine/dfine_decoder.py` |
| TensorRT 7 rejects or miscomputes some ONNX nodes | `trt7_onnx_fix` post-pass on the exported file | `export_dfine_trt7.py` |
| Checkpoint head is wider than the config (80 outputs for 2 classes) | `--slice-classes` cuts the head down to the config's `num_classes` | `export_dfine_trt7.py` |
| A failed fix could leave a broken `.onnx` | Export to `<name>.onnx.raw`, write the final file only after the fix passes | both `*_trt7.py` scripts |
| Host driver and Python constraints | Pinned Docker environment | `Dockerfile` |
| No way to know the file is accurate | Parity check against PyTorch on real images | `verify_onnx.py` |

## 1. Model code changes

### `gather` sampler (replaces `F.grid_sample`)
`grid_sample_bilinear` in `src/zoo/dfine/utils.py` is the same as
`F.grid_sample(im, grid, mode="bilinear", padding_mode="zeros", align_corners=False)`, built from ops that exist in
opset 11:
- Compute the pixel coordinates and the four bilinear weights from the grid.
- Pad the image by 1 pixel of zeros, so any corner outside the image is clamped onto the border and reads 0.
- Gather the four neighbours with `torch.gather` on the flattened padded image.
- Cast the corner coordinates to int64 before building `y * padded_w + x`. In fp16 (for example a TensorRT fp16
  engine) that product goes above 2048 and loses integer precision.

It is selected with `cross_attn_method: gather` (also added to the assert in `dfine_decoder.py` and to the comment in
`configs/dfine/include/dfine_hgnetv2.yml`). The custom configs under `configs/dfine/custom/` set it.

Checked: on the maritime checkpoint with a random input, the gather path and the original `grid_sample` path agree to
0.0002 px in boxes and 1e-7 in scores. On coco128 the exported ONNX matches PyTorch (section 5).

### `discrete` method
The clamp now runs before the int64 cast, because ONNX Clip only takes integer inputs from opset 12. (The `discrete`
method is not the one used for the exports here, but it no longer blocks an opset 11 export.)

### Anchors repeat
`anchors.repeat(memory.shape[0], 1, 1)` is now unconditional. The old `if memory.shape[0] > 1` was resolved at trace
time, so the exported graph ignored the batch size.

## 2. TensorRT 7 post-pass (`trt7_onnx_fix`)

Applied to the exported (and optionally `onnxslim`-simplified) ONNX. Every rewrite is meant to be equivalent to the
original graph.

| Node | Rewrite | Reason |
|---|---|---|
| `Constant` | Move into initializers | Lets the other patterns match them |
| `Pad` with an empty optional input | Drop the empty input | TensorRT 7 needs constants or nothing |
| `Clip` with an empty min or max | Fill with the largest or smallest value of the data type | Same |
| `Gather` with a scalar index | Use a `[1]` index and a `Squeeze`. A negative index is resolved to `idx + dim` | TensorRT 7 keeps the gathered axis for 0-d indices, and returns zeros for negative indices |
| `MatMul` with a 1-D constant right operand | Use a `[K,1]` operand and a `Squeeze` | TensorRT 7 returns rank 0 |
| `Split` with several outputs | One `Slice` per output | TensorRT 7 gives every slice of a Split the node name and fails with "repeated layer name" |

Notes:
- It refuses opset above 12, because the `Squeeze` rewrite relies on the `axes` attribute.
- Resolving negative Gather indices needs static shapes, so export with `--simplify`.
- It prints a warning for each rewrite it had to skip (unknown shape or type).
- `export_yolo11_trt7.py` imports and uses the same function.
- On the COCO D-FINE-N export it rewrote 3 Pad, 2 Clip, 30 Gather, 3 MatMul and 8 Split nodes.
- This is meant to be kept in sync with `utils/trt7_onnx_fix.py` in DeepStream-Yolo.

## 3. Export scripts

- `export_dfine.py`: D-FINE to ONNX with a DeepStream-style output head. No TensorRT 7 fix. Default opset 17.
- `export_dfine_trt7.py`: the same, plus `trt7_onnx_fix`, `--slice-classes`, default opset 11, and the safe
  `.onnx.raw` write.
- `export_yolo11_trt7.py`: YOLO11 export with `C2f.forward = forward_split` (feeds the Split rewrite),
  `Detect.export = True`, `Detect.dynamic = False`, and the same post-pass.
- Common to the D-FINE scripts: `HGNetv2.pretrained` is forced to `False` (the config default is `True`, which
  would try to download backbone weights), the EMA weights are preferred, and `torch.load(..., weights_only=False)` is
  used. Do not use that on untrusted checkpoints.

### `--slice-classes`
For a checkpoint trained with a wider class head than the config, for example 80 outputs for 2 real classes. It keeps
classes `0..N-1` and the last row of `denoising_class_embed` as the padding row. This assumes the labels used in
training were exactly `0..N-1`, which is a heuristic and not something the script can check.

## 4. Docker environment

- Base: `nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04`.
- Python 3.11 from deadsnakes, because Ubuntu 22.04's 3.11.0rc1 breaks torch.
- `torch==2.4.1` and `torchvision==0.19.1` from the cu121 index, to match the host driver (530.x) and keep the legacy
  ONNX exporter as the default.
- `tensorrt-cu12==10.3.0` (supports CUDA 12.0 to 12.5). The retry settings are passed as `PIP_*` env vars, because the
  nested pip for the 2 GB wheel ignores CLI flags.
- `onnxruntime-gpu==1.24.4`. Version 1.30 needs CUDA 13, which the driver cannot run.
- `onnx`, `onnxslim` (unpinned) and `ultralytics` (for the YOLO export).
- pip retry and resume flags, for flaky networks.

## 5. Verification

`verify_onnx.py` runs the ONNX through onnxruntime and the PyTorch model (with the original `grid_sample` path) on
real images, and compares detections by label and IoU. See `CONVERSION_INSTRUCTIONS.md` for the command.

Results (CPU, onnxruntime 1.24.4, optimizations off, fresh opset 11 exports):

| Model | Images | Result |
|---|---|---|
| D-FINE-N COCO, 640 | all 128 coco128 images | 1960 vs 1960 detections, none unmatched, IoU min 1.0000, score diff at most 0.0001 |
| D-FINE-N maritime with `--slice-classes`, 1280 | all 128 coco128 images, score 0.1 | 3426 vs 3426 detections, none unmatched, IoU min 1.0000, score diff at most 0.0001 |
| YOLO11n maritime, 1280 | all 128 coco128 images, score 0.1 | 1110 vs 1110 detections, none unmatched, IoU min 1.0000, score diff 0.0000 |

Isolating the changes showed that the plain opset 11 export without the post-pass (`export_dfine.py`) and the
opset 17 `grid_sample` export both match PyTorch exactly, so the `gather` sampler is correct.

### Known issue: onnxruntime Pad/MaxPool fusion
With onnxruntime's graph optimization on (the default), the post-pass export gives different results. On all 128 coco128
images at `--ort-opt basic`, `verify_onnx.py` FAILS: 11 of 1847 PyTorch detections have no match in the ONNX, 15 of
1846 ONNX detections have no match in PyTorch, and the worst matched score differs by 0.45. Most detections are
unaffected (median IoU 0.9989), but some boxes and scores move. (A 32-image subset looked much milder, with IoU of at
least 0.947 and score differences up to 0.04, so do not judge this on a small sample.) The cause is an onnxruntime
1.24.4 bug, reproduced on a two-node model: it fuses a `Pad` of constant 0 into a following `MaxPool`, which is wrong
(a MaxPool ignores its own padding, a Pad of 0 does not). The HGNetv2 stem has exactly that pattern.
- The optimizer is already at fault at `ENABLE_BASIC`. Only `ORT_DISABLE_ALL` avoids it.
- Dropping the empty Pad input (the TensorRT 7 rewrite) and giving the Pad an explicit constant 0 behave the same.
  Only the original empty `''` input stops onnxruntime from fusing.
- The ONNX file is correct. Any onnxruntime inference of it should set
  `graph_optimization_level = ORT_DISABLE_ALL`. `verify_onnx.py` does this by default.

## 6. Not verified

- The TensorRT 7 engine. All checks above are against onnxruntime, so whether the Pad, Clip, Gather, MatMul and Split
  rewrites are enough for the TensorRT 7 parser, and give the same values there, still has to be checked on the
  Jetson.
- Accuracy against ground truth (mAP). The checks compare the ONNX with the PyTorch model, not with labels.
- The maritime and YOLO11 models on maritime images (the checks used coco128 images).
- Dynamic batch size.
