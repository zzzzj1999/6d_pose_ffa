> **Stabilized debug version:** this tree includes additional safeguards for BlenderProc-based training (zero-background depth sanitation, bounded flow residuals, mask-aware pose pooling, anomaly dumps, and the `probe_refiner` tool). See `docs/STABILIZATION_DEBUG_GUIDE.md` before starting a new run.

# FAFA reproduction (paper-faithful training scaffold)

This repository is a **paper-faithful reproduction scaffold** for:

> **FAFA: Frequency-Aware Flow-Aided Self-Supervision for Underwater Object Pose Estimation**

It reproduces the method described in the paper as a clean PyTorch training codebase with:

- stage 1: **supervised synthetic pre-training** with **FFT-based augmentation**
- stage 2: **teacher-student self-supervised fine-tuning** on **unlabeled real RGB images**
- a **flow-aided pose network** with **shape-constraint flow**, **image-level alignment**, **feature-level alignment**, and **point-matching pose loss**
- evaluation code for **ADD / ADD-S** and **5° / 5cm** style metrics

The code is organized to be easy to read and easy to adapt to ROV6D, DeepURL, or your own CAD-based RGB pose dataset.

---

## 1. What is reproduced here

The public paper describes the following core ingredients, and this code mirrors them directly:

1. **Two-stage training**
   - synthetic supervised pre-training
   - self-supervised adaptation on unlabeled real underwater images

2. **FFT-based augmentation in pre-training only**
   - amplitude mix between synthetic and real images
   - amplitude dropout branch defined exactly as the paper's Eq. (2), i.e. amplitude is set to `1`

3. **Teacher / student self-supervision**
   - same architecture for both networks
   - teacher updated by EMA from the student
   - teacher runs more refinement iterations than the student

4. **Flow-aided pose refinement**
   - feature encoder
   - flow regressor
   - pose regressor
   - shape-constraint flow computed from current pose + synthetic depth + camera intrinsics

5. **Multi-level self-supervision**
   - `L_flow`
   - `L_photo`
   - `L_warp-mask`
   - `L_feat-level`
   - `L_pose`

---

## 2. What is intentionally explicit / approximate

This repo is **faithful to the paper**, but it is **not guaranteed to be byte-for-byte identical to the authors' internal training code**. The public PDF leaves a few implementation details underspecified.

### Explicit implementation choices

1. **Feature-level loss**
   The paper states that `L_feat-level` is the weighted average of the feature dissimilarity between warped real features and synthetic features, but the public PDF does **not** give a full closed-form equation. In this reproduction, it is implemented as a **weighted robust L1-style feature matching loss** between:

   - `warp(real_feat, flow_student)`
   - `synthetic_feat`

2. **Translation update parameterization**
   The paper says a decoupled relative pose `[RΔ | tΔ]` is predicted, but does not fully specify the exact translation parameterization in the public PDF. In this reproduction, pose is updated by:

   - left-multiplicative rotation update from a continuous 6D rotation representation
   - additive translation update with a configurable scale

3. **Offline synthetic context preparation**
   The paper renders `N` synthetic views around the current pose. For reproducibility and engineering simplicity, this repo assumes you prepare a **bank of rendered context views offline**, then select the nearest `N` views around each initial pose using the provided helper script.

4. **Training epochs**
   The public PDF gives optimizer type and major hyperparameters, but not the full epoch schedule. The configs therefore use **reasonable reproduction defaults** that you can tune.

---

## 3. Hyperparameters copied from the paper

The following settings are already encoded in the default configs because they are explicitly stated in the paper:

- crop / patch size: **256 × 256**
- number of synthetic context views: **N = 4**
- FFT augmentation only during pre-training
- `delta0 = 0.5`
- `beta = 1.0`
- teacher refinement iterations: **8**
- student refinement iterations: **4**
- EMA factor: **0.999**
- self-supervision weights:
  - `gamma1 = 0.1`
  - `gamma2 = 0.1`
  - `gamma3 = 10`
  - `gamma4 = 10`
- optimizer: **AdamW**
- learning-rate schedule: **OneCycle**
- max learning rate: **4e-4**
- batch size: **16**
- freeze BN during self-supervised training

---

## 4. Repository structure

```text
fafa/
  common.py
  fft.py
  train_pretrain.py
  train_selfsup.py
  evaluate.py

  data/
    dataset.py
    augment.py

  geometry/
    pose.py
    projection.py
    warp.py
    metrics.py

  losses/
    census.py
    core.py

  modeling/
    encoder.py
    flow_regressor.py
    pose_regressor.py
    fafa_net.py
    ema.py

  tools/
    select_context_views.py
    prepare_blenderproc_fafa.py
    sample_mesh_points.py
    check_index.py

configs/
  pretrain.yaml
  selfsup.yaml
  eval.yaml
```

---

## 5. Dataset contract

This code expects a **prepared JSONL index**. Each sample should already contain:

- the target crop (`image`)
- camera intrinsics (`K`)
- an initial pose (`init_pose`)
- optional ground-truth pose (`gt_pose`) for supervised pre-training / validation
- a mesh point cloud file (`mesh_points`) used for point matching and evaluation
- a list of `context` synthetic renderings

### Example sample

```json
{
  "id": "000123",
  "image": "real_crops/000123.png",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "init_pose": [[...], [...], [...], [...]],
  "gt_pose": [[...], [...], [...], [...]],
  "mesh_points": "meshes/bluerov_points.npy",
  "symmetric": true,
  "context": [
    {
      "image": "renders/000123/view_00_rgb.png",
      "depth": "renders/000123/view_00_depth.npy",
      "mask": "renders/000123/view_00_mask.png",
      "pose": [[...], [...], [...], [...]]
    },
    {
      "image": "renders/000123/view_01_rgb.png",
      "depth": "renders/000123/view_01_depth.npy",
      "mask": "renders/000123/view_01_mask.png",
      "pose": [[...], [...], [...], [...]]
    }
  ]
}
```

### Real style index for FFT pre-training

For stage-1 FFT augmentation, prepare a separate JSONL with records like:

```json
{"image": "real_pool/frame_0001.png"}
{"image": "real_pool/frame_0002.png"}
```

These images are used **only as style sources** for amplitude mixing in pre-training.

---

## 6. Data preparation workflow

### Step A: sample mesh points from your CAD model

```bash
python -m fafa.tools.sample_mesh_points \
  --mesh /path/to/object.obj \
  --num-points 2048 \
  --output /path/to/object_points.npy
```

### Step B: build a synthetic render bank

You need a bank of rendered synthetic views, each with:

- RGB
- depth
- mask
- pose

The paper mentions BlenderProc for synthetic data generation on DeepURL. This repo does not hard-code a renderer so that you can use BlenderProc, PyTorch3D, Kaolin, or your existing rendering pipeline.

### Step C: select the nearest `N=4` context views around each init pose

```bash
python -m fafa.tools.select_context_views \
  --sample-index /path/to/raw_samples.jsonl \
  --view-bank /path/to/view_bank.jsonl \
  --output /path/to/prepared_samples.jsonl \
  --n-context 4
```

### Step D: sanity-check the index

```bash
python -m fafa.tools.check_index \
  --index /path/to/prepared_samples.jsonl \
  --check-files
```

---

## 7. Training

### Stage 1: supervised synthetic pre-training

```bash
python -m fafa.train_pretrain --config configs/pretrain.yaml
```

What happens in this stage:

- synthetic crop is FFT-augmented using a random real image
- network predicts dense flow and refined pose
- supervision uses:
  - point-matching pose loss
  - dense flow supervision against geometry-derived ground-truth flow

### Stage 2: teacher-student self-supervision

```bash
python -m fafa.train_selfsup --config configs/selfsup.yaml
```

What happens in this stage:

- teacher sees clean real crop
- student sees noisy real crop
- both compare against the same prepared synthetic context views
- teacher output becomes pseudo supervision
- losses combine:
  - flow consistency
  - photometric consistency with Census loss
  - warped-mask consistency
  - feature-level alignment
  - point-matching pose loss against teacher pose pseudo-label
- teacher updated by EMA from the student

---

## 8. Evaluation

```bash
python -m fafa.evaluate \
  --config configs/eval.yaml \
  --checkpoint /path/to/outputs/selfsup/best_selfsup.pt
```

Metrics returned:

- `ADD(-S)_0.1d`
- `5deg`
- `5cm`
- mean rotation error (deg)
- mean translation error (cm)
- mean ADD / ADD-S distance (m)

---

## 9. Main implementation notes

### `fft.py`
Implements the paper's Fourier augmentation:

- amplitude / phase extraction with `torch.fft`
- amplitude mix with random real-image amplitude
- dropout branch that sets amplitude to `1`
- inverse FFT reconstruction using the synthetic phase

### `geometry/projection.py`
Implements analytic **shape-constraint flow** from:

- synthetic depth
- source synthetic pose
- current target pose estimate
- camera intrinsics

This is the central bridge between pose and flow.

### `modeling/flow_regressor.py`
Uses a compact RAFT-inspired recurrent head:

- synthetic feature
- warped real feature
- feature difference
- geometry prior flow

These are fused and recurrently refined into the predicted flow.

### `modeling/fafa_net.py`
This is the outer FAFA loop:

1. encode real / synthetic images
2. compute analytic shape prior flow from current pose
3. refine flow with the flow regressor
4. aggregate hidden states / flows across context views
5. regress pose delta
6. update pose
7. repeat

### `losses/core.py`
Contains all training losses in one place.

---

## 10. Important limitations

1. This repo is a **reproduction scaffold**, not the official released code.
2. Some paper details are under-specified in the public PDF, especially the exact formula for `L_feat-level` and the exact full training schedule.
3. The code assumes you already have **prepared synthetic depth + mask + pose** for the context views.
4. For best results, you should align your rendering pipeline, pose initialization pipeline, and crop-generation procedure to your target dataset.

---

## 11. Recommended first run

For a first end-to-end smoke run:

1. prepare a tiny toy dataset with 20 samples
2. render 8–16 context views per sample
3. select the nearest 4 views
4. run pre-training for 1 epoch
5. run self-supervision for 1 epoch
6. run evaluation

This will validate your full pipeline before you scale up to ROV6D or DeepURL.

---

## 12. Quick smoke test

```bash
python tests/smoke_test.py
```

If it prints `Smoke test passed.`, the core forward path and loss composition are working.


---

## 13. ROV6D / BOP-style adapter added in this version

If your data follows the BOP-like layout used by the public ROV6D repository, you can now build the FAFA JSONL indexes directly.

### What the new tools do

- `fafa.tools.build_bop_targets`
  - reads scene folders like `000000/rgb`, `scene_camera.json`, `scene_gt.json`, `scene_gt_info.json`
  - crops each object instance to a fixed patch size
  - adjusts camera intrinsics after crop+resize
  - writes a FAFA target-sample JSONL with `image`, `K`, `init_pose`, optional `gt_pose`

- `fafa.tools.build_bop_view_bank`
  - reads synthetic BOP-style renders with RGB + depth + mask + pose
  - crops RGB / depth / mask consistently
  - stores **per-context intrinsics `K`** in the view bank

- `fafa.tools.build_image_index`
  - builds the FFT style-image pool JSONL from a directory of real RGB images

### Example: prepare a real ROV6D split for self-supervision / evaluation

```bash
python -m fafa.tools.build_bop_targets   --dataset-root /data/ROV6D/test_pool   --output-root /data/fafa_prepared/rov6d_real_targets   --output-index /data/fafa_prepared/rov6d_real_targets.jsonl   --mesh-points /data/meshes/bluerov_points.npy   --symmetric   --crop-source bbox_visib   --output-size 256   --init-pose-source gt
```

For a true self-supervised setting, replace `--init-pose-source gt` with `--init-pose-source predictions --init-pose-jsonl /path/to/init_predictions.jsonl`.

### Example: prepare a synthetic render bank

```bash
python -m fafa.tools.build_bop_view_bank   --dataset-root /data/ROV6D_synth/train_pbr   --output-root /data/fafa_prepared/rov6d_view_bank   --output-index /data/fafa_prepared/rov6d_view_bank.jsonl   --crop-source mask_bbox   --output-size 256
```

### Example: choose the nearest 4 synthetic context views

```bash
python -m fafa.tools.select_context_views   --sample-index /data/fafa_prepared/rov6d_real_targets.jsonl   --view-bank /data/fafa_prepared/rov6d_view_bank.jsonl   --output /data/fafa_prepared/rov6d_real_prepared.jsonl   --n-context 4
```

### Example: build the FFT real-style pool

```bash
python -m fafa.tools.build_image_index   --root /data/fafa_prepared/rov6d_real_targets/images   --output /data/fafa_prepared/rov6d_real_style.jsonl
```

### Why `context.K` matters

The earlier scaffold assumed the target crop and all synthetic context crops shared the same camera intrinsics after preprocessing. That is often too strict for real BOP/ROV6D pipelines because target crops and synthetic crops may be generated from different windows. This version therefore supports **per-context intrinsics**:

- each synthetic context record may now carry its own `K`
- `shape_constraint_flow_from_depth(...)` now supports `k_src` and `k_tgt`
- the dataset loader returns `context_ks`
- training / evaluation forward passes now use those intrinsics automatically

This makes the crop-preparation pipeline much safer for ROV6D-style data.


## 14. BlenderProc synthetic dataset adapter

If you used the BlenderProc BOP-style generator provided earlier in this conversation, this repo now includes a direct preparation pipeline:

```bash
python -m fafa.tools.prepare_blenderproc_fafa \
  --synth-root /path/to/blenderproc_output_or_dataset_root \
  --real-root /path/to/real_bop_root \
  --workdir /path/to/fafa_prepared \
  --mesh-points /path/to/object_points.npy \
  --symmetric
```

It will:

- auto-detect whether `--synth-root` points to `output_root`, `dataset_name`, or `train_pbr`
- build the synthetic **view bank**
- build synthetic **pretrain train/val** indexes
- optionally build real **selfsup train/val / eval** indexes
- write ready-to-run configs under `workdir/configs/`

See:

- `docs/BLENDERPROC_FAFA_PIPELINE.md`
- `fafa/tools/prepare_blenderproc_fafa.py`

One important behavior change in this version is that synthetic pretraining can now exclude the identical rendered view from the context bank. This avoids the common failure mode where a target crop selects itself as the nearest synthetic context and the refinement task collapses into learning the identity map.
