# ROV6D preparation guide for the FAFA reproduction

This note adapts the generic FAFA scaffold to the **BOP-like folder structure** exposed by the public ROV6D repository.

## 1. What the public tree looks like

The linked GitHub repository exposes a `test_pool/000000/` example scene with:

- `rgb/000000.jpg`, `rgb/000001.jpg`, ...
- `mask/000000_000000.png`, `mask/000001_000000.png`, ...
- `scene_camera.json`
- `scene_gt.json`
- `scene_gt_info.json`

The crop / index builders added here were written for exactly this style of folder organization.

## 2. End-to-end preparation plan

### Real split (self-supervised / evaluation targets)

1. Build cropped target patches from the BOP scene folders.
2. Provide an `init_pose` for each target crop.
   - quick debug: use `--init-pose-source gt`
   - real self-supervision: use `--init-pose-source predictions --init-pose-jsonl ...`
3. Keep `gt_pose` for validation/evaluation.

### Synthetic split (pretraining targets + context-view bank)

1. Prepare synthetic BOP-style renders with RGB, depth, mask, pose.
2. Build the view bank with cropped RGB/depth/mask and adjusted context intrinsics.
3. Build target samples for supervised pretraining.
4. Use `select_context_views.py` to attach the nearest `N=4` views.

## 3. Real-target command

```bash
python -m fafa.tools.build_bop_targets \
  --dataset-root /data/ROV6D/test_pool \
  --output-root /data/fafa_prepared/rov6d_real_targets \
  --output-index /data/fafa_prepared/rov6d_real_targets.jsonl \
  --mesh-points /data/meshes/bluerov_points.npy \
  --symmetric \
  --crop-source bbox_visib \
  --output-size 256 \
  --translation-scale 0.001 \
  --init-pose-source gt
```

## 4. Synthetic view-bank command

```bash
python -m fafa.tools.build_bop_view_bank \
  --dataset-root /data/ROV6D_synth/train_pbr \
  --output-root /data/fafa_prepared/rov6d_view_bank \
  --output-index /data/fafa_prepared/rov6d_view_bank.jsonl \
  --crop-source mask_bbox \
  --output-size 256 \
  --translation-scale 0.001 \
  --raw-depth-to-meter 0.001
```

## 5. Attach nearest context views

```bash
python -m fafa.tools.select_context_views \
  --sample-index /data/fafa_prepared/rov6d_real_targets.jsonl \
  --view-bank /data/fafa_prepared/rov6d_view_bank.jsonl \
  --output /data/fafa_prepared/rov6d_real_prepared.jsonl \
  --n-context 4
```

## 6. FFT style-image pool

```bash
python -m fafa.tools.build_image_index \
  --root /data/fafa_prepared/rov6d_real_targets/images \
  --output /data/fafa_prepared/rov6d_real_style.jsonl
```

## 7. Pretraining targets from synthetic data

For pretraining, you can also build target crops from the synthetic BOP split:

```bash
python -m fafa.tools.build_bop_targets \
  --dataset-root /data/ROV6D_synth/train_pbr \
  --output-root /data/fafa_prepared/rov6d_synth_targets \
  --output-index /data/fafa_prepared/rov6d_synth_targets.jsonl \
  --mesh-points /data/meshes/bluerov_points.npy \
  --symmetric \
  --crop-source bbox_visib \
  --output-size 256 \
  --translation-scale 0.001 \
  --init-pose-source gt \
  --init-rot-noise-deg 10 \
  --init-trans-noise-m 0.02
```

Then attach context views exactly the same way:

```bash
python -m fafa.tools.select_context_views \
  --sample-index /data/fafa_prepared/rov6d_synth_targets.jsonl \
  --view-bank /data/fafa_prepared/rov6d_view_bank.jsonl \
  --output /data/fafa_prepared/rov6d_synth_pretrain.jsonl \
  --n-context 4
```

## 8. Init-pose prediction file format

When using `--init-pose-source predictions`, the loader expects JSONL lines like:

```json
{"scene_id": 0, "frame_id": 12, "inst_id": 0, "init_pose": [[...], [...], [...], [...]]}
```

The same format can be used for evaluation and for self-supervised real-data training.

## 9. Unit conventions

The helper scripts default to:

- `--translation-scale 0.001` for `cam_t_m2c` (mm -> m)
- `--raw-depth-to-meter 0.001` for depth images (raw depth -> m)

If your local export is already in meters, set those values to `1.0`.

## 10. Why this version adds `context.K`

The original scaffold assumed the target crop and synthetic context crops shared one common camera matrix after preprocessing. That assumption breaks easily when you crop target images and synthetic views independently.

This updated version therefore:

- stores `K` for each context view
- returns `context_ks` from the dataset
- computes shape-constraint flow with `k_src` and `k_tgt`

That keeps the synthetic-to-real geometry valid after BOP-style crop preparation.
