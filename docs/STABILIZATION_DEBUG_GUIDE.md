# FAFA reproduction stabilization and debugging guide

This codebase includes a set of targeted fixes for the failure mode we observed during BlenderProc-based training:

- positive background depth leaking into geometric priors,
- unconstrained flow residuals exploding numerically,
- pose pooling over large background regions,
- self-supervision starting before the pretrain teacher is reliable.

## What changed

1. **Depth sanitation**
   - background depth is zeroed with the source mask when the dataset is loaded,
   - BlenderProc/BOP view-bank builders also zero background depth before saving cropped `.npy` depth maps.

2. **Masked geometric priors**
   - `shape_constraint_flow_from_depth(..., mask_src=...)` now supports a source mask,
   - all prior-flow and GT-flow computations in training/model code pass the source mask.

3. **Bounded flow refinement**
   - the final flow residual head is zero-initialized,
   - residual flow is bounded with `tanh` and clamped to `model.max_disp_feat` in feature space,
   - predicted flow and hidden states are masked to the object region.

4. **Mask-aware pose pooling**
   - pose regression now performs masked average pooling per context view,
   - pooled context descriptors are averaged after pooling instead of averaging misaligned spatial maps first.

5. **Safer defaults**
   - default configs start with `student_outer_iters=1`,
   - pretraining uses `pose_only_epochs` + `flow_ramp_epochs`,
   - default checkpoint selection metric is `5cm`,
   - training includes anomaly guards and dumps a debug package on failure.

6. **One-scene real-data split fallback**
   - if real data only contains one scene, the preparation script now falls back to a frame-level split so selfsup train and eval are not silently identical.

## Recommended execution order

### Step 1. Rebuild the prepared data indices

If you already prepared data with an older version of the scripts, rebuild it so the zero-background-depth fix is applied to the saved synthetic view-bank depth maps.

```bash
python -m fafa.tools.prepare_blenderproc_fafa \
  --synth-root /path/to/blenderproc_output_or_dataset_root \
  --real-root /path/to/real_bop_root \
  --workdir /path/to/fafa_prepared \
  --mesh-points /path/to/object_points.npy \
  --symmetric \
  --output-size 256 \
  --n-context 4
```

### Step 2. Inspect one batch before training

Use the new probe tool before any long run.

```bash
python -m fafa.tools.probe_refiner \
  --config /path/to/fafa_prepared/configs/blenderproc_eval.yaml \
  --checkpoint /path/to/fafa_prepared/outputs/pretrain/best_pretrain.pt \
  --mode eval
```

What you want to see:

- `flow_debug.flow_abs_max_px` within a few hundred pixels, not thousands or millions,
- `pred_metrics` not significantly worse than `init_metrics`,
- `rot_improved_ratio` and `trans_improved_ratio` above `0.5` once pretraining is stable.

### Step 3. Run pretraining only

The generated config already starts with a pose-only warmup.

```bash
python -m fafa.train_pretrain \
  --config /path/to/fafa_prepared/configs/blenderproc_pretrain.yaml
```

If training encounters exploding flow, it will dump a package under:

```text
/path/to/fafa_prepared/outputs/pretrain/debug/
```

Each package contains:

- `meta.json` with stats and the reason for the failure,
- `tensors.pt` with the offending batch and model outputs.

### Step 4. Probe the pretrain checkpoint again

```bash
python -m fafa.tools.probe_refiner \
  --config /path/to/fafa_prepared/configs/blenderproc_eval.yaml \
  --checkpoint /path/to/fafa_prepared/outputs/pretrain/best_pretrain.pt \
  --mode eval
```

Only continue if the probe shows the refiner is not degrading the init pose.

### Step 5. Start self-supervision only after the pretrain teacher is sane

```bash
python -m fafa.train_selfsup \
  --config /path/to/fafa_prepared/configs/blenderproc_selfsup.yaml
```

The selfsup config includes an early guard:

- if the teacher flow-valid ratio is too low, training aborts and writes a debug dump,
- student or teacher flow explosions also abort immediately.

## How to interpret debug dumps

### Case A: `flow_abs_max_px` exceeds the hard threshold

Check `meta.json` first. If the dump shows:

- very large `prior_flows` too -> your data/geometry priors are still wrong,
- only `flows` are huge while `prior_flows` are reasonable -> the flow head is diverging.

### Case B: teacher flow-valid ratio is near zero

Do **not** continue self-supervision. Go back to pretraining and re-run the probe on the pretrain checkpoint.

### Case C: pretrain probe says `pred` is worse than `init`

This means the refiner is still not usable. Keep selfsup disabled and inspect the pretrain debug package.

## Safe escalation after the model stabilizes

Only after the probe is healthy:

1. increase `loss.pretrain_flow_weight` from `1e-4` to `1e-3`,
2. increase `model.student_outer_iters` from `1` to `2`,
3. then, if still stable, try `student_outer_iters=4` and `teacher_outer_iters=8`.

Do **not** jump directly from an unstable run to the full paper-style iteration counts.
