# BlenderProc synthetic dataset -> FAFA training pipeline

这份说明专门对应前面提供的 **BlenderProc BOP-style 单目标数据生成脚本**。目标是把类似下面的输出：

```text
<output_root>/
  <dataset_name>/
    camera.json
    train_pbr/
      000000/
        rgb/
        depth/
        mask/
        mask_visib/
        scene_camera.json
        scene_gt.json
        scene_gt_info.json
```

直接整理成 FAFA 训练工程需要的：

- synthetic **view bank**
- synthetic **pretrain train/val** index
- optional real **selfsup train/val / eval** index
- optional real **FFT style** index
- 训练配置 `blenderproc_pretrain.yaml / blenderproc_selfsup.yaml / blenderproc_eval.yaml`

---

## 1. 新增了什么

这一版训练工程增加了几个面向 BlenderProc/BOP 数据的改动：

1. `fafa.tools.prepare_blenderproc_fafa`
   - 一条龙准备 synthetic + real 的 FAFA 输入索引和配置。

2. `fafa.tools.bop_utils.resolve_bop_scene_root`
   - 自动识别你传入的是：
     - `output_root`
     - `output_root/dataset_name`
     - `output_root/dataset_name/train_pbr`
   - 不必手工切到 scene 根目录。

3. `fafa.tools.build_bop_view_bank`
   - 深度默认读取 `scene_camera.depth_scale * translation_scale`
   - `mask_subdir=auto`，会在 `mask` 和 `mask_visib` 中自动选择存在的目录。

4. `fafa.tools.select_context_views`
   - 支持排除“样本自己当 context”的情况：
     - `--exclude-same-id`
     - `--exclude-same-frame`
   - 这对 **synthetic pretraining** 很重要，否则目标图自己很容易成为最近邻 context。

---

## 2. 一条命令准备 BlenderProc synthetic + real 数据

假设你已经有：

- BlenderProc 合成数据：`/data/blenderproc_out/rov6d_synth`
- 真实 ROV6D / BOP 数据：`/data/ROV6D/test_pool`
- CAD 点云：`/data/meshes/bluerov_points.npy`

运行：

```bash
python -m fafa.tools.prepare_blenderproc_fafa \
  --synth-root /data/blenderproc_out/rov6d_synth \
  --real-root /data/ROV6D/test_pool \
  --workdir /data/fafa_blenderproc_prepared \
  --mesh-points /data/meshes/bluerov_points.npy \
  --symmetric \
  --output-size 256 \
  --n-context 4 \
  --synth-init-rot-noise-deg 8 \
  --synth-init-trans-noise-m 0.03 \
  --real-init-rot-noise-deg 8 \
  --real-init-trans-noise-m 0.03
```

输出目录会包含：

```text
/data/fafa_blenderproc_prepared/
  prepared/
    synth_view_bank/
    synth_targets/
    real_targets_all/
  indices/
    synth_view_bank.jsonl
    synth_pretrain_targets_raw.jsonl
    synth_pretrain_all.jsonl
    synth_pretrain_train.jsonl
    synth_pretrain_val.jsonl
    real_selfsup_train.jsonl
    real_selfsup_val.jsonl
    real_eval.jsonl
    real_style.jsonl
  configs/
    blenderproc_pretrain.yaml
    blenderproc_selfsup.yaml
    blenderproc_eval.yaml
```

---

## 3. 为什么 synthetic pretrain 要默认排除 self-context

如果 synthetic target 和 synthetic context 来自同一个 render bank，最近邻通常就是它自己。这样网络会看到：

- 目标图 ≈ context 图
- gt flow ≈ 0
- refinement 退化成学 identity

这一版默认会排除：

- 相同 `id`
- 相同 `scene_id / frame_id / inst_id`

也就是默认 **不会让样本自己充当 context**。如果你确实想保留这种行为，可以加：

```bash
--allow-self-context
```

---

## 4. 关于 init pose

FAFA 是 refinement/self-adaptation 框架，不是从零直接估计 6D pose。训练时最好给 `init_pose` 一定扰动，否则 refinement 学不到真正的修正能力。

因此 `prepare_blenderproc_fafa.py` 的默认值是：

- synthetic: `8° / 0.03m`
- real: `8° / 0.03m`

如果你已经有外部初始位姿预测，也可以改成：

```bash
--synth-init-pose-source predictions --synth-init-pose-jsonl /path/to/synth_init.jsonl
--real-init-pose-source predictions  --real-init-pose-jsonl  /path/to/real_init.jsonl
```

JSONL 格式与 `build_bop_targets.py` 兼容：

```json
{"scene_id": 0, "frame_id": 12, "inst_id": 0, "pose": [[...],[...],[...],[...]]}
```

---

## 5. 训练方式

### 5.1 预训练

```bash
python -m fafa.train_pretrain \
  --config /data/fafa_blenderproc_prepared/configs/blenderproc_pretrain.yaml
```

### 5.2 自监督

```bash
python -m fafa.train_selfsup \
  --config /data/fafa_blenderproc_prepared/configs/blenderproc_selfsup.yaml
```

### 5.3 评测

```bash
python -m fafa.evaluate \
  --config /data/fafa_blenderproc_prepared/configs/blenderproc_eval.yaml \
  --checkpoint /data/fafa_blenderproc_prepared/outputs/selfsup/best_selfsup.pt
```

---

## 6. 常见注意点

### 6.1 depth 单位

如果你的 BlenderProc BOP 导出使用默认 mm 标注，这一版会优先从 `scene_camera.depth_scale` 读取深度缩放，再乘 `translation_scale=0.001` 转为米。只有在你明确传 `--raw-depth-to-meter` 时，才会覆盖它。

### 6.2 dataset root 可以怎么传

`synth-root` 和 `real-root` 可以传下面任意一种：

- `.../train_pbr`
- `.../dataset_name`
- `.../output_root`（仅当里面只有一个数据集目录时）

### 6.3 real split 与 eval split

如果没有显式传：

- `--real-train-scene-ids`
- `--real-eval-scene-ids`

脚本会按 scene 自动切分，避免 selfsup train 和 eval 完全重叠。

---

## 7. 单独使用底层脚本也可以

如果你不想用一条龙脚本，也可以继续分步运行：

1. `fafa.tools.build_bop_view_bank`
2. `fafa.tools.build_bop_targets`
3. `fafa.tools.select_context_views`
4. `fafa.tools.build_image_index`

只是对于 BlenderProc synthetic 数据，推荐优先用 `prepare_blenderproc_fafa.py`，因为它已经把：

- root 自动识别
- depth scale 处理
- self-context 排除
- synthetic / real 按 scene 切分
- 配置文件生成

这些麻烦事一起包好了。
