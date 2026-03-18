# FAFA 复现说明

## 1. 论文方法到代码模块的映射

### 1.1 两阶段训练

- 预训练：`fafa/train_pretrain.py`
  - 合成图监督训练
  - 使用 `fafa/fft.py` 的 FFT 数据增强
- 自监督：`fafa/train_selfsup.py`
  - teacher / student 框架
  - teacher 由 EMA 更新
  - 使用图像层、特征层、位姿层联合损失

### 1.2 主干网络

- `fafa/modeling/encoder.py`
  - 特征编码器
- `fafa/modeling/flow_regressor.py`
  - RAFT 风格轻量光流回归器
- `fafa/modeling/pose_regressor.py`
  - 从隐藏状态和流场回归位姿增量
- `fafa/modeling/fafa_net.py`
  - 迭代细化总流程

### 1.3 Shape-constraint flow

- `fafa/geometry/projection.py`
  - `shape_constraint_flow_from_depth(...)`
  - 用 synthetic depth + source pose + current pose + K 构造解析几何流

### 1.4 损失函数

- `fafa/losses/core.py`
  - `flow_supervision_loss`
  - `photometric_consistency_loss`
  - `warp_mask_consistency_loss`
  - `feature_level_loss`
  - `point_matching_loss`
  - `self_supervised_loss`
- `fafa/losses/census.py`
  - Census photometric loss

---

## 2. 论文中的关键超参数

配置文件已按论文写入：

- `N = 4`
- `delta0 = 0.5`
- `beta = 1`
- teacher iteration = `8`
- student iteration = `4`
- EMA = `0.999`
- `gamma1 = 0.1`
- `gamma2 = 0.1`
- `gamma3 = 10`
- `gamma4 = 10`
- `batch_size = 16`
- `max_lr = 4e-4`
- self-supervision 阶段冻结 BN

---

## 3. 复现时做出的显式工程假设

### 3.1 特征层损失
论文正文只说明 `L_feat-level` 是 warped real feature 与 synthetic feature 的加权差异，但没有公开完整闭式表达式。

本复现中实现为：

- `warp(real_feat, flow_student)` 与 `synthetic_feat` 的加权鲁棒 L1 差异。

### 3.2 位姿平移更新
论文说明回归 decoupled `[RΔ | tΔ]`，但未公开完整平移参数化。

本复现中实现为：

- 旋转：6D rotation representation
- 平移：加性更新，带可配置缩放因子

### 3.3 上下文渲染
论文中是在 `P0` 周围渲染 `N` 张 synthetic image。

本复现为了保证训练代码清晰可靠，采用：

- 先离线渲染 synthetic bank
- 再通过 `fafa/tools/select_context_views.py` 为每个样本选择最近的 `N=4` 个上下文视角

---

## 4. 推荐复现实验顺序

1. 用 CAD 网格采样点云：`sample_mesh_points.py`
2. 构建 synthetic render bank
3. 生成带 context 的 JSONL 索引
4. 跑 `tests/smoke_test.py`
5. 跑 1 个 epoch 的 `train_pretrain.py`
6. 跑 1 个 epoch 的 `train_selfsup.py`
7. 跑 `evaluate.py`
8. 再扩展到完整数据集和更长 schedule

---

## 5. 本次针对 ROV6D 目录结构补充的适配

你提供的 ROV6D 链接是 BOP 风格目录，因此本版额外补充了：

- `fafa/tools/build_bop_targets.py`
  - 从 `rgb/ + scene_camera.json + scene_gt.json + scene_gt_info.json` 生成目标样本 JSONL
  - 自动裁剪目标 patch
  - 自动更新裁剪后的相机内参 `K`
  - 支持 `init_pose = gt` 或外部预测姿态 JSONL

- `fafa/tools/build_bop_view_bank.py`
  - 从 BOP 风格 synthetic 数据生成 FAFA 所需的 context view bank
  - 输出 `image / depth / mask / pose / K`

- `fafa/tools/build_image_index.py`
  - 为 FFT pretraining 生成 real-style 图像池索引

### 5.1 为什么新增 `context.K`

原始 scaffold 假设 target crop 与所有 synthetic crop 共享同一个内参矩阵，这在真实的 BOP/ROV6D 裁剪流程里通常不够稳妥。

因此本次代码把几何流计算扩展为：

- source synthetic intrinsics: `k_src`
- target crop intrinsics: `k_tgt`

对应改动包括：

- `PreparedContextPoseDataset` 输出 `context_ks`
- `shape_constraint_flow_from_depth(...)` 支持 `k_src, k_tgt`
- `FAFANet` / `train_pretrain.py` / `train_selfsup.py` / `evaluate.py` 自动传递 `context_ks`

这样就可以安全支持：

- real crop 和 synthetic crop 分别独立裁剪
- 但仍然保持几何投影和 flow 计算正确

### 5.2 单位约定

ROV6D / BOP 风格 JSON 常见的 `cam_t_m2c` 和 depth 通常不是以米存储。

因此工具脚本默认：

- `translation_scale = 0.001`
- `raw_depth_to_meter = 0.001`

如果你的本地导出已经是米制，需要把这两个参数改为 `1.0`。
