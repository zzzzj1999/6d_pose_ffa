export FAFA_REPO=/home/dongnan/zzj/6d_pose/fafa_repro-4

# 你的 BlenderProc synthetic 数据根目录
# 可以指向 output_root / dataset_name / train_pbr，脚本会自动识别
export SYNTH_ROOT=/home/dongnan/zzj/6d_pose/train_pbr

# 真实数据根目录（如果你要跑完整 FAFA，就填真实 BOP/ROV6D 数据）
# 如果你现在只有 synthetic，这个先别填
export REAL_ROOT=/home/dongnan/zzj/6d_pose/ROV6D/test_pool

# 你的 CAD mesh
export MESH=/home/dongnan/zzj/6d_pose/blenderproc_bop_synth/data/BlueROV2C.obj

# 采样后的点云
export MESH_POINTS=/home/dongnan/zzj/6d_pose/fafa_repro-4/mesh/bluerov_points_m.npy

# FAFA 准备后的工作目录
export WORKDIR=/home/dongnan/zzj/6d_pose/fafa_repro-4/fafa_prepared_rov6d

cd $FAFA_REPO


export FAFA_REPO=/home/dongnan/zzj/6d_pose/fafa_repro-4
export SYNTH_ROOT=/home/dongnan/zzj/6d_pose/blenderproc_bop_synth/data/rov6d_bluerov_synth/train_pbr
export REAL_ROOT=/home/dongnan/zzj/6d_pose/ROV6D/test_pool
export MESH=//home/dongnan/zzj/6d_pose/fafa_repro-4/mesh/BlueROV2C.obj
export MESH_POINTS=/home/dongnan/zzj/6d_pose/fafa_repro-4/mesh/bluerov_points_m.npy
export WORKDIR=/home/dongnan/zzj/6d_pose/fafa_repro-4/fafa_prepared_rov6d
cd $FAFA_REPO

python -m fafa.tools.sample_mesh_points --mesh $MESH --num-points 2048 --output $MESH_POINTS

python -m fafa.tools.prepare_blenderproc_fafa --synth-root $SYNTH_ROOT --real-root $REAL_ROOT --workdir $WORKDIR --mesh-points $MESH_POINTS --symmetric --output-size 256 --n-context 4 --translation-scale 0.001 --synth-init-rot-noise-deg 8 --synth-init-trans-noise-m 0.03 --real-init-rot-noise-deg 8 --real-init-trans-noise-m 0.03

find $WORKDIR -maxdepth 3 | sort | head -200
python -m fafa.tools.check_index --index $WORKDIR/indices/synth_pretrain_train.jsonl --check-files
python -m fafa.tools.check_index --index $WORKDIR/indices/real_selfsup_train.jsonl --check-files
python -m fafa.tools.check_index --index $WORKDIR/indices/real_eval.jsonl --check-files

cd $FAFA_REPO
python tests/smoke_test.py


Stage 1：synthetic supervised pretrain:
cd $FAFA_REPO
mkdir -p $WORKDIR/logs

python -m fafa.train_pretrain --config $WORKDIR/configs/blenderproc_pretrain.yaml 2>&1 | tee $WORKDIR/logs/pretrain.log

python -m fafa.train_selfsup --config $WORKDIR/configs/blenderproc_selfsup.yaml 2>&1 | tee $WORKDIR/logs/selfsup.log

-------------------------------------------check-------------------------------------------------
export FAFA_REPO=/home/dongnan/zzj/6d_pose/fafa_repro-4
export SYNTH_ROOT=/home/dongnan/zzj/6d_pose/blenderproc_bop_synth/data/rov6d_bluerov_synth/train_pbr
export REAL_ROOT=/home/dongnan/zzj/6d_pose/ROV6D/test_pool
export MESH=/home/dongnan/zzj/6d_pose/fafa_repro-4/mesh/BlueROV2C.obj
export MESH_POINTS=/home/dongnan/zzj/6d_pose/fafa_repro-4/mesh/bluerov_points_m.npy
export WORKDIR=/home/dongnan/zzj/6d_pose/fafa_repro-4/fafa_prepared_rov6d

export PRETRAIN_CFG=$WORKDIR/configs/blenderproc_pretrain.yaml
export SELFSUP_CFG=$WORKDIR/configs/blenderproc_selfsup.yaml
export EVAL_CFG=$WORKDIR/configs/blenderproc_eval.yaml

export PRETRAIN_CKPT=$WORKDIR/outputs/pretrain/best_pretrain.pt
export SELFSUP_CKPT=$WORKDIR/outputs/selfsup/best_selfsup.pt

export PRETRAIN_LOG=/home/dongnan/zzj/6d_pose/fafa_repro-4/fafa_prepared_rov6d/logs/pretrain.log
export SELFSUP_LOG=/home/dongnan/zzj/6d_pose/fafa_repro-4/fafa_prepared_rov6d/logs/selfsup.log

export DIAG_DIR=$WORKDIR/diag_$(date +%Y%m%d_%H%M%S)

----------------------------check2-----------------------------------
export FAFA_REPO=/home/dongnan/zzj/6d_pose/fafa_repro-4
export WORKDIR=/home/dongnan/zzj/6d_pose/fafa_repro-4/fafa_prepared_rov6d
export PRETRAIN_CFG=$WORKDIR/configs/blenderproc_pretrain.yaml
export EVAL_CFG=$WORKDIR/configs/blenderproc_eval.yaml
export PRETRAIN_CKPT=$WORKDIR/outputs/pretrain/best_pretrain.pt
export DIAG2_DIR=$WORKDIR/diag_fncheck_$(date +%Y%m%d_%H%M%S)

----------------------------code2-----------------------------------
export FAFA_REPO=/home/dongnan/zzj/6d_pose/fafa_repro-5
export SYNTH_ROOT=/home/dongnan/zzj/6d_pose/blenderproc_bop_synth/data/rov6d_bluerov_synth/train_pbr
export REAL_ROOT=/home/dongnan/zzj/6d_pose/ROV6D/test_pool
export MESH=/home/dongnan/zzj/6d_pose/fafa_repro-4/mesh/BlueROV2C.obj
export MESH_POINTS=/home/dongnan/zzj/6d_pose/fafa_repro-4/mesh/bluerov_points_m.npy
export WORKDIR=/home/dongnan/zzj/6d_pose/fafa_repro-5/fafa_prepared_rov6d

cd $FAFA_REPO

python -m fafa.tools.prepare_blenderproc_fafa --synth-root $SYNTH_ROOT --real-root $REAL_ROOT --workdir $WORKDIR --mesh-points $MESH_POINTS --symmetric --output-size 256 --n-context 4

python -m fafa.train_pretrain --config $WORKDIR/configs/blenderproc_pretrain.yaml 2>&1 | tee $WORKDIR/pretrain_fixed.log

python -m fafa.tools.probe_refiner --config $WORKDIR/configs/blenderproc_eval.yaml --checkpoint $WORKDIR/outputs/pretrain/best_pretrain.pt --mode eval

python -m fafa.train_selfsup --config $WORKDIR/configs/blenderproc_selfsup.yaml 2>&1 | tee $WORKDIR/selfsup_fixed.log

----------------------------------------vis-----------------------------------------


export WORKDIR=/home/dongnan/zzj/6d_pose/fafa_repro-5/fafa_prepared_rov6d
export REAL_BOP_ROOT=/home/dongnan/zzj/6d_pose/ROV6D/test_pool
export CKPT=$WORKDIR/outputs/selfsup/best_selfsup.pt
python -m fafa.tools.visualize_pose_overlay --config $WORKDIR/configs/blenderproc_eval.yaml --checkpoint $CKPT --output-dir $WORKDIR/vis_full_rov --mode eval --render-space full --bop-root $REAL_BOP_ROOT --num-samples 50 --save-jsonl



