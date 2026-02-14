#!/bin/bash
# ============ 用 Slurm 跑 train_SIDA_naidan.py ============
# 提交： cd /project/shih/nz85/SIDA_base && mkdir -p logs && sbatch run_naidan_train.slurm
# 查状态： squeue -u $USER
# 日志：  logs/naidan_<JOBID>.out 与 logs/naidan_<JOBID>.err
# 按你集群修改： #SBATCH 的 partition / time / gres / mem / account
# ===========================================================
#SBATCH --job-name=mehi_sida
#SBATCH --output=logs/mehi_sida_%j.out
#SBATCH --error=logs/mehi_sida_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --export=ALL
#SBATCH --qos=standard
#SBATCH --account=shih

set -e
cd /project/shih/mr882/LLM/SIDA_Nad/e
mkdir -p logs

# 缓存放到项目目录，不占 home 配额
export HF_HOME=/project/shih/nz85/SIDA_base/.cache/huggingface
export TRANSFORMERS_CACHE=/project/shih/nz85/SIDA_base/.cache/huggingface
export TORCH_HOME=/project/shih/nz85/SIDA_base/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# 激活 conda 环境（路径：/project/shih/nz85/SIDA_base/env）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/shih/nz85/SIDA_base/env

# 单卡时 DeepSpeed 仍会起一个进程，master_port 避免冲突
export CUDA_VISIBLE_DEVICES=0

deepspeed train_SIDA_naidan_earlystop.py \
  #--version="/project/shih/nz85/SIDA-main/ck/SIDA-13B" \
  --version="/project/shih/nz85/SIDA-main/ck/SIDA-7B" \
  --dataset_dir="/project/shih/nz85/SIDA_base/SID_Set" \
  --vision_pretrained="/project/shih/nz85/SIDA-main/ck/sam_vit_h_4b8939.pth" \
  --val_dataset="/project/shih/nz85/SIDA_base/SID_Set" \
  --batch_size=16 \
  --exp_name="mehry_7B" \
  --epochs=100 \
  --steps_per_epoch=1000 \
  --lr=0.00001 \
  --early_stop_patience=10 \
  --early_stop_metric=combined \
  --early_stop_alpha=0.7 



