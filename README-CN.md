<div align="center">

# RoboNix 推测解码 Toolkit

**面向视觉语言动作模型的 Drafter（草稿模型）、验证、接受与回退工具链**

[English](README.md) · [快速开始](#快速开始) · [训练](#drafter-训练) · [LIBERO Rollout](#libero-rollout) · [路线图](TODO.md)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![LIBERO](https://img.shields.io/badge/LIBERO-rollout_verified-1f9d72)

</div>

RoboNix 推测解码 Toolkit 将 KERV 整理为可以独立发布和运行的具身智能工具链。
轻量 Drafter 根据当前观测提出多个候选动作序列，目标 VLA（视觉语言动作模型）
并行验证候选；可靠候选被接受，不可靠候选被拒绝并回退到原始策略。仓库同时保留
数据生成、Drafter 训练、候选树、验证、接受与回退以及 LIBERO 评测流程。

## 架构

![推测解码架构](docs/assets/speculative-decoding-architecture.svg)

推测解码的系统收益不只取决于模型前向时间，还取决于候选接受率、候选树形状、
图像预处理、仿真执行、日志和回退开销。因此正式实验必须在相同硬件、模型和随机
种子下与自回归基线进行端到端比较。

## 已验证版本

| 验证项 | 结果 |
| --- | --- |
| 包结构与独立目录命令 | 6 项测试通过 |
| 目标模型与已有 Drafter | 成功加载为 `SpecVLAforActionPrediction` |
| LIBERO 冒烟 rollout | 任务 0，100 步上限，成功导出视频 |
| 视频 | H.264、224×224、100 帧、30 FPS |
| 训练入口 | DeepSpeed（分布式训练引擎）的模型、数据、输出和配置参数可解析 |

该 rollout 主动限制为 100 步，因此不用于证明任务成功率或复现论文指标；它证明了
目标模型加载、Drafter 挂载、仿真启动、动作生成和视频导出链路可以运行。

![已验证的 100 步 LIBERO rollout](docs/assets/validated-rollout-preview.png)

*100 步验证 rollout 的首帧、中间帧和末帧。*

## 快速开始

仓库不包含模型、数据集或输出。建议把大文件放到独立数据盘，再通过绝对路径引用。

```bash
conda create -n robonix-spec python=3.10 -y
conda activate robonix-spec
python -m pip install --upgrade pip
python -m pip install -e .

python -m pytest -q tests
python -m scripts.run --help
```

## 检查点从哪里来

| 资产 | 来源 |
| --- | --- |
| 目标 VLA | OpenVLA 官方仓库或 Hugging Face 上的 LIBERO 微调检查点 |
| Drafter | 使用本仓库的数据生成与 DeepSpeed 训练流程产生，或复用结构兼容的已有检查点 |
| LIBERO | 官方 LIBERO 源码、任务定义、初始状态和 MuJoCo/EGL 环境 |

目标 VLA 与 Drafter 必须在模型结构、词表、隐藏维度和动作编码上兼容。不要把任意
小模型检查点直接当作 Drafter 使用。

## LIBERO Rollout

```bash
export PYTHONPATH="$PWD/vendor/openvla:/path/to/LIBERO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

python -m scripts.run \
  experiments/robot/libero/run_libero_goal_Spec.py \
  --model_family openvla \
  --pretrained_checkpoint /data/checkpoints/openvla_goal \
  --spec_checkpoint /data/checkpoints/drafter_goal \
  --task_suite_name libero_goal \
  --task_ids 0 \
  --num_trials_per_task 1 \
  --max_steps_override 100 \
  --local_log_dir /data/outputs/speculative \
  --center_crop True \
  --use_wandb False
```

视频默认写入 `./rollouts/<日期>/`，日志写入 `--local_log_dir`。确认单任务、
单回合能够运行后，再移除任务和步数限制执行完整评测。

## Drafter 训练

先使用 `specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py`
生成训练样本，再运行：

```bash
cd vendor/openvla/specdecoding/train-scripts

OPENVLA_CHECKPOINT=/data/checkpoints/openvla_goal \
DRAFTER_TRAIN_DATA=/data/drafter-training-data \
DRAFTER_OUTPUT_DIR=/data/checkpoints/drafter_goal \
GPU_IDS=0,1 \
WANDB_MODE=offline \
bash train_ds_libero_goal.sh
```

部署验收不需要跑完训练，可以直接加载已经训练好的兼容 Drafter 完成 rollout。

## 目录结构

```text
.
├── modules/                  # Drafter、候选、验证、接受和策略目录
├── scripts/                  # 稳定脚本入口
├── benchmarks/libero/        # LIBERO 评测与速度测试
├── configs/                  # DeepSpeed 与模型配置
├── requirements/             # 依赖版本
├── tests/                    # 结构与独立入口测试
├── vendor/openvla/           # 权威 KERV/OpenVLA 实现
├── docs/assets/              # 架构图和 rollout 预览
└── service_bootstrap.py      # 原始代码激活与安全脚本分发
```

`vendor/openvla/` 是论文行为的权威实现，`modules/` 与 `scripts/` 提供便于服务化
和后续接入 RoboNix 的工程视图。

## 贡献者

感谢 [HuiruHe](https://github.com/HuiruHe) 和
[zhengzihaoPKU](https://github.com/zhengzihaoPKU) 对本工具包的贡献。贡献者记录
规则见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

## 引用与协议

引用信息见 `CITATION.cff`。本项目采用木兰宽松许可证第 2 版(Mulan PSL v2)，
详见 [LICENSE](LICENSE)；第三方代码继续遵循各自目录中的原始协议。
