<div align="center">

# RoboNix 推测解码 Toolkit

**面向视觉语言动作模型的 Drafter（草稿模型）、验证、接受与回退工具链**

[English](README.md) · [🚀 快速开始](#quick-start) · [⚙️ 环境要求](#requirements) · [🧪 验证结果](#validated-release) · [📝 引用](#citation)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![LIBERO](https://img.shields.io/badge/LIBERO-rollout_verified-1f9d72)
[![License](https://img.shields.io/badge/license-MulanPSL--2.0-red)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/lusunn111/RoboNix-Speculative-Decoding-Toolkit?style=flat&logo=github)](https://github.com/lusunn111/RoboNix-Speculative-Decoding-Toolkit/stargazers)

</div>

RoboNix 推测解码 Toolkit 将 KERV 整理为可以独立发布和运行的具身智能工具链。
轻量 Drafter 根据当前观测提出多个候选动作序列，目标 VLA（视觉语言动作模型）
并行验证候选；可靠候选被接受，不可靠候选被拒绝并回退到原始策略。仓库同时保留
数据生成、Drafter 训练、候选树、验证、接受与回退以及 LIBERO 评测流程。

## 📚 目录

- [📰 最新进展](#news)
- [🧠 架构总览](#architecture)
- [🧪 已验证版本](#validated-release)
- [⚙️ 环境要求](#requirements)
- [🚀 快速开始](#quick-start)
- [📦 检查点来源](#checkpoints)
- [🎬 LIBERO Rollout](#rollout)
- [🏋️ Drafter 训练](#training)
- [🗺️ 路线图](#roadmap)
- [📝 引用](#citation)
- [🤝 贡献者](#contributors)
- [📄 协议](#license)

<a id="news"></a>
## 📰 最新进展

- **2026-07-19**：🆕 重构中英文开源文档，将进展、依赖、路线图和引用统一
  收入 README，并更新架构总览。
- **2026-07-18**：🔥 完成独立目录运行验证，成功加载目标模型与已有 Drafter，
  并导出 100 步 LIBERO H.264 rollout 视频。
- **2026-07-18**：🛠️ 开放 DeepSpeed 路径、任务选择和 rollout 步数上限配置。

<a id="architecture"></a>
## 🧠 架构总览

<!--
IMAGEGEN SLOT
最终图片：docs/assets/speculative-decoding-overview-v2.png
生成提示词：docs/assets/IMAGEGEN_PROMPTS.md
新版 PNG 通过检查前继续显示当前可编辑 SVG，避免 README 出现破图。
-->

<div align="center">
  <img width="96%" alt="RoboNix 推测解码架构" src="docs/assets/speculative-decoding-architecture.svg" />
  <p><b>图 1.</b> 离线 Drafter 准备，以及包含置信度、运动学接受和目标策略回退的在线推测执行链路。</p>
</div>

推测解码的系统收益不只取决于模型前向时间，还取决于候选接受率、候选树形状、
图像预处理、仿真执行、日志和回退开销。因此正式实验必须在相同硬件、模型和随机
种子下与自回归基线进行端到端比较。

<a id="validated-release"></a>
## 🧪 已验证版本

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

<a id="quick-start"></a>
## 🚀 快速开始

仓库不包含模型、数据集或输出。建议把大文件放到独立数据盘，再通过绝对路径引用。

```bash
conda create -n robonix-spec python=3.10 -y
conda activate robonix-spec
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python -m pytest -q tests
python -m scripts.run --help
```

<a id="requirements"></a>
## ⚙️ 环境要求

| 组件 | 要求 |
| --- | --- |
| 操作系统 | 推荐 Linux，DeepSpeed 与无头 LIBERO/MuJoCo 评测依赖 Linux 环境 |
| Python | 3.10 或更高版本 |
| PyTorch | 2.2.0 |
| CUDA | 已验证环境为 CUDA 12.1，需与 PyTorch 和驱动版本匹配 |
| 仿真环境 | LIBERO 0.1.0、MuJoCo 与 EGL |
| 训练环境 | DeepSpeed 0.16.6，推荐多 GPU |

根目录 `requirements.txt` 是统一依赖入口，实际固定版本位于
`requirements/requirements-min.txt`。仓库不包含模型、数据集或运行输出。

<a id="checkpoints"></a>
## 📦 检查点从哪里来

| 资产 | 来源 |
| --- | --- |
| 目标 VLA | OpenVLA 官方仓库或 Hugging Face 上的 LIBERO 微调检查点 |
| Drafter | 使用本仓库的数据生成与 DeepSpeed 训练流程产生，或复用结构兼容的已有检查点 |
| LIBERO | 官方 LIBERO 源码、任务定义、初始状态和 MuJoCo/EGL 环境 |

目标 VLA 与 Drafter 必须在模型结构、词表、隐藏维度和动作编码上兼容。不要把任意
小模型检查点直接当作 Drafter 使用。

<a id="rollout"></a>
## 🎬 LIBERO Rollout

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

<a id="training"></a>
## 🏋️ Drafter 训练

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

## 🗂️ 目录结构

```text
.
├── modules/                  # Drafter、候选、验证、接受和策略目录
├── scripts/                  # 稳定脚本入口
├── benchmarks/libero/        # LIBERO 评测与速度测试
├── configs/                  # DeepSpeed 与模型配置
├── requirements.txt          # 统一安装入口
├── requirements/             # 依赖固定版本
├── tests/                    # 结构与独立入口测试
├── vendor/openvla/           # 权威 KERV/OpenVLA 实现
├── docs/assets/              # 架构图和 rollout 预览
└── service_bootstrap.py      # 原始代码激活与安全脚本分发
```

`vendor/openvla/` 是论文行为的权威实现，`modules/` 与 `scripts/` 提供便于服务化
和后续接入 RoboNix 的工程视图。

<a id="roadmap"></a>
## 🗺️ 路线图

- [x] 发布可独立运行的纯源码仓库。
- [x] 验证目标模型、Drafter 加载和有界视频 rollout。
- [x] 采用与 RoboNix 一致的木兰宽松许可证并补全正式引用。
- [ ] 发布带校验值和模型卡的兼容 Drafter 检查点。
- [ ] 补充自回归与推测解码的端到端基准和接受率指标。
- [ ] 接入更多 Drafter 结构与验证策略。
- [ ] 提供带版本号的 RoboNix 服务适配器。

<a id="citation"></a>
## 📝 引用

如果本工具包对你的研究有帮助，欢迎给仓库一个 Star ⭐，并引用本软件仓库：

```bibtex
@software{mao2026robonix_speculative_decoding_toolkit,
  author  = {Mao, Zhihao and He, Huiru and Zheng, Zihao},
  title   = {RoboNix Speculative Decoding Toolkit},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/lusunn111/RoboNix-Speculative-Decoding-Toolkit}
}
```

<a id="contributors"></a>
## 🤝 贡献者

感谢 [HuiruHe](https://github.com/HuiruHe) 和
[zhengzihaoPKU](https://github.com/zhengzihaoPKU) 对本工具包的贡献。贡献者记录
规则见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

<a id="license"></a>
## 📄 协议

本项目采用木兰宽松许可证第 2 版(Mulan PSL v2)，详见 [LICENSE](LICENSE)；
第三方代码继续遵循各自目录中的原始协议。
