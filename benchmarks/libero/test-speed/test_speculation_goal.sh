SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="${KERV_SERVICE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python "${SERVICE_ROOT}/vendor/openvla/experiments/robot/libero/run_libero_goal_Spec.py" \
    --model_family openvla \
    --pretrained_checkpoint "${KERV_PRETRAINED_CHECKPOINT:-/path/to/openvla-7b-finetuned-libero-goal}" \
    --task_suite_name libero_goal \
    --center_crop True
