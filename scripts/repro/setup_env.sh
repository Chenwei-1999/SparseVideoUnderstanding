#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-verlrun}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_BACKENDS="${INSTALL_BACKENDS:-vllm}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.1}"
RAY_VERSION="${RAY_VERSION:-2.53.0}"
DATASETS_VERSION="${DATASETS_VERSION:-4.5.0}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy<3.0.0}"
TENSORDICT_SPEC="${TENSORDICT_SPEC:-tensordict>=0.8.0,!=0.9.0,<=0.10.0}"
DECORD_SPEC="${DECORD_SPEC:-decord}"
VLLM_VERSION="${VLLM_VERSION:-0.14.1}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.6}"

if [[ ",${INSTALL_BACKENDS}," == *,vllm,* && ",${INSTALL_BACKENDS}," == *,sglang,* ]]; then
  cat <<'EOF'
scripts/repro/setup_env.sh refuses to install vLLM and SGLang into the same environment by default.
They currently pin incompatible transitive dependencies, and a combined install leaves the environment
in a partially broken state for at least one backend.

Use separate environments instead, for example:
  ENV_NAME=verlrun-vllm INSTALL_BACKENDS=vllm bash scripts/repro/setup_env.sh
  ENV_NAME=verlrun-sglang INSTALL_BACKENDS=sglang bash scripts/repro/setup_env.sh
EOF
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"

if [[ ",${INSTALL_BACKENDS}," == *,vllm,* && ",${INSTALL_BACKENDS}," != *,sglang,* ]]; then
  if python -m pip show sglang >/dev/null 2>&1; then
    cat <<EOF
The target environment '$ENV_NAME' already contains sglang.
For reproducible vLLM runs, start from a clean env or choose a fresh name, for example:
  ENV_NAME=${ENV_NAME}-vllm INSTALL_BACKENDS=vllm bash scripts/repro/setup_env.sh
EOF
    exit 2
  fi
fi

if [[ ",${INSTALL_BACKENDS}," == *,sglang,* && ",${INSTALL_BACKENDS}," != *,vllm,* ]]; then
  if python -m pip show vllm >/dev/null 2>&1; then
    cat <<EOF
The target environment '$ENV_NAME' already contains vllm.
For reproducible SGLang runs, start from a clean env or choose a fresh name, for example:
  ENV_NAME=${ENV_NAME}-sglang INSTALL_BACKENDS=sglang bash scripts/repro/setup_env.sh
EOF
    exit 2
  fi
fi

python -m pip install -U pip setuptools wheel

if [[ ",${INSTALL_BACKENDS}," == *,vllm,* ]]; then
  python -m pip install \
    -r requirements.txt \
    "torch==${TORCH_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "ray[default]==${RAY_VERSION}" \
    "datasets==${DATASETS_VERSION}" \
    "${NUMPY_SPEC}" \
    "${TENSORDICT_SPEC}" \
    "${DECORD_SPEC}" \
    "vllm==${VLLM_VERSION}"
fi

if [[ ",${INSTALL_BACKENDS}," == *,sglang,* ]]; then
  python -m pip install \
    -r requirements_sglang.txt \
    "torch==${TORCH_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "ray[default]==${RAY_VERSION}" \
    "datasets==${DATASETS_VERSION}" \
    "${NUMPY_SPEC}" \
    "${TENSORDICT_SPEC}" \
    "${DECORD_SPEC}" \
    "sglang[all]==${SGLANG_VERSION}"
fi

python -m pip install -e . --no-deps
python -m pip install scikit-learn

python - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path


def normalize_decord_wheel_tag() -> None:
    try:
        import decord  # noqa: F401
    except Exception:
        return

    pure_py_tag = "Tag: py3-none-manylinux2010_x86_64"
    dist_info_dirs = sorted(Path(sys.prefix, "lib").glob("python*/site-packages/decord-*.dist-info"))
    for dist_info in dist_info_dirs:
        wheel_file = dist_info / "WHEEL"
        if not wheel_file.exists():
            continue
        text = wheel_file.read_text(encoding="utf-8")
        if "Tag: cp36-cp36m-manylinux2010_x86_64" not in text:
            continue
        wheel_file.write_text(
            text.replace("Tag: cp36-cp36m-manylinux2010_x86_64", pure_py_tag),
            encoding="utf-8",
        )
        print(f"Normalized decord wheel tag in {wheel_file}")


normalize_decord_wheel_tag()
PY

echo "Environment ready:"
echo "  env: $ENV_NAME"
echo "  python: $(python --version)"
echo "  repo: $(pwd)"
echo
echo "Next:"
echo "  python scripts/repro/doctor.py"
