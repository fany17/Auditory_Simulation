#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${M6A_REMOTE_ROOT:-/home/fanyu/auditory_simulation_m6a}"
ENV_NAME="auditory_m6a_public_001"
ENV_FILE="${SOURCE_ROOT}/environment/m6a_public_2203.yml"

mkdir -p \
  "${PROJECT_ROOT}/cache/huggingface" \
  "${PROJECT_ROOT}/data/ds004703/v1.1.0" \
  "${PROJECT_ROOT}/logs" \
  "${PROJECT_ROOT}/metadata" \
  "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/outputs"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Environment already exists: ${ENV_NAME}"
else
  conda env create --file "${ENV_FILE}"
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python "${SOURCE_ROOT}/scripts/remote_selfcheck.py" \
  --config "${SOURCE_ROOT}/configs/m6a_public_001.json" \
  --output "${PROJECT_ROOT}/metadata/remote_selfcheck.json"

conda run -n "${ENV_NAME}" python -m pip list --format=freeze \
  > "${PROJECT_ROOT}/metadata/environment_packages.txt"

find "${PROJECT_ROOT}/metadata" -maxdepth 1 -type f \
  -printf '%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
