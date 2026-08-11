#!/usr/bin/env bash

set -uo pipefail

task_root=/home/fanyu/auditory_simulation_tb001_demo001_20260806
audit_root=/home/fanyu/.codex-run-audit/tb001-demo001-20260806
spec_path="$task_root/docs/STANDALONE_EXECUTION_SPEC.md"
attempt_tag=${1:-attempt2}
events_path="$audit_root/codex_exec_${attempt_tag}.jsonl"
stderr_path="$audit_root/codex_exec_${attempt_tag}.stderr"
meta_path="$audit_root/run_meta_${attempt_tag}.txt"

mkdir -p "$task_root/.cache/huggingface" "$audit_root"
chmod 700 "$audit_root"

if [[ -f /home/fanyu/.codex/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /home/fanyu/.codex/.env
  set +a
fi

export PATH="/usr/bin:/bin:/home/fanyu/.local/bin:/home/fanyu/.conda/envs/auditory_demos/bin:$PATH"
export HF_HOME="$task_root/.cache/huggingface"

start_utc=$(date -u +%FT%TZ)
codex_version=$(codex --version 2>&1)

{
  sed -n '1,$p' "$spec_path"
  printf '\n\n%s\n' '本轮是初步部署，只执行第18章定义的S0-S2覆盖范围。优先复用当前PATH中的auditory_demos环境；不要修改base环境。严格停止在INITIAL_DEPLOYMENT报告，不要执行S3-S6。'
} | timeout 2700 codex exec \
  --json \
  --skip-git-repo-check \
  -s workspace-write \
  -c shell_environment_policy.inherit=all \
  -C "$task_root" \
  - \
  >"$events_path" \
  2>"$stderr_path"
run_exit=$?

end_utc=$(date -u +%FT%TZ)
events_sha=$(sha256sum "$events_path" | awk '{print $1}')
printf 'START_UTC=%s\nEND_UTC=%s\nEXIT_CODE=%s\nCODEX_VERSION=%s\nEVENTS_SHA256=%s\n' \
  "$start_utc" "$end_utc" "$run_exit" "$codex_version" "$events_sha" \
  >"$meta_path"

exit "$run_exit"
