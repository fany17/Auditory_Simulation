#!/usr/bin/env bash
set -u

DEST="/home/fanyu/auditory_simulation_m6a/m6a_public_002/cache/parakeet_tdt_0.6b_v3"
BASE="https://hf-mirror.com/nvidia/parakeet-tdt-0.6b-v3/resolve/main"
LOG="/home/fanyu/auditory_simulation_m6a/m6a_public_002/logs/parakeet_snapshot_download.log"
mkdir -p "$DEST" "$(dirname "$LOG")"

files=(
  ".gitattributes"
  "README.md"
  "config.json"
  "generation_config.json"
  "model.safetensors"
  "parakeet-tdt-0.6b-v3.nemo"
  "parakeet-tdt-0.6b-v3.q8_0.gguf"
  "processor_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  ".eval_results/open_asr_leaderboard.yaml"
  "plots/asr.png"
)

printf 'source=%s\n' "$BASE" | tee -a "$LOG"
failed=0
for rel in "${files[@]}"; do
  target="$DEST/$rel"
  partial="$target.partial-download"
  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" && -s "$target" ]]; then
    bytes=$(stat -c '%s' "$target")
    printf 'existing|%s|bytes=%s|time=%s\n' "$rel" "$bytes" "$(date --iso-8601=seconds)" | tee -a "$LOG"
    continue
  fi
  printf 'start|%s|time=%s\n' "$rel" "$(date --iso-8601=seconds)" | tee -a "$LOG"
  if curl --fail --location --retry 4 --retry-delay 5 --continue-at - --output "$partial" "$BASE/$rel"; then
    bytes=$(stat -c '%s' "$partial")
    if [[ "$bytes" -le 0 ]]; then
      printf 'failed|%s|reason=empty|time=%s\n' "$rel" "$(date --iso-8601=seconds)" | tee -a "$LOG"
      failed=1
      continue
    fi
    mv -- "$partial" "$target"
    printf 'complete|%s|bytes=%s|time=%s\n' "$rel" "$bytes" "$(date --iso-8601=seconds)" | tee -a "$LOG"
  else
    printf 'failed|%s|reason=http_or_transfer|time=%s\n' "$rel" "$(date --iso-8601=seconds)" | tee -a "$LOG"
    failed=1
  fi
done
exit "$failed"
