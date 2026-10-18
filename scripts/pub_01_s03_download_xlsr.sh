#!/usr/bin/env bash
set -euo pipefail
BASE=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/continuation01
REV=46f221381d200f7bef268309b3f02023ccf11fcc
DEST="$BASE/models/xlsr_dutch_$REV"
mkdir -p "$DEST"
# Identity and Apache-2.0 checked against official HuggingFace API/model card.
# Mirror transport only; revision stays fixed. No dataset download.
for filename in config.json preprocessor_config.json README.md pytorch_model.bin; do
  if [[ -e "$DEST/$filename" ]]; then
    printf 'Existing immutable file: %s\n' "$filename"
    continue
  fi
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 7200 --continue-at - \
    "https://hf-mirror.com/jonatasgrosman/wav2vec2-large-xlsr-53-dutch/resolve/$REV/$filename" \
    --output "$DEST/$filename.part"
  mv -- "$DEST/$filename.part" "$DEST/$filename"
done
printf 'Pinned XLSR download complete: %s\n' "$DEST"
