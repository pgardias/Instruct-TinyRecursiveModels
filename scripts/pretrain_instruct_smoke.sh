#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOKENIZER_PATH="${LLAMA_TOKENIZER:-${REPO_ROOT}/tokenizers/llama-32k/tokenizer.model}"
DATA_DIR="${SMOKE_DATA_DIR:-${REPO_ROOT}/data/slimorca_full}"

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer model not found at ${TOKENIZER_PATH}."
  echo "Point LLAMA_TOKENIZER to a valid SentencePiece .model file or place one at tokenizers/llama-32k/tokenizer.model."
  exit 1
fi

export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

python "${REPO_ROOT}/pretrain_instruct.py" \
  dataset.tokenizer_path="${TOKENIZER_PATH}" \
  dataset.data_dir="${DATA_DIR}" \
  dataset.subset_size="${SMOKE_SUBSET_SIZE:-64}" \
  dataset.seq_len="${SMOKE_SEQ_LEN:-512}" \
  dataset.test_ratio="${SMOKE_TEST_RATIO:-0.05}" \
  global_batch_size="${SMOKE_GLOBAL_BATCH:-32}" \
  epochs="${SMOKE_EPOCHS:-1}" \
  "$@"
