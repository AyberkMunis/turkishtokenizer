#!/usr/bin/env bash
# Unigram 32k/64k + SentencePiece BPE 32k/64k from wikidemo/text.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
DATASET="${DATASET:-ayberkmunis/wikidemo}"
REVISION="${REVISION:-d98813022e1e644b7bd5f28dbcf91023004793fb}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/final_tokenizers/v2_byte_fallback_$(date +%Y%m%d_%H%M%S)}"

dry_run=false
extra_args=()
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            cat <<'EOF'
Kullanım: bash train_sentencepiece.sh [--dry-run] [Python script seçenekleri]

Unigram 32k/64k ve SentencePiece BPE 32k/64k modellerini sırayla eğitir.
Kaynak: ayberkmunis/wikidemo, text sütunu; revision sabitlenmiştir.
Eksikse .venv oluşturur ve requirements.txt bağımlılıklarını kurar.

--dry-run                     Kurulum/indirme/eğitim yapmadan komutu gösterir.
--max-sentence-length auto     Korpus p99 uzunluğu (4192..65536 bayt); varsayılan.
--max-sentence-length 65536    Sabit UTF-8 bayt sınırı. Uzun metinler parçalanır.
--input-sentence-size 0        Tüm hazırlanmış satırlar; varsayılan.
--threads 8                   Eğitim thread sayısı (varsayılan en fazla 8).
--output-dir PATH             Yeni teslim klasörü; mevcut klasöre yazmaz.

Ortam değişkenleri: VENV_DIR, DATASET, REVISION, OUTPUT_DIR.
Yeni bir DATASET seçersen ona ait sabit REVISION değerini de belirt.
EOF
            exit 0
            ;;
        --dry-run) dry_run=true ;;
        *) extra_args+=("$arg") ;;
    esac
done

command_args=(
    "$VENV_DIR/bin/python" -u "$SCRIPT_DIR/build_final_tokenizers.py"
    --dataset "$DATASET"
    --revision "$REVISION"
    --text-column text
    --input-sentence-size 0
    --max-sentence-length auto
    --output-dir "$OUTPUT_DIR"
)
if ((${#extra_args[@]})); then
    command_args+=("${extra_args[@]}")
fi

if "$dry_run"; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

# exec preserves the training exit status and lets Ctrl+C reach Python directly.
exec "${command_args[@]}"
