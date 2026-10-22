#!/usr/bin/env bash
# Unigram 32k/64k + SentencePiece BPE 32k/64k from wikidemo/text.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
BASE_PYTHON="${PYTHON_BIN:-python3}"
python_mode=auto
DATASET="${DATASET:-ayberkmunis/wikidemo}"
REVISION="${REVISION:-d98813022e1e644b7bd5f28dbcf91023004793fb}"
DRIVE_ROOT="/content/drive/MyDrive"
OUTPUT_DIR="${OUTPUT_DIR:-$DRIVE_ROOT/tokenizers/final_tokenizers/v2_byte_fallback_$(date +%Y%m%d_%H%M%S)}"

dry_run=false
extra_args=()
while (($#)); do
    case "$1" in
        --help|-h)
            cat <<'EOF'
Kullanım: bash train_sentencepiece.sh [--dry-run] [Python script seçenekleri]

Unigram 32k/64k ve SentencePiece BPE 32k/64k modellerini sırayla eğitir.
Kaynak: ayberkmunis/wikidemo, text sütunu; revision sabitlenmiştir.
Çıktı: /content/drive/MyDrive/tokenizers/final_tokenizers/v2_byte_fallback_TARIH_SAAT
Önce Colab hücresinde drive.mount('/content/drive') ile Drive'ı bağla.
Colab'da hazır Python ortamını, diğer ortamlarda .venv kullanır.
requirements.txt bağımlılıklarını seçilen Python ortamına kurar.

--dry-run                     Kurulum/indirme/eğitim yapmadan komutu gösterir.
--system-python               .venv/ensurepip kullanmadan hazır Python ile çalışır.
--venv                        Sanal ortam kullanımını zorlar.
--max-sentence-length auto     Korpus p99 uzunluğu (4192..65536 bayt); varsayılan.
--max-sentence-length 65536    Sabit UTF-8 bayt sınırı. Uzun metinler parçalanır.
--input-sentence-size 0        Tüm hazırlanmış satırlar; varsayılan.
--threads 8                   Eğitim thread sayısı (varsayılan en fazla 8).
--output-dir PATH             Yeni teslim klasörü; mevcut klasöre yazmaz.

Ortam değişkenleri: PYTHON_BIN, VENV_DIR, DATASET, REVISION, OUTPUT_DIR.
Yeni bir DATASET seçersen ona ait sabit REVISION değerini de belirt.
EOF
            exit 0
            ;;
        --dry-run) dry_run=true ;;
        --system-python) python_mode=system ;;
        --venv) python_mode=venv ;;
        --output-dir)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf '%s\n' '--output-dir için klasör yolu gerekli.' >&2
                exit 2
            fi
            OUTPUT_DIR="$2"
            shift
            ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}" ;;
        *) extra_args+=("$1") ;;
    esac
    shift
done

if [[ "$python_mode" == auto ]]; then
    if [[ -n "${COLAB_RELEASE_TAG:-}" || -n "${COLAB_BACKEND_VERSION:-}" ]] || \
        "$BASE_PYTHON" -c 'import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec("google") and importlib.util.find_spec("google.colab") else 1)' 2>/dev/null; then
        python_mode=system
    else
        python_mode=venv
    fi
fi
if [[ "$python_mode" == system ]]; then
    TRAINING_PYTHON="$BASE_PYTHON"
else
    TRAINING_PYTHON="$VENV_DIR/bin/python"
fi

command_args=(
    "$TRAINING_PYTHON" -u "$SCRIPT_DIR/build_final_tokenizers.py"
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

if [[ "$OUTPUT_DIR" == "$DRIVE_ROOT/"* && ! -d "$DRIVE_ROOT" ]]; then
    printf '%s\n' "Google Drive bağlı değil. Önce Colab'da çalıştır:" \
        'from google.colab import drive' "drive.mount('/content/drive')" >&2
    exit 1
fi

if [[ "$python_mode" == venv ]]; then
    if [[ ! -x "$TRAINING_PYTHON" ]] || ! "$TRAINING_PYTHON" -m pip --version >/dev/null 2>&1; then
        if ! "$BASE_PYTHON" -m venv "$VENV_DIR"; then
            printf '%s\n' 'Sanal ortam kurulamadı. Hazır Python ortamını kullanmak için --system-python ile tekrar çalıştır.' >&2
            exit 1
        fi
    fi
fi
"$TRAINING_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

# exec preserves the training exit status and lets Ctrl+C reach Python directly.
exec "${command_args[@]}"
