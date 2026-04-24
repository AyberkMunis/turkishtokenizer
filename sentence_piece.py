import os
import re
from pathlib import Path
from datasets import load_dataset
import sentencepiece as spm

dataset_name = "yasarefe/turkish-texts-dataset"
split = "train"

output_dir = Path("turkish_corpus_shards")
output_dir.mkdir(exist_ok=True)

model_prefix = "spm_tr"

vocab_size = 32_000
model_type = "unigram"
character_coverage = 1.0

# Büyük veri için önemli ayarlar
max_samples = None
min_text_length = 20

# Her shard yaklaşık kaç MB olsun?
max_shard_size_mb = 250
max_shard_size_bytes = max_shard_size_mb * 1024 * 1024

# SentencePiece eğitim sırasında kaç cümle örneklesin?
input_sentence_size = 5_000_000

special_tokens = {
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "pad_piece": "[PAD]",
    "unk_piece": "[UNK]",
    "bos_piece": "[CLS]",
    "eos_piece": "[SEP]",
    "user_defined_symbols": "[MASK]",
}


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def get_text_from_example(example):
    possible_columns = ["text", "content", "sentence", "document", "body"]

    for col in possible_columns:
        if col in example:
            return example[col]

    raise KeyError(
        f"Text kolonu bulunamadı. Mevcut kolonlar: {list(example.keys())}"
    )


def write_corpus_shards():
    print("Dataset yükleniyor...")

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True
    )

    print("Shard corpus dosyaları oluşturuluyor...")

    shard_id = 0
    current_size = 0
    count = 0

    shard_path = output_dir / f"part_{shard_id:05d}.txt"
    f = open(shard_path, "w", encoding="utf-8")

    try:
        for example in dataset:
            if max_samples is not None and count >= max_samples:
                break

            text = get_text_from_example(example)
            text = clean_text(text)

            if len(text) < min_text_length:
                continue

            line = text + "\n"
            encoded_line = line.encode("utf-8")

            if current_size + len(encoded_line) > max_shard_size_bytes:
                f.close()

                print(
                    f"Shard tamamlandı: {shard_path} "
                    f"({current_size / 1024 / 1024:.2f} MB)"
                )

                shard_id += 1
                shard_path = output_dir / f"part_{shard_id:05d}.txt"
                f = open(shard_path, "w", encoding="utf-8")
                current_size = 0

            f.write(line)
            current_size += len(encoded_line)
            count += 1

            if count % 100_000 == 0:
                print(f"{count} örnek yazıldı...")

    finally:
        f.close()

    print("Shard corpus oluşturma tamamlandı.")
    print(f"Toplam örnek sayısı: {count}")
    print(f"Toplam shard sayısı: {shard_id + 1}")


write_corpus_shards()


input_files = sorted(str(p) for p in output_dir.glob("part_*.txt"))

if len(input_files) == 0:
    raise ValueError("Hiç shard dosyası bulunamadı.")

print(f"SentencePiece için toplam input dosyası: {len(input_files)}")

# SentencePiece input parametresi virgülle ayrılmış dosya listesi alabilir.
input_arg = ",".join(input_files)


print("SentencePiece tokenizer eğitimi başlıyor...")

spm.SentencePieceTrainer.Train(
    input=input_arg,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage,

    pad_id=special_tokens["pad_id"],
    unk_id=special_tokens["unk_id"],
    bos_id=special_tokens["bos_id"],
    eos_id=special_tokens["eos_id"],

    pad_piece=special_tokens["pad_piece"],
    unk_piece=special_tokens["unk_piece"],
    bos_piece=special_tokens["bos_piece"],
    eos_piece=special_tokens["eos_piece"],

    user_defined_symbols=special_tokens["user_defined_symbols"],

    input_sentence_size=input_sentence_size,
    shuffle_input_sentence=True,

    train_extremely_large_corpus=True,
    num_threads=os.cpu_count()
)

print("Training tamamlandı.")
print("Model:", f"{model_prefix}.model")
print("Vocab:", f"{model_prefix}.vocab")
tokenizer_deberta = DebertaV2Tokenizer(
    vocab_file  = "/content/spm_tr.model",
    max_len = 512,
)
tokenizer_deberta.save_pretrained(model_prefix)
