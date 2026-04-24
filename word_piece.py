import os
import re
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, Sequence
from tokenizers.processors import TemplateProcessing


dataset_name = "yasarefe/turkish-texts-dataset"
split = "train"

output_dir = Path("turkish_wordpiece_corpus_shards")
output_dir.mkdir(exist_ok=True)

tokenizer_save_path = "turkish_wordpiece_tokenizer.json"

vocab_size = 30_000
min_frequency = 5

max_samples = None
min_text_length = 20

# Her shard yaklaşık kaç MB olsun?
max_shard_size_mb = 250
max_shard_size_bytes = max_shard_size_mb * 1024 * 1024

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def clean_text(text):
    """
    Türkçe karakterleri koruyan basit temizleme.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def get_text_from_example(example):
    """
    Dataset içindeki metin kolonunu güvenli şekilde bulur.
    """

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



print("WordPiece tokenizer başlatılıyor...")

tokenizer = Tokenizer(
    WordPiece(
        unk_token="[UNK]"
    )
)

# Türkçe için StripAccents kullanmıyoruz.
# Çünkü ç, ğ, ı, ö, ş, ü gibi karakterleri bozabilir.
tokenizer.normalizer = Sequence([
    NFD(),
    Lowercase()
])

tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=special_tokens,
    show_progress=True
)



input_files = sorted(str(p) for p in output_dir.glob("part_*.txt"))

if len(input_files) == 0:
    raise ValueError("Hiç shard dosyası bulunamadı.")

print(f"Training için toplam shard dosyası: {len(input_files)}")
print("WordPiece tokenizer eğitimi başlıyor...")

tokenizer.train(
    files=input_files,
    trainer=trainer
)

print("Training tamamlandı.")



cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_id),
        ("[SEP]", sep_id),
    ],
)



tokenizer.save(tokenizer_save_path)

print(f"Tokenizer JSON olarak kaydedildi: {tokenizer_save_path}")



text = "ODTÜ NLP'den Merhabalar. Bugün Türkçe WordPiece tokenizer eğitiyoruz."

encoding = tokenizer.encode(text)

print("\nInput text:")
print(text)

print("\nTokens:")
print(encoding.tokens)

print("\nIDs:")
print(encoding.ids)

print("\nDecoded:")
print(tokenizer.decode(encoding.ids))