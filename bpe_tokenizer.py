from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing


dataset_hf_path="yasarefe/turkish-texts-dataset"
save_path = "turkish_byte_bpe_tokenizer.json"
vocab_size = 32_000
min_frequency = 3
print("Veri seti yükleniyor...")

dataset = load_dataset(
    dataset_hf_path,
    split="train",
    streaming=True
)

print("Veri seti başarıyla yüklendi.")


print("Tokenizer başlatılıyor...")

tokenizer = Tokenizer(
    models.BPE(unk_token="<unk>")
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<mask>"
    ]
)

def batch_iterator(dataset_stream, batch_size=2000, max_samples=1_000_000):
    """
    Dataset streaming olarak okunur.
    Tokenizer eğitimi için batch batch text listeleri üretir.
    """

    batch = []
    count = 0

    for example in dataset_stream:
        if count >= max_samples:
            break

        # Dataset kolon adını güvenli kontrol et
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        elif "sentence" in example:
            text = example["sentence"]
        else:
            # İlk örnekte kolonları görmen için hata verelim
            raise KeyError(
                f"Text kolonu bulunamadı. Mevcut kolonlar: {list(example.keys())}"
            )

        # Çok kısa veya boş textleri atla
        if len(text) < 10:
            continue

        batch.append(text)
        count += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

print("Tokenizer eğitimi başlıyor...")

tokenizer.train_from_iterator(
    batch_iterator(dataset),
    trainer=trainer,
    length=1_000_000
)

print("Tokenizer eğitimi tamamlandı.")


bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")

tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <eos> $B:1 <eos>:1",
    special_tokens=[
        ("<bos>", bos_id),
        ("<eos>", eos_id),
    ],
)



tokenizer.save(save_path)

print("İşlem tamam! Tokenizer başarıyla kaydedildi.")