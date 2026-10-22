# Turkish tokenizer

## SentencePiece v2: byte fallback

Eğitim kullanıcı isteğiyle durdurulmuştur; yeni final modeller üretilmemiştir.
Aşağıdaki komutlarla script daha sonra çalıştırılabilir.

`train_sentencepiece.sh`, `build_final_tokenizers.py` üzerinden dört **gerçek SentencePiece** modeli eğitir:

- `spm_unigram_32000`
- `spm_unigram_64000`
- `spm_bpe_32000`
- `spm_bpe_64000`

WordPiece ve Byte BPE eğitim dosyalarına dokunmaz. Her model `.model`, `.vocab`
ve doğrudan `tokenizers.Tokenizer.from_file(...)` ile açılan `.json` olarak çıkar.
Özel token ID'leri: `0=<pad>`, `1=<unk>`, `2=<s>`, `3=</s>`.

### Kurulum ve test

Eğitimi tek komutla başlatmak için:

```bash
bash train_sentencepiece.sh
```

Bash script `.venv` yoksa oluşturur, `requirements.txt` bağımlılıklarını kurar
ve dört modeli sırayla eğitir. Çıktı varsayılan olarak
`final_tokenizers/v2_byte_fallback_YYYYMMDD_HHMMSS` adlı yeni klasöre gider.
Komutu indirme, kurulum veya eğitim yapmadan görmek için:

```bash
bash train_sentencepiece.sh --dry-run
bash train_sentencepiece.sh --help
```

Seçenekleri değiştirme örneği:

```bash
bash train_sentencepiece.sh --threads 8 --max-sentence-length 65536 \
  --output-dir final_tokenizers/v2_byte_fallback_custom
```

Manuel kurulum ve tüm testler:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Testler küçük, geçici Unigram/BPE modelleri eğitir. Bunların geçmesi, henüz
üretilmemiş 32k/64k modellerinin eğitim veya teslim doğrulaması değildir.

### Önceki korpusu kullanarak eğitim

Önceki eğitimin hazırlanmış korpus shard'ları varsa:

```bash
.venv/bin/python build_final_tokenizers.py \
  --corpus /path/to/turkish_corpus_shards/part_*.txt \
  --validation-texts /path/to/raw_wikipedia_documents.jsonl \
  --output-dir final_tokenizers/v2_byte_fallback_2
```

`--validation-texts` en az 500 **ham doküman** içeren JSONL dosyasıdır. Her satır
bir JSON string veya `{"text": "..."}` nesnesi olabilir; dokümandaki satır sonları
JSON içinde escape edilmelidir. Korpus shard dosyaları değiştirilmez; uzun
satırlar eğitim için geçici bir kopyada parçalara bölünür.

Kaynak Hugging Face üzerindeyse önceki kaynak, config ve commit kesinleştirilmelidir:

```bash
.venv/bin/python build_final_tokenizers.py \
  --dataset ayberkmunis/wikidemo \
  --revision d98813022e1e644b7bd5f28dbcf91023004793fb \
  --text-column text \
  --output-dir final_tokenizers/v2_byte_fallback_2
```

Gerekirse `--dataset-config CONFIG` ve `--split SPLIT` eklenir. `main` gibi değişken
revision kabul edilmez. HF verisinin tüm uygun dokümanları, eski
`sentence_piece.py` içindeki aynı `clean_text` ve minimum 20 karakter filtresiyle
hazırlanır; ilk 500 ham doküman temizlenmeden doğrulamaya ayrılır. Bu örnekler
ayrı bir held-out değerlendirme seti değildir. `--validation-texts` verilirse
onun dokümanları kullanılır. Varsayılan `--input-sentence-size 0`, hazırlanmış
tüm eğitim satırlarını kullanır; 5 milyon satırlık örnekleme sınırı kaldırılmıştır.
İstenirse bu seçenekle örnekleme sayısı ayrıca sınırlandırılabilir.

### Cümle uzunluğu sınırı

Bu korpusta bir satır bir Wikipedia dokümanıdır, dilbilgisel bir cümle değildir.
Hazırlanmış 486.414 dokümanda eski `max_sentence_length=4192` bayt sınırı
51.311 dokümanı ve metin baytlarının %49,68'ini dışarıda bırakıyordu. Bu nedenle
bu sınır artık uzun metinleri atmak için kullanılmaz.

Varsayılan `--max-sentence-length auto` korpusu tarar, satır uzunluklarının
99. yüzdelik değerini 1024 baytın katına yukarı yuvarlar ve sonucu 4.192–65.536
bayt aralığında tutar. Mevcut korpusta p99=18.140 bayt olduğundan seçilen sınır
**18.432 bayttır**. En uzun doküman 1.610.652 bayttır; sınırı doğrudan bu değere
çıkarmak yerine uzun dokümanlar parçalanır.

Sınırı aşan satırlar mümkünse boşluk/sekme sınırından, çok uzun tek sözcüklerde
UTF-8 karakter sınırından bölünür. Hazırlanmış metindeki hiçbir bayt atılmaz;
parçalar eğitim dosyasında ayrı satır olur. Bu işlem eğitim bağlamını parça
sınırlarında keser, fakat uzun makalelerin tamamının atılmasını önler.
`--max-sentence-length 65536` gibi açık bir değer de kullanılabilir; uzun satırlar
bu durumda da bölünür. Ham doğrulama dokümanları bölünmez veya temizlenmez.
Seçilen sınır, dağılım ve parça sayıları `validation_report.json` içine kaydedilir.
Tam korpusla eğitim önceki filtreli çalıştırmadan daha fazla RAM ve süre ister.

Bu teslim için kullanıcı tarafından seçilen kaynak `ayberkmunis/wikidemo`,
kolon `text`, split `train`, sabit revision
`d98813022e1e644b7bd5f28dbcf91023004793fb` olarak belirlenmiştir. Repodaki eski
WordPiece/Byte BPE script'lerinin kaynak ayarları değiştirilmemiştir.

### Kayıpsız HF dönüşümü

SentencePiece ayarları `byte_fallback=True`,
`normalization_rule_name="identity"`, `remove_extra_whitespaces=False` ve
`add_dummy_prefix=False` olarak sabittir. Sözlük boyutları tam olarak istenir;
yetersiz korpusla daha küçük sözlük sessizce kabul edilmez.

HF sözlüğünde SentencePiece'in `▁` boşluk işareti gerçek ASCII boşluğa çevrilir.
Token ID'leri, Unigram skorları ve BPE birleştirme sırası korunur. HF tarafında
normalizer veya boşluk atan pre-tokenizer yoktur. Böylece baştaki/sondaki/çoklu
boşluklar korunur; metindeki gerçek `▁` karakteri de byte fallback ile ayrı
kodlanabilir. Decoder `Sequence([ByteFallback(), Fuse()])` içerir. SPM ve HF
token ID eşitliği, normal metinler ve boşluk/Unicode örnekleri üzerinde test edilir.
Metindeki gerçek `▁` için HF davranışı bilerek farklıdır: karakter aynen korunur.

HF'nin özel token kuralları geçerlidir: `<pad>`, `<unk>`, `<s>`, `</s>` kontrol
sembolleridir. Literal kontrol sembolleri ve `<0xNN>` byte token yazımları sıradan
metin gibi round-trip garantisi taşımaz. Doğrulama bu örnekleri filtrelemez veya
atlamaz; bir doküman başarısız olursa teslimi durdurur.

### Teslim ve doğrulama

Mevcut çıktı klasörünün üzerine yazılmaz. Repodaki önceden bulunan
`final_tokenizers/v2_byte_fallback/spm_unigram_32000.json` korunmuştur; eski
hatalı taslaktır ve teslim için kullanılmamalıdır. Yukarıdaki komut bu nedenle
yeni `v2_byte_fallback_2` klasörünü kullanır.

Dört JSON da diskte yeniden açılıp aşağıdakiler doğrulanmadan teslim klasörü oluşmaz:

- Tam sözlük boyutu, dört özel token ID'si ve 256 byte token'ın tamamı.
- JSON modelinde `byte_fallback: true`, decoder'da `ByteFallback`.
- Varsayılan 500 ham doküman ve ek boşluk/Unicode örneklerinde
  `decode(encode(text, add_special_tokens=False).ids) == text` ve hiç `<unk>` olmaması.

Çıktıda on iki model dosyasına ek olarak `validation_texts.jsonl` ve
`validation_report.json` bulunur. Rapor kaynak kimliğini/korpus SHA-256'sını,
paket sürümlerini, eğitim ayarlarını ve her JSON için SHA-256 ile `bad=0`
sonucunu kaydeder. GCS'e yükleme yapmaz.
