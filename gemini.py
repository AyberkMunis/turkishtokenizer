from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


import google.generativeai as genai
import os


os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("models/gemini-2.5-flash")

prompt_template = """
Sen bir Türk dili, dilbilim ve morfoloji uzmanısın. Temel görevin, sana verilen Türkçe kelimeleri kök ve eklerine ayırmaktır. Sana virgülle ayrılmış veya cümle halinde birden fazla kelime verilebilir.

Kurallar:
1. Kökleri bulurken ses olaylarını (ünsüz yumuşaması, ünlü düşmesi vb.) tersine çevirerek kökün orijinal (yalın) halini yaz. (Örn: "gidip" kelimesinin kökü "git", "ağaca" kelimesinin kökü "ağaç" olarak yazılmalıdır).
2. DİKKAT: Ekleri ise kelimenin içinde nasıl geçiyorsa, yani ses olayına (sertleşme, daralma vb.) uğramış haliyle yaz. Orijinal ek formunu KULLANMA. (Örn: "sokakta" kelimesinin ekini "da" değil, "ta" olarak; "kitapçı" kelimesinin ekini "cı" değil, "çı" olarak yaz).
3. Çıktıyı SADECE geçerli bir JSON dizisi (array) formatında ver. Listedeki her bir obje şu anahtarları içermelidir: "kelime", "kok", "ekler" (liste formatında) ve "ayrilmis_hali" (kök ve eklerin tire ile ayrılmış string hali).
4. JSON dışında hiçbir ek açıklama, kod bloğu işareti veya sohbet metni yazma. Doğrudan köşeli parantez ([]) ile başlayan JSON dizisini döndür.

Örnekler:

Kullanıcı: arabalarımızdan, gidip, ağaca
Asistan: [
  {
    "kelime": "arabalarımızdan",
    "kok": "araba",
    "ekler": ["lar", "ımız", "dan"],
    "ayrilmis_hali": "araba-lar-ımız-dan"
  },
  {
    "kelime": "gidip",
    "kok": "git",
    "ekler": ["ip"],
    "ayrilmis_hali": "git-ip"
  },
  {
    "kelime": "ağaca",
    "kok": "ağaç",
    "ekler": ["a"],
    "ayrilmis_hali": "ağaç-a"
  }
]

Kullanıcı: sokakta başlıyor fikrim
Asistan: [
  {
    "kelime": "sokakta",
    "kok": "sokak",
    "ekler": ["ta"],
    "ayrilmis_hali": "sokak-ta"
  },
  {
    "kelime": "başlıyor",
    "kok": "baş",
    "ekler": ["lı", "yor"],
    "ayrilmis_hali": "baş-lı-yor"
  },
  {
    "kelime": "fikrim",
    "kok": "fikir",
    "ekler": ["im"],
    "ayrilmis_hali": "fikir-im"
  }
]

Kullanıcı: kitapçıya hissetti
Asistan: [
  {
    "kelime": "kitapçıya",
    "kok": "kitap",
    "ekler": ["çı", "ya"],
    "ayrilmis_hali": "kitap-çı-ya"
  },
  {
    "kelime": "hissetti",
    "kok": "his",
    "ekler": ["et", "ti"],
    "ayrilmis_hali": "his-et-ti"
  }
]

Kullanıcı: [gözlükçüden,sessizliğe,aydınlıktı,kalbimiz,yaptıracakmış,okuyucularımız,getirttik,büyüdükçe,burnunu,bekletiliyor]
Asistan:
"""

response = model.generate_content(prompt_template)

data = json.loads(response.text)

morph_tokens=[]
for i in range(len(data)):
    data[i]["ekler"].append(data[i]["kok"])
    for y in range(len(data[i]["ekler"])):
        morph_tokens.append(data[i]["ekler"][y])

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

vocab = {}

for token in special_tokens:
    vocab[token] = len(vocab)

for token in morph_tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

tokenizer = Tokenizer(
    WordLevel(
        vocab=vocab,
        unk_token="[UNK]"
    )
)

tokenizer.pre_tokenizer = Whitespace()

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", vocab["[CLS]"]),
        ("[SEP]", vocab["[SEP]"]),
    ],
)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

text = "göz lük çü den"

encoded = fast_tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=12,
    return_tensors="pt"
)

print(fast_tokenizer.tokenize(text))
print(encoded["input_ids"])
print(encoded["attention_mask"])

fast_tokenizer.save_pretrained("custom_morph_tokenizer")