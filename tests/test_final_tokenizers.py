import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from tokenizers import Tokenizer

from build_final_tokenizers import (
    EDGE_TEXTS, MODEL_SPECS, SPECIAL_TOKENS, convert_sentencepiece,
    load_validation_texts, main, parse_args, train_sentencepiece, validate_tokenizer,
    prepare_dataset, prepare_training_lines, split_utf8_line,
)


@pytest.fixture(scope="module", params=["unigram", "bpe"])
def trained(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    documents = [
        "Türkçe bir örnek metin. Ankara ve İstanbul şehirleri, Türkiye tarihi.",
        "  Başta boşluk var.  Birden çok boşluk, noktalama: 12345!  ",
        "Bilim, sanat, eğitim, kültür, tarih, coğrafya ve bilgisayar yazılımı.",
        "Kitaplarımızdan okuyucularımıza, okullarımızdan öğrencilerimize.",
        "Dünya güzel; ağaç, güneş, deniz, nehir, yağmur ve gökyüzü.",
    ]
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(documents * 100), encoding="utf-8")
    native = train_sentencepiece(
        [corpus], root / "test", request.param, 400,
        threads=1, input_sentence_size=0, hard_vocab_limit=False,
    )
    path = root / "test.json"
    convert_sentencepiece(native, path)
    return native, path, documents


def test_training_settings_and_special_ids(trained):
    native, path, _ = trained
    proto = sp_pb2.ModelProto.FromString(native.read_bytes())
    tok = Tokenizer.from_file(str(path))
    data = json.loads(path.read_text())
    assert proto.trainer_spec.byte_fallback is True
    assert proto.normalizer_spec.name == "identity"
    assert proto.normalizer_spec.remove_extra_whitespaces is False
    assert proto.normalizer_spec.add_dummy_prefix is False
    assert data["model"]["byte_fallback"] is True
    assert any(d["type"] == "ByteFallback" for d in data["decoder"]["decoders"])
    for index, token in enumerate(SPECIAL_TOKENS):
        assert proto.pieces[index].piece == token
        assert tok.token_to_id(token) == index
    assert tok.get_vocab_size() == len(proto.pieces)
    assert all(tok.token_to_id(f"<0x{i:02X}>") is not None for i in range(256))


def test_raw_documents_and_edge_cases_roundtrip(trained):
    _, path, documents = trained
    tok = Tokenizer.from_file(str(path))
    report = validate_tokenizer(path, documents, tok.get_vocab_size())
    assert report["bad"] == 0
    assert report["unk_documents"] == 0


def test_unseen_unicode_and_whitespace_roundtrip(trained):
    _, path, _ = trained
    tok = Tokenizer.from_file(str(path))
    rng = random.Random(42)
    for _ in range(300):
        codepoints = [rng.randrange(0x110000) for _ in range(15)]
        text = " \t" + "".join(chr(c) for c in codepoints if not 0xD800 <= c <= 0xDFFF) + "\n  "
        ids = tok.encode(text, add_special_tokens=False).ids
        assert 1 not in ids
        assert tok.decode(ids) == text


def test_export_preserves_native_ids_scores_and_segmentation(trained):
    native, path, documents = trained
    source = spm.SentencePieceProcessor(model_file=str(native))
    tok = Tokenizer.from_file(str(path))
    for text in documents + [text for text in EDGE_TEXTS if "▁" not in text]:
        assert tok.encode(text, add_special_tokens=False).ids == source.encode(text)
    proto = sp_pb2.ModelProto.FromString(native.read_bytes())
    data = json.loads(path.read_text())
    for index, piece in enumerate(proto.pieces):
        assert tok.id_to_token(index) == piece.piece.replace("▁", " ")
        if data["model"]["type"] == "Unigram":
            assert data["model"]["vocab"][index][1] == pytest.approx(piece.score)


def test_validation_rejects_wrong_size(trained):
    _, path, documents = trained
    with pytest.raises(ValueError, match="exactly 32000"):
        validate_tokenizer(path, documents, 32_000)


def test_validation_rejects_missing_byte_fallback(trained, tmp_path):
    _, path, documents = trained
    data = json.loads(path.read_text())
    data["model"]["byte_fallback"] = False
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="model.byte_fallback"):
        validate_tokenizer(broken, documents, Tokenizer.from_file(str(path)).get_vocab_size())


def test_validation_rejects_lossy_decoder(trained, tmp_path):
    _, path, documents = trained
    data = json.loads(path.read_text())
    data["decoder"]["decoders"].append({"type": "Replace", "pattern": {"String": " "}, "content": ""})
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="validation failed"):
        validate_tokenizer(broken, documents, Tokenizer.from_file(str(path)).get_vocab_size())


def test_load_validation_documents_without_cleaning(tmp_path):
    path = tmp_path / "texts.jsonl"
    texts = ["  iki  boşluk\n\t", "\r\nüçüncü satır  "]
    path.write_text(json.dumps(texts[0]) + "\n" + json.dumps({"text": texts[1]}) + "\n")
    assert load_validation_texts(path, 2) == texts
    with pytest.raises(ValueError, match="Need 3"):
        load_validation_texts(path, 3)


def test_cli_refuses_overwrite_and_requires_source_identity(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "example/wiki"])
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "example/wiki", "--revision", "main"])
    with pytest.raises(SystemExit):
        parse_args(["--corpus", "old.txt"])
    with pytest.raises(SystemExit):
        parse_args(["--corpus", "old.txt", "--validation-texts", "raw.jsonl", "--output-dir", str(tmp_path)])


def test_only_requested_models_are_trained():
    assert MODEL_SPECS == [("unigram", 32_000), ("unigram", 64_000), ("bpe", 32_000), ("bpe", 64_000)]


@pytest.mark.parametrize("limit", [4, 17, 4192])
def test_chunking_preserves_utf8_and_whitespace(limit):
    for text in ["", "  çok  boşluk \t", "🧿ğ" * 2000, " " + "x" * 9000,
                 "Türkçe bir metin. " * 1000]:
        raw = text.encode("utf-8")
        chunks = list(split_utf8_line(raw, limit))
        assert b"".join(chunks) == raw
        assert all(0 < len(chunk) <= limit for chunk in chunks)
        for chunk in chunks:
            chunk.decode("utf-8")


def test_auto_length_increases_and_chunks_outliers(tmp_path):
    corpus = tmp_path / "source.txt"
    rows = [b"a" * 18_140] * 99 + [b"b" * 100_000]
    corpus.write_bytes(b"\n".join(rows) + b"\n")
    paths, policy = prepare_training_lines([corpus], tmp_path, "auto")
    assert policy["max_sentence_length"] == 18_432
    assert policy["p99_line_bytes"] == 18_140
    assert policy["split_source_lines"] == 1
    assert policy["dropped_for_length"] == 0
    chunks = paths[0].read_bytes().splitlines()
    assert max(map(len, chunks)) <= 18_432
    assert b"".join(chunks) == b"".join(rows)


def test_auto_length_cap_and_explicit_limit(tmp_path):
    corpus = tmp_path / "source.txt"
    corpus.write_bytes(b"x" * 100_000 + b"\n")
    _, policy = prepare_training_lines([corpus], tmp_path, "auto")
    assert policy["max_sentence_length"] == 65_536
    _, policy = prepare_training_lines([corpus], tmp_path, "8192")
    assert policy["max_sentence_length"] == 8192
    assert policy["training_lines"] == 13


def test_trainer_receives_length_and_sampling_limits(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(spm.SentencePieceTrainer, "train", lambda **kwargs: captured.update(kwargs))
    train_sentencepiece([tmp_path / "source.txt"], tmp_path / "model", "bpe", 32_000,
                        threads=2, max_sentence_length=18_432)
    assert captured["max_sentence_length"] == 18_432
    assert captured["input_sentence_size"] == 0
    assert captured["vocab_size"] == 32_000


@pytest.mark.parametrize("limit", ["0", "3", "-1", "invalid", "2147483648"])
def test_cli_rejects_invalid_length(limit, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--corpus", "old.txt", "--validation-texts", "raw.jsonl",
                    "--output-dir", str(tmp_path / "new"), "--max-sentence-length", limit])


def test_dataset_uses_only_requested_text_column(monkeypatch, tmp_path):
    import datasets

    raw = "  Bu dokümanın metni\n\tboşluklarla korunmalı.  "
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: iter([
        {"title": "Başlık eğitime girmemeli", "text": raw, "content": "Yanlış alan"},
    ]))
    args = SimpleNamespace(dataset="ayberkmunis/wikidemo", dataset_config=None,
                           revision="a" * 40, split="train", cache_dir=tmp_path,
                           text_column="text", validation_count=500)
    corpus, samples, provenance = prepare_dataset(args, tmp_path)
    assert corpus[0].read_text() == "Bu dokümanın metni boşluklarla korunmalı.\n"
    assert samples == [raw]
    assert provenance["text_column"] == "text"


def test_failed_validation_never_publishes(monkeypatch, tmp_path):
    import build_final_tokenizers as build

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Türkçe test metni\n")
    samples = tmp_path / "raw.jsonl"
    samples.write_text((json.dumps("ham metin") + "\n") * 200)
    output = tmp_path / "delivery"
    monkeypatch.setattr(build, "train_sentencepiece", lambda *a, **kw: tmp_path / "unused.model")
    monkeypatch.setattr(build, "convert_sentencepiece", lambda *a: None)
    def fail_validation(*args):
        raise ValueError("validation failed")
    monkeypatch.setattr(build, "validate_tokenizer", fail_validation)
    with pytest.raises(ValueError, match="validation failed"):
        main(["--corpus", str(corpus), "--validation-texts", str(samples),
              "--validation-count", "200", "--output-dir", str(output)])
    assert not output.exists()
    assert not list(tmp_path.glob(".spm-build-*"))
