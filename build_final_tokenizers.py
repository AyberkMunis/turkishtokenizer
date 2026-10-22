"""Train four SentencePiece models and validate their standalone HF exports."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from tokenizers import AddedToken, Tokenizer, decoders, models

ROOT = Path(__file__).resolve().parent
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
MODEL_SPECS = [("unigram", 32_000), ("unigram", 64_000), ("bpe", 32_000), ("bpe", 64_000)]
EDGE_TEXTS = [
    "", " ", "   ", "  başta  ortada   sonda  ",
    "Birinci satır\n\nİkinci\r\n\tsekme\tson\n",
    "İı Iİ Şş Ğğ Çç Öö Üü — € 12345!",
    "é e\u0301 İ I\u0307 ﬁ Ａ", "🧿 🫠 𐍈 中文 العربية",
    "\u00a0ince\u2009boşluk\u202f\u200b", "a\x00b", "▁ gerçek alt çizgi",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    """Match the original sentence_piece.py training-corpus preparation only."""
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def extract_text(example: dict) -> str:
    for key in ("text", "content", "sentence", "document", "body"):
        if key in example and isinstance(example[key], str):
            return example[key]
    raise ValueError(f"No string text column in {list(example)}")


def prepare_dataset(args, work: Path) -> tuple[list[Path], list[str], dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset, name=args.dataset_config, revision=args.revision,
        split=args.split, streaming=True, cache_dir=str(args.cache_dir),
    )
    corpus = work / "training.txt"
    samples = []
    count = 0
    with corpus.open("w", encoding="utf-8", newline="\n") as stream:
        for example in dataset:
            raw = example[args.text_column]
            if not isinstance(raw, str):
                raise ValueError(f"Expected a string in column {args.text_column!r}")
            cleaned = clean_text(raw)
            if len(cleaned) < 20:
                continue
            # Keep original documents for validation: never validate cleaned text.
            if len(samples) < args.validation_count:
                samples.append(raw)
            stream.write(cleaned + "\n")
            count += 1
            if count % 100_000 == 0:
                print(f"Prepared {count:,} documents", flush=True)
    return [corpus], samples, {
        "dataset": args.dataset, "revision": args.revision,
        "config": args.dataset_config, "split": args.split,
        "text_column": args.text_column,
        "documents": count, "preparation": "sentence_piece.py clean_text; min length 20",
        "sha256": sha256(corpus),
    }


def load_validation_texts(path: Path, count: int) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            texts.append(value if isinstance(value, str) else extract_text(value))
            if len(texts) == count:
                break
    if len(texts) < count:
        raise ValueError(f"Need {count} raw documents; found {len(texts)} in {path}")
    return texts


def split_utf8_line(raw: bytes, limit: int):
    """Keep every byte, prefer whitespace boundaries, never split a UTF-8 codepoint."""
    start = 0
    while len(raw) - start > limit:
        end = start + limit
        while raw[end] & 0xC0 == 0x80:
            end -= 1
        # Include the delimiter in the previous chunk to preserve the text.
        boundary = max(raw.rfind(b" ", start, end), raw.rfind(b"\t", start, end))
        if boundary >= start:
            end = boundary + 1
        yield raw[start:end]
        start = end
    if start < len(raw):
        yield raw[start:]


def prepare_training_lines(corpus: list[Path], work: Path, requested_limit: str):
    """Choose a length from the corpus, then chunk outliers instead of dropping them."""
    histogram = Counter()
    for path in corpus:
        with path.open("rb") as stream:
            for line in stream:
                histogram[len(line.rstrip(b"\r\n"))] += 1
    count = sum(histogram.values())
    total_bytes = sum(length * frequency for length, frequency in histogram.items())
    if not count or not total_bytes:
        raise ValueError("Training corpus is empty")
    seen = 0
    for length, frequency in sorted(histogram.items()):
        seen += frequency
        if seen * 100 >= count * 99:
            p99 = length
            break
    # Cover 99% of documents directly, bounded to avoid pathological giant lines.
    limit = min(65_536, max(4_192, ((p99 + 1023) // 1024) * 1024)) if requested_limit == "auto" else int(requested_limit)
    if limit < 4:
        raise ValueError("max_sentence_length must be at least 4 UTF-8 bytes")
    output = work / "training_chunked.txt"
    training_lines = 0
    with output.open("wb") as target:
        for path in corpus:
            with path.open("rb") as stream:
                for line in stream:
                    for chunk in split_utf8_line(line.rstrip(b"\r\n"), limit):
                        target.write(chunk + b"\n")
                        training_lines += 1
    statistics = {
        "requested_max_sentence_length": requested_limit,
        "max_sentence_length": limit, "source_lines": count,
        "source_text_bytes": total_bytes, "p99_line_bytes": p99,
        "max_source_line_bytes": max(histogram),
        "lines_over_old_4192_limit": sum(n for length, n in histogram.items() if length > 4192),
        "bytes_over_old_4192_limit": sum(length * n for length, n in histogram.items() if length > 4192),
        "split_source_lines": sum(n for length, n in histogram.items() if length > limit),
        "training_lines": training_lines, "dropped_for_length": 0,
        "chunking": "whitespace boundary, UTF-8 boundary fallback; no text bytes discarded",
        "training_corpus_sha256": sha256(output),
    }
    print(f"Corpus length policy: {json.dumps(statistics)}", flush=True)
    return [output], statistics


def train_sentencepiece(
    corpus: list[Path], prefix: Path, model_type: str, vocab_size: int,
    *, threads: int, input_sentence_size: int = 0, max_sentence_length: int = 4192,
    hard_vocab_limit: bool = True,
) -> Path:
    spm.SentencePieceTrainer.train(
        input=",".join(str(p.resolve()) for p in corpus),
        model_prefix=str(prefix), model_type=model_type, vocab_size=vocab_size,
        character_coverage=1.0, byte_fallback=True,
        normalization_rule_name="identity", remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        input_sentence_size=input_sentence_size, shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
        train_extremely_large_corpus=True, num_threads=threads,
        hard_vocab_limit=hard_vocab_limit,
    )
    return prefix.with_suffix(".model")


def convert_sentencepiece(model_path: Path, json_path: Path) -> Tokenizer:
    proto = sp_pb2.ModelProto.FromString(model_path.read_bytes())
    if not proto.trainer_spec.byte_fallback:
        raise ValueError("SentencePiece model must have byte_fallback=True")
    spec = proto.normalizer_spec
    if spec.name != "identity" or spec.remove_extra_whitespaces or spec.add_dummy_prefix:
        raise ValueError("Expected identity normalization, preserved spaces and no dummy prefix")
    # Store the SPM space marker as the actual space in HF's vocabulary. IDs,
    # scores and merges stay intact; no whitespace-dropping pretokenizer is used.
    # This also lets a literal U+2581 in input use byte fallback without collision.
    vocab = [(piece.piece.replace("▁", " "), piece.score) for piece in proto.pieces]
    if [piece for piece, _ in vocab[:4]] != SPECIAL_TOKENS:
        raise ValueError("Unexpected SentencePiece special token IDs")
    if len({piece for piece, _ in vocab}) != len(vocab):
        raise ValueError("Space mapping produced duplicate vocabulary entries")
    if proto.trainer_spec.model_type == sp_pb2.TrainerSpec.UNIGRAM:
        model = models.Unigram(vocab, unk_id=1, byte_fallback=True)
    elif proto.trainer_spec.model_type == sp_pb2.TrainerSpec.BPE:
        indices = {piece: index for index, (piece, _) in enumerate(vocab)}
        # Recover candidate merges, ordered by SPM score (merge rank).
        merges = []
        for piece, score in vocab:
            for split in range(1, len(piece)):
                left, right = piece[:split], piece[split:]
                if left in indices and right in indices:
                    merges.append((left, right, score))
        merges.sort(key=lambda m: (-m[2], len(m[0]), len(m[1]), indices[m[0]], indices[m[1]]))
        model = models.BPE(
            indices, [(left, right) for left, right, _ in merges],
            unk_token="<unk>", byte_fallback=True, fuse_unk=True,
        )
    else:
        raise ValueError("Only SentencePiece Unigram and BPE are supported")
    tokenizer = Tokenizer(model)
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    tokenizer.add_special_tokens([
        AddedToken(token, normalized=False, special=True) for token in SPECIAL_TOKENS
    ])
    tokenizer.save(str(json_path))
    return tokenizer


def validate_tokenizer(path: Path, texts: list[str], expected_size: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size() != expected_size:
        raise ValueError(f"{path.name}: expected exactly {expected_size} tokens")
    if data["model"].get("byte_fallback") is not True:
        raise ValueError(f"{path.name}: missing model.byte_fallback=true")
    decoder = data.get("decoder") or {}
    if decoder.get("type") != "Sequence" or not any(
        item.get("type") == "ByteFallback" for item in decoder.get("decoders", [])
    ):
        raise ValueError(f"{path.name}: missing ByteFallback decoder")
    for index, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(token) != index:
            raise ValueError(f"{path.name}: {token} must have ID {index}")
    for byte in range(256):
        if tokenizer.token_to_id(f"<0x{byte:02X}>") is None:
            raise ValueError(f"{path.name}: missing fallback byte {byte}")
    result = {"documents": len(texts), "edge_cases": len(EDGE_TEXTS), "bad": 0,
              "roundtrip_failures": 0, "unk_documents": 0, "vocab_size": expected_size}
    for text in texts + EDGE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        mismatch = tokenizer.decode(ids) != text
        unknown = 1 in ids
        result["roundtrip_failures"] += int(mismatch)
        result["unk_documents"] += int(unknown)
        result["bad"] += int(mismatch or unknown)
    if result["bad"]:
        raise ValueError(f"{path.name}: validation failed: {result}")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Exact HF source used for the previous training")
    source.add_argument("--corpus", nargs="+", type=Path, help="Original prepared SPM corpus shards")
    parser.add_argument("--revision", help="Immutable HF dataset commit SHA")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--validation-texts", type=Path, help="JSONL: raw strings or objects with text")
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_tokenizers/v2_byte_fallback")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--input-sentence-size", type=int, default=0,
                        help="0 uses all prepared training lines (default)")
    parser.add_argument("--max-sentence-length", default="auto",
                        help="UTF-8 byte limit: auto uses corpus p99 (4192..65536); longer lines are chunked")
    args = parser.parse_args(argv)
    if args.dataset and (not args.revision or not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision)):
        parser.error("--dataset requires an immutable 40-character --revision commit SHA")
    if args.corpus and not args.validation_texts:
        parser.error("--corpus requires --validation-texts with unmodified documents")
    if args.validation_count < 200:
        parser.error("--validation-count must be at least 200")
    if args.threads < 1 or args.input_sentence_size < 0 or args.input_sentence_size == 1:
        parser.error("Invalid thread count or SentencePiece sample size")
    if args.max_sentence_length != "auto" and (
        not args.max_sentence_length.isdigit() or not 4 <= int(args.max_sentence_length) <= 2**31 - 1
    ):
        parser.error("--max-sentence-length must be 'auto' or an integer from 4 to 2147483647")
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Failed runs cannot leave apparently finished artifacts in the delivery folder.
    with tempfile.TemporaryDirectory(prefix=".spm-build-", dir=args.output_dir.parent) as temporary:
        work = Path(temporary)
        if args.dataset:
            corpus, texts, provenance = prepare_dataset(args, work)
        else:
            corpus = [p.resolve() for p in args.corpus]
            texts = load_validation_texts(args.validation_texts, args.validation_count)
            provenance = {"files": [{"path": str(p), "sha256": sha256(p)} for p in corpus],
                          "preparation": "existing shards, unchanged"}
        if args.validation_texts and args.dataset:
            texts = load_validation_texts(args.validation_texts, args.validation_count)
        if len(texts) < args.validation_count:
            raise ValueError(f"Only {len(texts)} validation documents; need {args.validation_count}")
        corpus, length_policy = prepare_training_lines(corpus, work, args.max_sentence_length)
        deliverables = work / "deliverables"
        deliverables.mkdir()
        samples = deliverables / "validation_texts.jsonl"
        samples.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
        report = {
            "source": provenance,
            "versions": {name: importlib.metadata.version(name) for name in ("sentencepiece", "tokenizers", "protobuf", "datasets")},
            "training": {"byte_fallback": True, "normalization_rule_name": "identity",
                         "remove_extra_whitespaces": False, "add_dummy_prefix": False,
                         "input_sentence_size": args.input_sentence_size, "threads": args.threads,
                         "length_policy": length_policy},
            "special_tokens": dict(zip(SPECIAL_TOKENS, range(4))),
            "validation_texts_sha256": sha256(samples), "models": {},
        }
        for model_type, size in MODEL_SPECS:
            name = f"spm_{model_type}_{size}"
            print(f"Training {name}", flush=True)
            model_path = train_sentencepiece(
                corpus, deliverables / name, model_type, size,
                threads=args.threads, input_sentence_size=args.input_sentence_size,
                max_sentence_length=length_policy["max_sentence_length"],
            )
            json_path = deliverables / f"{name}.json"
            convert_sentencepiece(model_path, json_path)
            result = validate_tokenizer(json_path, texts, size)
            result["sha256"] = sha256(json_path)
            report["models"][name] = result
            print(f"{name}: {result}", flush=True)
        (deliverables / "validation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        args.output_dir.mkdir(exist_ok=False)
        for path in deliverables.iterdir():
            path.rename(args.output_dir / path.name)
    print(f"All four models passed. Delivery: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
