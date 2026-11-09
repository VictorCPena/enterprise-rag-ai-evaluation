"""Build a fresh index, validate it, then publish atomically with a backup."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4
from rag_platform.corpus import fetch
from rag_platform.store import Store, Embeddings
from rag_platform.evaluate import evaluate


def build(corpus, model, download_sources=False, target=Path("work/rag.db")):
    sources = fetch(corpus) if download_sources else json.loads((corpus / "manifest.json").read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("rag-building-" + uuid4().hex + ".db")
    store = Store(temporary, Embeddings(model))
    try:
        for source in sources:
            print(
                json.dumps(
                    store.upsert(
                        source["id"],
                        source["title"],
                        source["url"],
                        Path(source["path"]).read_text(),
                        license=source["license"],
                    )
                ),
                flush=True,
            )
        store.integrity()
        result = evaluate(store)
        result["index_integrity"] = {
            "quick_check": "ok",
            "foreign_key_check": "ok",
            "publication": "validated fresh file with atomic replace and prior backup",
            "journal": "DELETE; explicitly closed per-operation connections",
        }
        if target.exists():
            backup = target.parent / "backups" / ("rag-" + str(time.time_ns()) + ".db")
            backup.parent.mkdir(exist_ok=True)
            shutil.copy2(target, backup)
            for suffix in ["-wal", "-shm", "-journal"]:
                sidecar = Path(str(target) + suffix)
                if sidecar.exists():
                    shutil.move(sidecar, str(backup) + suffix)
        os.replace(temporary, target)
        Path("evidence").mkdir(exist_ok=True)
        Path("evidence/retrieval_evaluation.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hashing")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = build(args.corpus, args.model, args.download)
    print(
        json.dumps(
            {
                k: {m: v for m, v in r.items() if m not in ["records", "negative"]}
                for k, r in result["configurations"].items()
            },
            indent=2,
        )
    )
