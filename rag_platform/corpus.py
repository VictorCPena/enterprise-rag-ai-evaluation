"""Pinned upstream documentation with license attribution and checksums."""

import json
from pathlib import Path
from rag_platform.fetch import download

SKLEARN = [
    "model_evaluation",
    "cross_validation",
    "preprocessing",
    "pipeline",
    "model_persistence",
    "ensemble",
    "feature_extraction",
    "common_pitfalls",
]
PYTHON = ["sqlite3", "csv", "asyncio-task", "statistics"]


def fetch(directory: Path):
    sources = []
    for name in SKLEARN:
        relative = (
            f"doc/modules/{name}.rst"
            if name not in ["model_persistence", "common_pitfalls"]
            else f"doc/{name}.rst"
        )
        if name == "pipeline":
            relative = "doc/modules/compose.rst"
        url = f"https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.6.1/{relative}"
        p = directory / f"sklearn-{name}.rst"
        meta = download(url, p, 10_000_000)
        sources.append(
            {
                "id": f"sklearn-{name}",
                "title": name.replace("_", " "),
                "path": str(p),
                "url": url,
                "license": "BSD-3-Clause",
                "sha256": meta["sha256"],
            }
        )
    for name in PYTHON:
        url = f"https://raw.githubusercontent.com/python/cpython/v3.12.8/Doc/library/{name}.rst"
        p = directory / f"python-{name}.rst"
        meta = download(url, p, 10_000_000)
        sources.append(
            {
                "id": f"python-{name}",
                "title": name,
                "path": str(p),
                "url": url,
                "license": "PSF-2.0",
                "sha256": meta["sha256"],
            }
        )
    for name, url in [
        ("LICENSE-SKLEARN", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.6.1/COPYING"),
        ("LICENSE-PYTHON", "https://raw.githubusercontent.com/python/cpython/v3.12.8/LICENSE"),
    ]:
        download(url, directory / name, 1_000_000)
    (directory / "manifest.json").write_text(json.dumps(sources, indent=2))
    return sources
