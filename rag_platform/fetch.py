"""Public data download with atomic publication, limits, digest and provenance."""

import hashlib
import json
import os
from pathlib import Path
import requests


def download(url: str, target: Path, max_bytes: int = 600_000_000) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with tmp.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("Download exceeds configured size limit")
                    digest.update(chunk)
                    output.write(chunk)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    result = {"url": url, "sha256": digest.hexdigest(), "bytes": size}
    target.with_suffix(target.suffix + ".provenance.json").write_text(json.dumps(result, indent=2))
    return result
