"""SQLite transactional document store, exact vector search and filtered hybrid retrieval."""

from __future__ import annotations
import hashlib
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer

BAD = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior)|system\s+prompt|reveal.{0,30}(?:secret|password)|you are now|<\|(?:system|assistant)",
    re.I,
)
TOKEN = re.compile(r"[a-zA-Z0-9_]+(?:['’][a-zA-Z]+)?")
NUMBER = re.compile(r"\b\d+(?:\.\d+)*\b")
MIN_SPAN_COVERAGE = 0.4


STOP = set(
    "the a an is are was were be been to for of and or in on with how what when who does do can i use using it this that from into why will all my as by during".split()
)


def tokens(text):
    # Do not mistake won't for the distinct word won; normalize apostrophes.
    return [w.replace("’", "'") for w in TOKEN.findall(text.lower()) if w not in STOP]


def chunks(text, size=1100, overlap=150):
    if not 100 <= size <= 5000 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk settings")
    # Preserve exact normalized character offsets for citation verification.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        body = text[start:end]
        if len(body.strip()) > 40:
            yield start, end, body
        if end == len(text):
            break


class Embeddings:
    def __init__(self, model="hashing"):
        from threading import Lock

        self.name = model
        self.lock = Lock()
        self.model = None
        if model == "hashing":
            self.vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")

    def encode(self, texts):
        if self.name == "hashing":
            return self.vectorizer.transform(texts).toarray().astype("float32")
        with self.lock:
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    self.name,
                    device="cpu",
                    trust_remote_code=False,
                    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
                    if self.name == "sentence-transformers/all-MiniLM-L6-v2"
                    else None,
                )
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(
                "float32"
            )


class Store:
    def __init__(self, path: Path, embeddings=None, neural_reranker=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding = embeddings or Embeddings()
        self.neural_reranker = neural_reranker
        with self.connect() as c:
            c.executescript("""PRAGMA journal_mode=DELETE;
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,title TEXT,url TEXT,sha TEXT,tenant TEXT,license TEXT,model TEXT);
            CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,doc TEXT REFERENCES documents(id) ON DELETE CASCADE,start INTEGER,end INTEGER,text TEXT,vector BLOB,blocked INTEGER);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def integrity(self):
        with self.connect() as connection:
            check = [row[0] for row in connection.execute("PRAGMA quick_check")]
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if check != ["ok"] or foreign_keys:
            raise RuntimeError("Index integrity check failed; rebuild from source into a new file")
        return True

    def upsert(self, doc_id, title, url, text, tenant="public", license="unknown", size=1100, overlap=150):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", doc_id):
            raise ValueError("Invalid document ID")
        if len(text.encode()) > 5_000_000:
            raise ValueError("Document too large")
        if not url.startswith("https://"):
            raise ValueError("Source URL must be HTTPS")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = list(chunks(text, size, overlap))
        if len(parts) > 6000:
            raise ValueError("Chunk budget exceeded")
        vectors = self.embedding.encode([p[2] for p in parts]) if parts else []
        sha = hashlib.sha256(text.encode()).hexdigest()
        with self.connect() as c:
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            c.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?,?,?)",
                (doc_id, title, url, sha, tenant, license, self.embedding.name),
            )
            for (start, end, body), vec in zip(parts, vectors, strict=True):
                key = hashlib.sha256(f"{doc_id}:{sha}:{start}:{end}".encode()).hexdigest()[:24]
                c.execute(
                    "INSERT INTO chunks VALUES(?,?,?,?,?,?,?)",
                    (key, doc_id, start, end, body, vec.tobytes(), int(bool(BAD.search(body)))),
                )
        return {"document": doc_id, "chunks": len(parts), "sha256": sha}

    def delete(self, doc_id):
        with self.connect() as c:
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def search(self, query, k=5, strategy="hybrid", tenant="public", rerank=True):
        if not 1 <= k <= 20 or not 1 <= len(query) <= 1000:
            raise ValueError("Query limits exceeded")
        if strategy not in ["lexical", "vector", "hybrid"]:
            raise ValueError("Unknown strategy")
        if BAD.search(query):
            return []
        # Authorization BEFORE candidate generation prevents cross-tenant scoring leakage.
        with self.connect() as c:
            rows = c.execute(
                "SELECT c.*,d.title,d.url,d.sha,d.license FROM chunks c JOIN documents d ON d.id=c.doc WHERE c.blocked=0 AND d.tenant IN (?,?) AND d.model=?",
                ("public", tenant, self.embedding.name),
            ).fetchall()
        if not rows:
            return []
        q = tokens(query)
        docs = [tokens(r["text"]) for r in rows]
        bm = np.asarray(BM25Okapi(docs).get_scores(q))
        vectors = np.stack([np.frombuffer(r["vector"], dtype="float32") for r in rows])
        dense = vectors @ self.embedding.encode([query])[0]
        lexical_order = np.argsort(-bm, kind="stable")
        dense_order = np.argsort(-dense, kind="stable")
        if strategy == "lexical":
            score = bm.copy()
        elif strategy == "vector":
            score = dense.copy()
        else:
            score = np.zeros(len(rows))
            for order in [lexical_order, dense_order]:
                for rank, idx in enumerate(order[:50]):
                    score[idx] += 1 / (60 + rank + 1)
        order = np.argsort(-score, kind="stable")[: max(k * 4, 20)]
        if rerank:
            # Transparent heuristic reranker. Neural reranking is an optional comparison.
            order = sorted(
                order,
                key=lambda i: (
                    float(score[i]) + 0.02 * len(set(q) & set(tokens(rows[i]["title"]))) / max(len(set(q)), 1)
                ),
                reverse=True,
            )
        if rerank and self.neural_reranker is not None:
            values = self.neural_reranker.score(query, [rows[i]["text"] for i in order])
            order = [order[j] for j in np.argsort(-values, kind="stable")]
        results = []
        for i in order:
            overlap = len(set(q) & set(docs[i])) / max(len(set(q)), 1)
            if bm[i] <= 0 and overlap == 0:
                continue
            r = dict(rows[i])
            r.pop("vector")
            r.update(
                {
                    "score": float(score[i]),
                    "cosine": float(dense[i]),
                    "lexical": float(bm[i]),
                    "query_coverage": overlap,
                }
            )
            results.append(r)
            if len(results) >= k:
                break
        return results

    def evidence_hits(self, query, k=5, strategy="hybrid", tenant="public"):
        """Filter exact source spans independently of retrieval scores.

        A conservative lexical heuristic, shared by all strategies and applied
        to each citation. It can reject valid paraphrases and does not certify
        semantic entailment or universal out-of-domain detection.
        """
        found = self.search(query, k, strategy, tenant)
        terms = set(tokens(query))
        numbers = set(NUMBER.findall(query))
        supported = []
        if not terms:
            return supported
        minimum_matches = min(2, len(terms))
        seen_excerpts = set()
        for r in found:
            candidates = []
            for line in re.split(r"(?<=[.!?])\s+|\n\s*\n", r["text"]):
                quote = line.strip()
                matched = len(terms & set(tokens(quote)))
                coverage = matched / len(terms)
                if (
                    len(quote) > 25
                    and matched >= minimum_matches
                    and coverage >= MIN_SPAN_COVERAGE
                    and numbers.issubset(NUMBER.findall(quote))
                ):
                    candidates.append((coverage, quote))
            if not candidates:
                continue
            coverage, quote = max(candidates, key=lambda x: (x[0], -len(x[1])))
            # Overlapping chunks can surface the same excerpt from one source.
            excerpt = (r["doc"], r["sha"], quote)
            if excerpt in seen_excerpts:
                continue
            seen_excerpts.add(excerpt)
            pos = r["text"].find(quote)
            start = r["start"] + pos
            supported.append(
                {**r, "text": quote, "start": start, "end": start + len(quote), "query_coverage": coverage}
            )
        return supported

    def answer(self, query, k=5, strategy="hybrid", tenant="public"):
        found = self.evidence_hits(query, k, strategy, tenant)
        if not found:
            return {
                "answer": "Insufficient evidence in the indexed corpus.",
                "abstained": True,
                "citations": [],
                "mode": "extractive",
                "abstention_reason": "No source span met the lexical coverage and numeric-literal policy",
            }
        citations = []
        excerpts = []
        for r in found[:3]:
            quote = r["text"]
            cid = len(citations) + 1
            excerpts.append(f"{quote} [{cid}]")
            citations.append(
                {
                    "id": cid,
                    "doc": r["doc"],
                    "chunk": r["id"],
                    "url": r["url"],
                    "quote": quote,
                    "start": r["start"],
                    "end": r["end"],
                    "source_sha256": r["sha"],
                    "license": r["license"],
                }
            )
        return {
            "answer": "\n\n".join(excerpts) if excerpts else "Insufficient evidence in the indexed corpus.",
            "abstained": not bool(excerpts),
            "citations": citations,
            "mode": "extractive",
            "interpretation": "Retrieved source excerpts; no unsupported narrative synthesis",
        }
