import hmac
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from rag_platform.store import Store, Embeddings


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|lexical)$")
    mode: str = Field(default="extractive", pattern="^(extractive|local-llm)$")


def create_app(store=None):
    store = store or Store(
        Path(os.environ.get("RAG_DB", "work/rag.db")),
        Embeddings(os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")),
    )
    app = FastAPI(title="Evidence Retrieval & Evaluation")
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    from rag_platform.observability import instrument

    instrument(app, 16000)
    buckets = defaultdict(deque)

    def auth_tenant(token):
        # Public-only default. Private tenant identities are mapped by server secret config, not user labels.
        identities = json.loads(os.environ.get("RAG_TENANT_TOKENS", "{}"))
        if not token:
            return "public"
        for tenant, expected in identities.items():
            if hmac.compare_digest(token, expected):
                return tenant
        raise HTTPException(401, "Unknown tenant token")

    @app.middleware("http")
    async def limit(request: Request, call_next):
        from fastapi.responses import JSONResponse

        key = request.client.host if request.client else "local"
        now = time.monotonic()
        q = buckets[key]
        while q and q[0] < now - 60:
            q.popleft()
        if len(q) >= 120:
            return JSONResponse({"detail": "Rate limit"}, status_code=429)
        q.append(now)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Elapsed-Ms"] = str(round((time.perf_counter() - started) * 1000, 3))
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    def health():
        store.integrity()
        store.embedding.encode(["readiness"])
        return {"ready": True, "embedding": store.embedding.name, "mode": "extractive"}

    @app.post("/query")
    def query(q: Query, x_tenant_token: str | None = Header(default=None)):
        tenant = auth_tenant(x_tenant_token)
        if q.mode == "local-llm":
            if os.environ.get("ENABLE_LOCAL_LLM") != "1":
                raise HTTPException(503, "Local generation is disabled; prepare model and enable explicitly")
            from rag_platform.generator import LocalGenerator, grounded_answer

            if not hasattr(app.state, "generator"):
                app.state.generator = LocalGenerator(local_files_only=True)
            return grounded_answer(store, app.state.generator, q.text, q.k, q.strategy, tenant)
        return store.answer(q.text, q.k, q.strategy, tenant)

    @app.post("/search")
    def search(q: Query, x_tenant_token: str | None = Header(default=None)):
        return {"results": store.search(q.text, q.k, q.strategy, auth_tenant(x_tenant_token))}

    @app.delete("/documents/{doc_id}")
    def delete(doc_id: str, authorization: str | None = Header(default=None)):
        secret = os.environ.get("RAG_ADMIN_TOKEN")
        if not secret or not authorization or not hmac.compare_digest(authorization, "Bearer " + secret):
            raise HTTPException(403, "Admin token required")
        store.delete(doc_id)
        return {"deleted": doc_id}

    @app.get("/", response_class=HTMLResponse)
    def ui():
        return (Path(__file__).parent / "ui.html").read_text()

    @app.get("/evaluation")
    def evaluation():
        path = Path("evidence/retrieval_evaluation.json")
        return json.loads(path.read_text()) if path.exists() else {"status": "Run evaluation first"}

    return app


app = create_app()
