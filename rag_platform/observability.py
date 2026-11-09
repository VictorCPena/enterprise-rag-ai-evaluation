"""Safe operational events, no tokens, documents, queries or historical payloads."""

import json
import logging
import time
from threading import Lock
from fastapi.responses import Response, JSONResponse


def instrument(app, max_body):
    logger = logging.getLogger("portfolio.operations")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
    lock = Lock()
    values = {"requests": 0, "errors": 0, "seconds": 0.0}

    @app.middleware("http")
    async def telemetry(request, call_next):
        try:
            size = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "Invalid length"}, status_code=400)
        if size > max_body:
            return JSONResponse({"detail": "Payload too large"}, status_code=413)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        with lock:
            values["requests"] += 1
            values["errors"] += int(response.status_code >= 500)
            values["seconds"] += elapsed
        # Route templates avoid logging arbitrary user-controlled path segments.
        route = request.scope.get("route")
        label = getattr(route, "path", "unknown")
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "route": label,
                    "status": response.status_code,
                    "seconds": round(elapsed, 6),
                }
            )
        )
        return response

    @app.get("/metrics")
    def metrics():
        with lock:
            data = dict(values)
        return Response(
            f"portfolio_requests_total {data['requests']}\nportfolio_errors_total {data['errors']}\nportfolio_request_seconds_sum {data['seconds']}\n",
            media_type="text/plain",
        )
