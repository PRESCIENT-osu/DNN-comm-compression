from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_storage_dir: Path = Path("metrics_data")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Metrics server started, storage_dir=%s", _storage_dir)
    _storage_dir.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Metrics server shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    """Return server health status."""
    return JSONResponse({"status": "ok"})


@app.post("/metrics")
async def ingest_metrics(request: Request) -> JSONResponse:
    """Ingest a batch of metric events and append them to per-experiment NDJSON files.

    Accepts a JSON array of event objects.  Each event must contain
    ``event_type`` and ``experiment_id`` fields.  Events are appended to
    ``{storage_dir}/{experiment_id}/{event_type}.ndjson``.

    Args:
        request: HTTP request carrying a JSON array of event dicts.

    Returns:
        JSON object with ``accepted`` count.
    """
    try:
        events: list[dict[str, Any]] = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array of events")

    accepted = 0
    for event in events:
        experiment_id = event.get("experiment_id")
        event_type = event.get("event_type")
        if not experiment_id or not event_type:
            logger.warning(
                "Skipping event missing experiment_id or event_type: %s", event
            )
            continue
        _append_event(experiment_id, event_type, event)
        accepted += 1

    return JSONResponse({"accepted": accepted})


@app.get("/experiments")
async def list_experiments() -> JSONResponse:
    """List all experiment IDs that have stored metrics.

    Returns:
        JSON object with ``experiments`` list.
    """
    if not _storage_dir.exists():
        return JSONResponse({"experiments": []})
    experiments = [d.name for d in _storage_dir.iterdir() if d.is_dir()]
    return JSONResponse({"experiments": sorted(experiments)})


@app.get("/experiments/{experiment_id}")
async def list_event_types(experiment_id: str) -> JSONResponse:
    """List available event types for a given experiment.

    Args:
        experiment_id: The experiment to query.

    Returns:
        JSON object with ``event_types`` list.
    """
    exp_dir = _storage_dir / experiment_id
    if not exp_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Experiment '{experiment_id}' not found"
        )
    event_types = [f.stem for f in exp_dir.glob("*.ndjson")]
    return JSONResponse(
        {"experiment_id": experiment_id, "event_types": sorted(event_types)}
    )


@app.get("/metrics/query")
async def query_metrics(request: Request) -> JSONResponse:
    """Query metric events across experiments with optional filters.

    Scans all experiment directories (or a filtered subset), reads the
    requested event type NDJSON file, and applies equality filters on event
    fields.  Returns up to ``limit`` matching events.

    Query parameters:
        event_type (required): NDJSON filename stem to read (e.g. ``link_probe``).
        experiment_name_contains (optional): Only scan experiment directories
            whose name contains this substring.  Use the profile name suffix
            (e.g. ``"100mbps"``) to scope probe events to a specific hardware
            profile without modifying InfraConfig.
        limit (optional): Maximum number of events to return (default: 1000).
        <field>=<value> (optional): Any additional query parameters are treated
            as equality filters on event fields.  Values are compared as strings,
            so ``from_node=A`` matches events where ``event["from_node"] == "A"``.

    Returns:
        JSON object with ``events`` list and ``truncated`` bool flag.

    Example::

        GET /metrics/query?event_type=link_probe&from_node=A&to_node=B
            &experiment_name_contains=100mbps&limit=500
    """
    params = dict(request.query_params)

    event_type = params.pop("event_type", None)
    if not event_type:
        raise HTTPException(
            status_code=400, detail="event_type query parameter is required"
        )

    name_contains: str | None = params.pop("experiment_name_contains", None)
    try:
        limit = int(params.pop("limit", "1000"))
    except ValueError:
        raise HTTPException(
            status_code=400, detail="limit must be an integer"
        ) from None

    # Remaining params are field equality filters.
    field_filters: dict[str, str] = params

    results: list[dict[str, Any]] = []
    truncated = False

    if _storage_dir.exists():
        for exp_dir in sorted(_storage_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if name_contains and name_contains not in exp_dir.name:
                continue
            ndjson_path = exp_dir / f"{event_type}.ndjson"
            if not ndjson_path.exists():
                continue
            with open(ndjson_path) as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if all(
                        str(event.get(k)) == str(v) for k, v in field_filters.items()
                    ):
                        results.append(event)
                        if len(results) >= limit:
                            truncated = True
                            break
            if truncated:
                break

    return JSONResponse({"events": results, "truncated": truncated})


def _append_event(experiment_id: str, event_type: str, event: dict[str, Any]) -> None:
    """Append a single event to the appropriate NDJSON file.

    Args:
        experiment_id: Experiment the event belongs to.
        event_type: Type of the event, used as filename stem.
        event: Event dict to serialise and append.
    """
    exp_dir = _storage_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / f"{event_type}.ndjson"
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def main() -> None:
    """Entry point for the metrics server CLI."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Central metrics ingestion server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9100, help="Bind port")
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path(os.getenv("METRICS_STORAGE_DIR", "metrics_data")),
        help="Directory for NDJSON storage (default: metrics_data)",
    )
    args = parser.parse_args()

    global _storage_dir
    _storage_dir = args.storage_dir

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
