from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

import httpx
import typer
import yaml

app = typer.Typer(help="LabLoop CLI")


def _api_url() -> str:
    return os.getenv("LABLOOP_API", "http://127.0.0.1:8000")


def _client() -> httpx.Client:
    return httpx.Client(base_url=_api_url(), timeout=60)


@app.command()
def submit_plan(path: Path) -> None:
    """Submit an experiment plan defined in YAML."""
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    with _client() as client:
        resp = client.post("/experiments", json=content)
        resp.raise_for_status()
        data = resp.json()
    typer.echo(json.dumps(data, indent=2))


@app.command()
def run_loop(run_id: str, max_steps: int = typer.Option(10, min=1)) -> None:
    """Trigger the scheduler repeatedly until completion or max steps."""
    async def _loop() -> None:
        async with httpx.AsyncClient(base_url=_api_url(), timeout=60) as client:
            for step in range(max_steps):
                resp = await client.post("/actions/run-next", json={"run_id": run_id})
                if resp.status_code != 200:
                    typer.echo(f"Stop: {resp.text}")
                    break
                data = resp.json()
                typer.echo(f"Step {step+1}: {json.dumps(data)}")
                status = await client.get(f"/experiments/{run_id}")
                status.raise_for_status()
                payload = status.json()
                if payload["record"]["status"] in {"completed", "aborted"}:
                    typer.echo(f"Run finished with status {payload['record']['status']}")
                    break
                await asyncio.sleep(0.1)
    asyncio.run(_loop())


@app.command()
def abort(run_id: str, reason: str = "cli abort") -> None:
    with _client() as client:
        resp = client.post("/abort", json={"reason": reason})
        if resp.status_code != 200:
            typer.echo(resp.text)
            raise typer.Exit(code=1)
        typer.echo(resp.text)


if __name__ == "__main__":
    app()
