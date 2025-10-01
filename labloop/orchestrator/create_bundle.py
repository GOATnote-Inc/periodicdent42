from __future__ import annotations

from pathlib import Path

import typer
from pydantic import parse_file_as

from .bundle import BundleBuilder
from .models.schemas import RunRecord

app = typer.Typer()


@app.command()
def bundle(
    run_id: str = typer.Argument(..., help="Run identifier"),
    output: Path = typer.Option(Path("bundle.zip"), help="Output archive path"),
) -> None:
    data_dir = Path("labloop_data")
    record_path = data_dir / "data" / "runs" / f"{run_id}.json"
    if not record_path.exists():
        raise typer.BadParameter(f"Run record not found at {record_path}")
    record = parse_file_as(RunRecord, record_path)
    builder = BundleBuilder(data_dir)
    builder.build(record, output)
    typer.echo(f"Bundle written to {output}")


if __name__ == "__main__":
    app()
