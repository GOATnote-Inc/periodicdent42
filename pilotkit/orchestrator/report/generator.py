from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analytics.stats import bootstrap_percent_change, percent_change, select_test


class PilotReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _to_df(self, rows: Iterable[dict]) -> pd.DataFrame:
        df = pd.DataFrame(list(rows))
        if df.empty:
            return df
        df["period"] = pd.to_datetime(df["period"])
        return df

    def generate(self, baseline_rows: Iterable[dict], pilot_rows: Iterable[dict], title: str) -> dict:
        baseline_df = self._to_df(baseline_rows)
        pilot_df = self._to_df(pilot_rows)
        chart_path = self.output_dir / f"impact-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.png"
        markdown_path = chart_path.with_suffix(".md")
        if baseline_df.empty or pilot_df.empty:
            chart_path.write_text("No data to chart")
            markdown_path.write_text("No data available")
            return {"chart": chart_path, "markdown": markdown_path}

        cycle_pct, cycle_ci = bootstrap_percent_change(
            baseline_df["cycle_time_s"], pilot_df["cycle_time_s"]
        )
        yield_pct = percent_change(baseline_df["yield_ok"], pilot_df["yield_ok"])
        test_name, p_value = select_test(
            baseline_df["cycle_time_s"], pilot_df["cycle_time_s"]
        )

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(
            [baseline_df["cycle_time_s"], pilot_df["cycle_time_s"]],
            label=["Baseline", "Pilot"],
            bins=15,
            color=["#8888ff", "#88ff88"],
            alpha=0.7,
        )
        ax[0].set_title("Cycle Time Distribution")
        ax[0].set_xlabel("Seconds")
        ax[0].legend()

        baseline_yield = baseline_df.groupby("period")["yield_ok"].mean()
        pilot_yield = pilot_df.groupby("period")["yield_ok"].mean()
        ax[1].plot(baseline_yield.index, baseline_yield.values, label="Baseline")
        ax[1].plot(pilot_yield.index, pilot_yield.values, label="Pilot")
        ax[1].set_title("Yield Trend")
        ax[1].set_ylim(0, 1)
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(chart_path)
        plt.close(fig)

        markdown_lines = [
            f"# {title}",
            "",
            f"- Cycle time median change: {cycle_pct:.1f}% (CI {cycle_ci[0]:.1f}% to {cycle_ci[1]:.1f}%)",
            f"- Yield delta: {yield_pct:.1f}%",
            f"- Statistical test: {test_name} (p={p_value:.4f})",
            "",
            "## Funnel & Defects",
        ]
        defect_counts = pilot_df["defect_code"].value_counts(dropna=False).to_dict()
        for code, count in defect_counts.items():
            markdown_lines.append(f"- {code or 'ok'}: {count}")
        markdown_lines.append("")
        markdown_lines.append(f"![Pilot Impact Chart]({chart_path.name})")
        markdown_path.write_text("\n".join(markdown_lines))
        return {"chart": chart_path, "markdown": markdown_path}
