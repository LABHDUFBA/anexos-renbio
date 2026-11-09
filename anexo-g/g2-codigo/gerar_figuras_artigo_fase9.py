#!/usr/bin/env python3
"""Gera as duas figuras do artigo reduzido a partir dos produtos auditados.

Figura 1: série mensal das mensagens do corpus canônico.
Figura 2: principais associações locais G² entre os textos distintos.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2C5F8A"
LIGHT_BLUE = "#B9D4E8"
DARK = "#24313B"
GRID = "#D8DEE3"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def figure_timeline(input_csv: Path, output: Path) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    frame = frame.loc[frame["corpus"] == "A-canônico", ["month", "count"]].copy()
    if frame.empty:
        raise ValueError("Série mensal do Corpus A não encontrada")
    frame["month"] = pd.to_datetime(frame["month"], format="%Y-%m")
    full = pd.date_range(frame["month"].min(), frame["month"].max(), freq="MS")
    series = frame.set_index("month")["count"].reindex(full, fill_value=0)
    if int(series.sum()) != 4060:
        raise ValueError(f"Soma mensal inesperada: {int(series.sum())}")

    peak_month = series.idxmax()
    peak_count = int(series.max())

    fig, ax = plt.subplots(figsize=(7.1, 3.5), constrained_layout=True)
    ax.fill_between(series.index, series.values, color=LIGHT_BLUE, alpha=0.65, linewidth=0)
    ax.plot(series.index, series.values, color=BLUE, linewidth=1.5)
    ax.scatter([peak_month], [peak_count], color="#9E3D35", s=24, zorder=3)
    ax.annotate(
        f"Pico mensal: {peak_count}\n{peak_month.strftime('%m/%Y')}",
        xy=(peak_month, peak_count),
        xytext=(12, -34),
        textcoords="offset points",
        fontsize=8,
        color=DARK,
        arrowprops={"arrowstyle": "-", "color": "#7A8790", "lw": 0.8},
    )
    ax.set_ylabel("Mensagens")
    ax.set_xlabel("Mês de publicação")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return {
        "sum": int(series.sum()),
        "start": series.index.min().strftime("%Y-%m"),
        "end": series.index.max().strftime("%Y-%m"),
        "peak_month": peak_month.strftime("%Y-%m"),
        "peak_count": peak_count,
    }


def figure_g2(input_csv: Path, output: Path, top_n: int = 15) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    frame = frame.loc[(frame["corpus"] == "B") & (frame["word"] != "charles")]
    frame = frame.sort_values(["rank", "word"]).head(top_n).copy()
    if len(frame) != top_n:
        raise ValueError(f"Esperadas {top_n} associações; encontradas {len(frame)}")
    frame = frame.sort_values("g2", ascending=True)

    fig, ax = plt.subplots(figsize=(7.1, 4.8), constrained_layout=True)
    bars = ax.barh(frame["word"], frame["g2"], color=BLUE, edgecolor="none")
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in frame["g2"]], padding=3, fontsize=7)
    ax.set_xlabel("Razão de verossimilhança G²")
    ax.set_ylabel("Termo na janela de ±5 tokens")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, metadata={"Software": "matplotlib"})
    plt.close(fig)
    top = frame.sort_values("rank").iloc[0]
    return {"n": len(frame), "top_word": str(top["word"]), "top_g2": float(top["g2"])}


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "metric", "value"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            for metric, value in row["metrics"].items():
                writer.writerow({"figure": row["figure"], "metric": metric, "value": value})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figuras"))
    parser.add_argument("--manifest", type=Path, default=Path("data/figuras_fase9_manifesto.csv"))
    args = parser.parse_args()

    set_style()
    fig1 = args.output_dir / "fig01_timeline_mensal_corpus_canonico.png"
    fig2 = args.output_dir / "fig02_associacoes_g2_textos_distintos.png"
    metrics1 = figure_timeline(args.monthly, fig1)
    metrics2 = figure_g2(args.g2, fig2)
    write_manifest(
        args.manifest,
        [
            {"figure": f"figuras/{fig1.name}", "metrics": metrics1},
            {"figure": f"figuras/{fig2.name}", "metrics": metrics2},
        ],
    )
    print(fig1)
    print(fig2)
    print(args.manifest)


if __name__ == "__main__":
    main()
