#!/usr/bin/env python3
"""Gera estatísticas descritivas reproduzíveis para os corpora Darwin."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from detect_idioma import detect_language
from extrair_links_externos import (
    canonical_key,
    extract_raw_urls,
    host,
    is_telegram_url,
    structured_youtube_urls,
)

TZ = ZoneInfo("America/Sao_Paulo")
FORWARD_FIELDS = (
    "forward_from_chat",
    "forward_chat_id",
    "forward_date",
    "forward_from",
    "forward_sender_name",
    "forward_signature",
    "forward_user_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Corpus A-canônico")
    parser.add_argument("--b", type=Path, required=True, help="Corpus B")
    parser.add_argument("--c", type=Path, required=True, help="Corpus C formalizado")
    parser.add_argument("--c-occ", type=Path, required=True, help="Corpus C-recorrências")
    parser.add_argument("--c-original", type=Path, required=True, help="Corpus histórico original")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Arquivo inválido: {path}")
    return rows


def present(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, dict):
        return any(present(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(present(v) for v in value)
    return True


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def date_value(row: dict[str, Any], text_level: bool) -> int:
    field = "text_first_date" if text_level and row.get("text_first_date") is not None else "strict_date"
    value = row.get(field)
    if value is None:
        raise ValueError(f"Data ausente no campo {field}")
    return int(value)


def to_iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=TZ).isoformat()


def has_confirmed_forward_source(row: dict[str, Any]) -> bool:
    value = row.get("forward_from_chat")
    return isinstance(value, dict) and present(value.get("id"))


def external_link_map(row: dict[str, Any]) -> dict[str, str]:
    candidates = extract_raw_urls(row) + structured_youtube_urls(row)
    links: dict[str, dict[str, str]] = {}
    for url, source_field in candidates:
        if not url or is_telegram_url(url):
            continue
        key = canonical_key(url)
        # Preserva a URL efetivamente escrita na mensagem. O campo estruturado
        # youtube_id só preenche quando não há URL bruta equivalente.
        if key in links and links[key]["source_field"] != "youtube_id":
            continue
        links[key] = {"url": url, "source_field": source_field}
    return {key: item["url"] for key, item in links.items()}


def link_domain(key: str, url: str) -> str:
    _ = key
    return host(url) or "sem_dominio"


def peak(counter: Counter[str]) -> dict[str, Any]:
    if not counter:
        return {"period": None, "count": 0}
    period, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return {"period": period, "count": count}


def summarize(name: str, unit: str, rows: list[dict[str, Any]], text_level: bool, note: str) -> dict[str, Any]:
    total = len(rows)
    dates = [date_value(row, text_level) for row in rows]
    years = Counter(str(datetime.fromtimestamp(value, tz=TZ).year) for value in dates)
    months = Counter(datetime.fromtimestamp(value, tz=TZ).strftime("%Y-%m") for value in dates)
    types = Counter(str(row.get("type") or "ausente") for row in rows)
    languages = Counter(detect_language(str(row.get("all_text") or "")) for row in rows)
    chat_ids = {row.get("chat_id") for row in rows if present(row.get("chat_id"))}
    titles = {str(row.get("chat_title")) for row in rows if present(row.get("chat_title"))}
    title_counts = Counter(str(row.get("chat_title") or "ausente") for row in rows)
    view_values = [value for row in rows if (value := numeric(row.get("views"))) is not None]
    text_chars = [float(len(str(row.get("all_text") or ""))) for row in rows]
    text_words = [float(len(str(row.get("all_text") or "").split())) for row in rows]

    confirmed_forward = sum(has_confirmed_forward_source(row) for row in rows)
    any_forward = sum(any(present(row.get(field)) for field in FORWARD_FIELDS) for row in rows)
    legacy_chat_fields = sum(
        present(row.get("forward_from_chat")) or present(row.get("forward_chat_id")) for row in rows
    )

    rows_with_links = 0
    link_assignments = 0
    unique_links: set[str] = set()
    unique_domains: set[str] = set()
    for row in rows:
        links = external_link_map(row)
        if links:
            rows_with_links += 1
        link_assignments += len(links)
        unique_links.update(links)
        unique_domains.update(link_domain(key, url) for key, url in links.items())

    return {
        "corpus": name,
        "unit": unit,
        "note": note,
        "n": total,
        "period": {
            "min_timestamp": min(dates),
            "min_iso": to_iso(min(dates)),
            "max_timestamp": max(dates),
            "max_iso": to_iso(max(dates)),
        },
        "chat_ids_unique": len(chat_ids),
        "titles_unique": len(titles),
        "types": dict(sorted(types.items())),
        "languages": dict(sorted(languages.items())),
        "views": {
            **describe(view_values),
            "coverage_n": len(view_values),
            "coverage_percent": len(view_values) / total * 100,
        },
        "text_chars": describe(text_chars),
        "text_words": describe(text_words),
        "forward": {
            "confirmed_source_chat_id_n": confirmed_forward,
            "confirmed_source_chat_id_percent": confirmed_forward / total * 100,
            "any_substantive_metadata_n": any_forward,
            "any_substantive_metadata_percent": any_forward / total * 100,
            "legacy_chat_fields_n": legacy_chat_fields,
            "legacy_chat_fields_percent": legacy_chat_fields / total * 100,
        },
        "links": {
            "rows_with_external_links": rows_with_links,
            "rows_with_external_links_percent": rows_with_links / total * 100,
            "link_assignments": link_assignments,
            "unique_links": len(unique_links),
            "unique_domains": len(unique_domains),
        },
        "annual": dict(sorted(years.items())),
        "monthly": dict(sorted(months.items())),
        "peak_year": peak(years),
        "peak_month": peak(months),
        "top_titles": dict(title_counts.most_common(20)),
    }


def metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    name = summary["corpus"]
    unit = summary["unit"]
    n = summary["n"]
    rows: list[dict[str, Any]] = []

    def add(group: str, metric: str, value: Any, denominator: Any = "", percent: Any = "", note: str = "") -> None:
        rows.append(
            {
                "corpus": name,
                "unit": unit,
                "group": group,
                "metric": metric,
                "value": value,
                "denominator": denominator,
                "percent": percent,
                "note": note,
            }
        )

    add("base", "units", n, n, 100, summary["note"])
    add("period", "min_timestamp", summary["period"]["min_timestamp"], note=summary["period"]["min_iso"])
    add("period", "max_timestamp", summary["period"]["max_timestamp"], note=summary["period"]["max_iso"])
    add("chats", "unique_chat_ids", summary["chat_ids_unique"], n)
    add("chats", "unique_titles", summary["titles_unique"], n)
    for category, count in summary["types"].items():
        add("type", category, count, n, count / n * 100)
    for category, count in summary["languages"].items():
        add("language", category, count, n, count / n * 100)
    add("views", "coverage", summary["views"]["coverage_n"], n, summary["views"]["coverage_percent"])
    for metric in ("mean", "median", "p25", "p75", "min", "max"):
        add("views_valid", metric, summary["views"][metric], summary["views"]["n"], note="Somente unidades com views válido")
    for metric in ("mean", "median", "p25", "p75", "min", "max"):
        add("text_chars", metric, summary["text_chars"][metric], summary["text_chars"]["n"])
        add("text_words", metric, summary["text_words"][metric], summary["text_words"]["n"])
    add("forward", "confirmed_source_chat_id", summary["forward"]["confirmed_source_chat_id_n"], n, summary["forward"]["confirmed_source_chat_id_percent"])
    add("forward", "any_substantive_metadata", summary["forward"]["any_substantive_metadata_n"], n, summary["forward"]["any_substantive_metadata_percent"])
    add("forward", "legacy_chat_fields", summary["forward"]["legacy_chat_fields_n"], n, summary["forward"]["legacy_chat_fields_percent"])
    add("links", "rows_with_external_links", summary["links"]["rows_with_external_links"], n, summary["links"]["rows_with_external_links_percent"])
    add("links", "link_assignments", summary["links"]["link_assignments"], n)
    add("links", "unique_links", summary["links"]["unique_links"])
    add("links", "unique_domains", summary["links"]["unique_domains"])
    add("peaks", "peak_year", summary["peak_year"]["count"], n, summary["peak_year"]["count"] / n * 100, summary["peak_year"]["period"])
    add("peaks", "peak_month", summary["peak_month"]["count"], n, summary["peak_month"]["count"] / n * 100, summary["peak_month"]["period"])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def series_rows(summaries: list[dict[str, Any]], field: str, period_label: str) -> list[dict[str, Any]]:
    periods = sorted({period for summary in summaries for period in summary[field]})
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for period in periods:
            count = summary[field].get(period, 0)
            rows.append(
                {
                    "corpus": summary["corpus"],
                    "unit": summary["unit"],
                    period_label: period,
                    "count": count,
                    "denominator": summary["n"],
                    "percent": count / summary["n"] * 100,
                }
            )
    return rows


def top_title_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for rank, (title, count) in enumerate(summary["top_titles"].items(), start=1):
            rows.append(
                {
                    "corpus": summary["corpus"],
                    "unit": summary["unit"],
                    "rank": rank,
                    "chat_title": title,
                    "count": count,
                    "denominator": summary["n"],
                    "percent": count / summary["n"] * 100,
                }
            )
    return rows


def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%"


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    datasets = (
        ("A-canônico", "mensagem", load(args.a), False, "Uma linha por mensagem canônica; base para frequência e circulação."),
        ("B", "texto único", load(args.b), True, "Uma linha por texto exato; base para análise temática."),
        ("C", "texto histórico", load(args.c), True, "Subconjunto histórico formalizado com primeira ocorrência como representante."),
        ("C-recorrências", "mensagem histórica", load(args.c_occ), False, "Mensagens canônicas cujos textos pertencem a C."),
        ("C-original", "registro histórico", load(args.c_original), False, "Saída original da agregação composta; usada somente para reproduzir o draft atual."),
    )
    summaries = [summarize(name, unit, rows, text_level, note) for name, unit, rows, text_level, note in datasets]

    by_name = {summary["corpus"]: summary for summary in summaries}
    if by_name["A-canônico"]["n"] != 4060 or by_name["B"]["n"] != 2718 or by_name["C"]["n"] != 1269:
        raise ValueError("Contagens centrais divergiram dos manifestos")
    legacy = by_name["C-original"]
    expected_legacy = {
        "chat_ids_unique": 335,
        "titles_unique": 381,
        "peak_year_count": 424,
        "link_assignments": 120,
        "link_rows": 112,
        "link_domains": 61,
    }
    observed_legacy = {
        "chat_ids_unique": legacy["chat_ids_unique"],
        "titles_unique": legacy["titles_unique"],
        "peak_year_count": legacy["peak_year"]["count"],
        "link_assignments": legacy["links"]["link_assignments"],
        "link_rows": legacy["links"]["rows_with_external_links"],
        "link_domains": legacy["links"]["unique_domains"],
    }
    if observed_legacy != expected_legacy:
        raise ValueError(f"C-original não reproduziu a linha de base: {observed_legacy}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = f"estatisticas_descritivas_corpora_{args.tag}"
    json_path = args.output_dir / f"{base}.json"
    metrics_path = args.output_dir / f"{base}.csv"
    annual_path = args.output_dir / f"{base}_serie_anual.csv"
    monthly_path = args.output_dir / f"{base}_serie_mensal.csv"
    titles_path = args.output_dir / f"{base}_top_titulos.csv"
    md_path = args.output_dir / f"{base}.md"

    json_path.write_text(json.dumps({"summaries": by_name}, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = [row for summary in summaries for row in metric_rows(summary)]
    write_csv(metrics_path, metrics, ["corpus", "unit", "group", "metric", "value", "denominator", "percent", "note"])
    write_csv(annual_path, series_rows(summaries, "annual", "year"), ["corpus", "unit", "year", "count", "denominator", "percent"])
    write_csv(monthly_path, series_rows(summaries, "monthly", "month"), ["corpus", "unit", "month", "count", "denominator", "percent"])
    write_csv(titles_path, top_title_rows(summaries), ["corpus", "unit", "rank", "chat_title", "count", "denominator", "percent"])

    report = """# Estatísticas descritivas reproduzíveis — corpus Darwin

## Unidades analíticas

| Corpus | Unidade | N | Uso |
|---|---|---:|---|
"""
    for summary in summaries:
        report += f"| {summary['corpus']} | {summary['unit']} | {summary['n']} | {summary['note']} |\n"

    report += """

## Período, chats e tipos

| Corpus | Período observado | IDs de chat | Títulos | Supergrupos | Canais | Pico anual |
|---|---|---:|---:|---:|---:|---|
"""
    for summary in summaries:
        supergroups = summary["types"].get("supergroup", 0)
        channels = summary["types"].get("channel", 0)
        start = summary["period"]["min_iso"][:10]
        end = summary["period"]["max_iso"][:10]
        peak_year = summary["peak_year"]
        report += (
            f"| {summary['corpus']} | {start}–{end} | {summary['chat_ids_unique']} | {summary['titles_unique']} | "
            f"{supergroups} ({pct(supergroups, summary['n'])}) | {channels} ({pct(channels, summary['n'])}) | "
            f"{peak_year['period']}: {peak_year['count']} ({pct(peak_year['count'], summary['n'])}) |\n"
        )

    report += """

## Idioma provável

| Corpus | Português | Inglês | Espanhol | Indeterminado |
|---|---:|---:|---:|---:|
"""
    for summary in summaries:
        languages = summary["languages"]
        report += (
            f"| {summary['corpus']} | {languages.get('pt', 0)} ({pct(languages.get('pt', 0), summary['n'])}) | "
            f"{languages.get('en', 0)} ({pct(languages.get('en', 0), summary['n'])}) | "
            f"{languages.get('es', 0)} ({pct(languages.get('es', 0), summary['n'])}) | "
            f"{languages.get('indeterminado', 0)} ({pct(languages.get('indeterminado', 0), summary['n'])}) |\n"
        )

    report += """

## Visualizações

A cobertura usa todas as unidades do corpus. Média, mediana e máximo usam somente valores válidos.

| Corpus | Views disponíveis | Cobertura | Média válida | Mediana válida | Máximo |
|---|---:|---:|---:|---:|---:|
"""
    for summary in summaries:
        views = summary["views"]
        report += (
            f"| {summary['corpus']} | {views['coverage_n']}/{summary['n']} | {views['coverage_percent']:.1f}% | "
            f"{fmt(views['mean'])} | {fmt(views['median'])} | {fmt(views['max'], 0)} |\n"
        )

    report += """

## Encaminhamentos

| Corpus | Origem de chat confirmada | Qualquer metadado substantivo | Campos legados de chat |
|---|---:|---:|---:|
"""
    for summary in summaries:
        forward = summary["forward"]
        report += (
            f"| {summary['corpus']} | {forward['confirmed_source_chat_id_n']} ({forward['confirmed_source_chat_id_percent']:.1f}%) | "
            f"{forward['any_substantive_metadata_n']} ({forward['any_substantive_metadata_percent']:.1f}%) | "
            f"{forward['legacy_chat_fields_n']} ({forward['legacy_chat_fields_percent']:.1f}%) |\n"
        )

    report += """

## Links externos

| Corpus | Unidades com link | Atribuições | Links distintos | Domínios distintos |
|---|---:|---:|---:|---:|
"""
    for summary in summaries:
        links = summary["links"]
        report += (
            f"| {summary['corpus']} | {links['rows_with_external_links']} ({links['rows_with_external_links_percent']:.1f}%) | "
            f"{links['link_assignments']} | {links['unique_links']} | {links['unique_domains']} |\n"
        )

    report += f"""

## Auditoria dos números atualmente presentes no draft

C-original reproduz os seguintes números do texto atual:

- 1.269 registros;
- 335 identificadores e 381 títulos;
- 878 supergrupos e 391 canais;
- pico de 424 registros em 2023;
- idiomas 871/51/28/319;
- média de views 561,2, mediana 113 e máximo 13.633;
- 120 atribuições de links em 112 mensagens e 61 domínios.

A afirmação de **217 encaminhamentos** não foi reproduzida por critérios explícitos aplicados aos campos exportados. C-original contém 92 registros com `forward_from_chat.id`, 158 com algum metadado substantivo de encaminhamento e 134 com `forward_from_chat` ou `forward_chat_id` substantivo. O número 217 deve ser removido ou ter seu procedimento original recuperado antes da revisão do artigo.

O intervalo da consulta começava em 1º de janeiro de 2016 e terminava em 26 de maio de 2026. O intervalo efetivamente observado nas mensagens vai de 9 de outubro de 2016 a 25 de maio de 2026 no fuso `America/Sao_Paulo`.

## Denominadores recomendados para o artigo

- Frequência temporal, tipos de chat, visualizações, encaminhamentos e circulação: **A-canônico, N={by_name['A-canônico']['n']} mensagens**.
- Idioma, extensão, eixos e análise lexical: **B, N={by_name['B']['n']} textos únicos**.
- Sensibilidade do procedimento histórico: **C, N={by_name['C']['n']} textos**.
- Reprodução dos números atualmente presentes no draft: **C-original, N={by_name['C-original']['n']} registros**.

## Limites

- Títulos não equivalem a chats, pois um mesmo `chat_id` pode mudar de nome.
- Para B e C, data, tipo, título e visualizações pertencem à primeira ocorrência observada do texto.
- `views` mede exposição potencial e tem cobertura desigual por tipo de chat.
- Encaminhamento confirmado exige `forward_from_chat.id`; outras contagens são diagnósticas e não substituem esse critério.
- Links são deduplicados dentro de cada unidade por chave canônica; atribuições podem repetir o mesmo link entre unidades.
- C-original preserva o representante retornado pela agregação histórica e não deve substituir B nas novas análises.

## Produtos

- `{json_path.name}` — resumo estruturado.
- `{metrics_path.name}` — tabela longa de métricas e denominadores.
- `{annual_path.name}` — série anual.
- `{monthly_path.name}` — série mensal.
- `{titles_path.name}` — vinte títulos mais frequentes em cada corpus.
- `{md_path.name}` — este relatório.
"""
    md_path.write_text(report, encoding="utf-8")

    print(" ".join(f"{summary['corpus']}={summary['n']}" for summary in summaries))
    print(f"relatorio={md_path}")


if __name__ == "__main__":
    main()
