#!/usr/bin/env python3
"""Regenera o inventário de links externos dos corpora B e C."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from extrair_links_externos import (
    canonical_key,
    classify_url,
    extract_raw_urls,
    host,
    is_telegram_url,
    structured_youtube_urls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Arquivo inválido: {path}")
    return rows


def external_links(row: dict[str, Any]) -> dict[str, dict[str, str]]:
    candidates = extract_raw_urls(row) + structured_youtube_urls(row)
    unique: dict[str, dict[str, str]] = {}
    for url, source_field in candidates:
        if not url or is_telegram_url(url):
            continue
        key = canonical_key(url)
        if key in unique and unique[key]["source_field"] != "youtube_id":
            continue
        unique[key] = {"url": url, "source_field": source_field}
    return unique


def build_assignments(b_rows: list[dict[str, Any]], c_hashes: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in b_rows:
        for key, item in external_links(row).items():
            url = item["url"]
            output.append(
                {
                    "text_id": row["text_id"],
                    "in_corpus_c": row["text_id"] in c_hashes,
                    "occurrence_count": int(row.get("text_occurrence_count") or 1),
                    "chat_count": int(row.get("text_chat_count") or 1),
                    "representative_chat_id": row.get("chat_id"),
                    "representative_chat_title": row.get("chat_title") or "",
                    "representative_message_id": row.get("message_id"),
                    "link_key": key,
                    "url": url,
                    "domain": host(url) or "sem_dominio",
                    "link_type": classify_url(url),
                    "source_field": item["source_field"],
                    "collection_status": "nao_coletado_nesta_etapa",
                }
            )
    return sorted(output, key=lambda row: (row["text_id"], row["link_key"]))


def summarize(assignments: list[dict[str, Any]], c_hashes: set[str]) -> dict[str, Any]:
    b_texts = {row["text_id"] for row in assignments}
    c_rows = [row for row in assignments if row["in_corpus_c"]]
    c_texts = {row["text_id"] for row in c_rows}
    b_links = {row["link_key"] for row in assignments}
    c_links = {row["link_key"] for row in c_rows}
    b_domains = {row["domain"] for row in assignments}
    c_domains = {row["domain"] for row in c_rows}
    return {
        "B": {
            "texts_with_links": len(b_texts),
            "link_assignments": len(assignments),
            "unique_links": len(b_links),
            "unique_domains": len(b_domains),
        },
        "C": {
            "texts_with_links": len(c_texts),
            "link_assignments": len(c_rows),
            "unique_links": len(c_links),
            "unique_domains": len(c_domains),
        },
        "sensitivity": {
            "unique_links_in_B_not_C": len(b_links - c_links),
            "unique_domains_in_B_not_C": len(b_domains - c_domains),
            "C_link_retention_percent": len(c_links) / len(b_links) * 100 if b_links else None,
            "C_domain_retention_percent": len(c_domains) / len(b_domains) * 100 if b_domains else None,
            "C_hashes_total": len(c_hashes),
        },
    }


def aggregate_rows(assignments: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    values = sorted({row[field] for row in assignments})
    output: list[dict[str, Any]] = []
    for value in values:
        b_rows = [row for row in assignments if row[field] == value]
        c_rows = [row for row in b_rows if row["in_corpus_c"]]
        output.append(
            {
                field: value,
                "b_assignments": len(b_rows),
                "b_texts": len({row["text_id"] for row in b_rows}),
                "b_unique_links": len({row["link_key"] for row in b_rows}),
                "c_assignments": len(c_rows),
                "c_texts": len({row["text_id"] for row in c_rows}),
                "c_unique_links": len({row["link_key"] for row in c_rows}),
            }
        )
    return sorted(output, key=lambda row: (-row["b_assignments"], str(row[field])))


def distinct_link_rows(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        grouped[row["link_key"]].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        first = sorted(rows, key=lambda row: (row["text_id"], row["url"]))[0]
        output.append(
            {
                "link_key": key,
                "representative_url": first["url"],
                "domain": first["domain"],
                "link_type": first["link_type"],
                "b_texts": len({row["text_id"] for row in rows}),
                "b_assignments": len(rows),
                "b_occurrence_weight": sum(int(row["occurrence_count"]) for row in rows),
                "in_corpus_c": any(row["in_corpus_c"] for row in rows),
                "c_texts": len({row["text_id"] for row in rows if row["in_corpus_c"]}),
                "collection_status": "nao_coletado_nesta_etapa",
            }
        )
    return sorted(output, key=lambda row: (-row["b_texts"], row["link_key"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    b_rows = load(args.b)
    c_rows = load(args.c)
    c_hashes = {row["text_id"] for row in c_rows}
    b_hashes = {row["text_id"] for row in b_rows}
    if not c_hashes.issubset(b_hashes):
        raise ValueError("Corpus C não é subconjunto de B")

    assignments = build_assignments(b_rows, c_hashes)
    summary = summarize(assignments, c_hashes)
    domains = aggregate_rows(assignments, "domain")
    types = aggregate_rows(assignments, "link_type")
    distinct_links = distinct_link_rows(assignments)

    expected = {
        "B": {"texts_with_links": 421, "link_assignments": 592, "unique_links": 521, "unique_domains": 250},
        "C": {"texts_with_links": 112, "link_assignments": 120, "unique_links": 112, "unique_domains": 61},
    }
    if summary["B"] != expected["B"] or summary["C"] != expected["C"]:
        raise ValueError(f"Inventário não reproduziu as estatísticas validadas: {summary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"links_corpus_b_{args.tag}"
    json_path = args.output_dir / f"{prefix}.json"
    inventory_path = args.output_dir / f"{prefix}_inventario_restrito.csv"
    distinct_path = args.output_dir / f"{prefix}_links_distintos.csv"
    domains_path = args.output_dir / f"{prefix}_dominios.csv"
    types_path = args.output_dir / f"{prefix}_tipos.csv"
    md_path = args.output_dir / f"{prefix}.md"

    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "method": {
                    "unit": "texto único do Corpus B",
                    "deduplication": "link_key canônica dentro de cada texto",
                    "telegram_links_excluded": True,
                    "external_page_collection": "não executada nesta etapa",
                },
                "top_domains": domains[:30],
                "types": types,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(inventory_path, assignments)
    write_csv(distinct_path, distinct_links)
    write_csv(domains_path, domains)
    write_csv(types_path, types)

    b = summary["B"]
    c = summary["C"]
    sensitivity = summary["sensitivity"]
    report = f"""# Inventário de links externos — Corpus B

## Escopo

- Unidade principal: texto único do Corpus B.
- Links internos do Telegram foram excluídos.
- Links equivalentes foram deduplicados dentro de cada texto por chave canônica.
- Páginas externas não foram coletadas nesta etapa.
- Social-RAG não foi executado.

## Resumo

| Corpus | Textos com links | Atribuições | Links distintos | Domínios distintos |
|---|---:|---:|---:|---:|
| B | {b['texts_with_links']} | {b['link_assignments']} | {b['unique_links']} | {b['unique_domains']} |
| C | {c['texts_with_links']} | {c['link_assignments']} | {c['unique_links']} | {c['unique_domains']} |

## Sensibilidade

- Links distintos presentes em B e ausentes de C: {sensitivity['unique_links_in_B_not_C']}.
- Domínios presentes em B e ausentes de C: {sensitivity['unique_domains_in_B_not_C']}.
- Retenção de links em C: {sensitivity['C_link_retention_percent']:.1f}%.
- Retenção de domínios em C: {sensitivity['C_domain_retention_percent']:.1f}%.

## Tipos de link

| Tipo | Atribuições B | Textos B | Links B | Atribuições C | Textos C | Links C |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in types:
        report += (
            f"| {row['link_type']} | {row['b_assignments']} | {row['b_texts']} | {row['b_unique_links']} | "
            f"{row['c_assignments']} | {row['c_texts']} | {row['c_unique_links']} |\n"
        )

    report += f"""

## Limites e próximos passos

- O inventário registra URLs compartilhadas, não o conteúdo atual das páginas.
- Um link pode ter mudado, redirecionado ou desaparecido desde a publicação.
- A fila de 521 links distintos exige estratégia própria de coleta, controle de bloqueios e validação humana.
- A coleta de páginas deve ser decidida antes da revisão da seção de ecologias de autoridade.

## Produtos

- `{json_path.name}` — resumo e parâmetros.
- `{inventory_path.name}` — atribuições texto-link com metadados; arquivo restrito.
- `{distinct_path.name}` — links canônicos distintos e fila para eventual coleta externa.
- `{domains_path.name}` — agregação por domínio.
- `{types_path.name}` — agregação por tipo.
- `{md_path.name}` — este relatório.
"""
    md_path.write_text(report, encoding="utf-8")
    print(f"B_links={b['unique_links']} C_links={c['unique_links']} distinct={len(distinct_links)}")
    print(f"relatorio={md_path}")


if __name__ == "__main__":
    main()
