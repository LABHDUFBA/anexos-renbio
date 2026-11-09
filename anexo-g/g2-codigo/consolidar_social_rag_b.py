#!/usr/bin/env python3
"""Consolida respostas Social-RAG do Corpus B com rastreabilidade B–C."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--historical-matrix", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Esperada lista em {path}")
    return data


def norm_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def source_key(chat_id: Any, message_id: Any) -> tuple[str, str]:
    return norm_id(chat_id), norm_id(message_id)


def norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def validate_excerpt(excerpt: Any, source_text: Any) -> str:
    raw_excerpt = str(excerpt or "").strip()
    raw_source = str(source_text or "")
    if raw_excerpt and raw_excerpt in raw_source:
        return "exact"
    excerpt_norm = norm_text(raw_excerpt)
    source_norm = norm_text(raw_source)
    if excerpt_norm and excerpt_norm in source_norm:
        return "normalized"
    fragments = [
        part.strip(" .…")
        for part in re.split(r"\.{3}|…|\s+/\s+", excerpt_norm)
        if len(part.strip(" .…")) >= 12
    ]
    if fragments and all(fragment in source_norm for fragment in fragments):
        return "ellipsis_fragments"
    if excerpt_norm and SequenceMatcher(None, excerpt_norm, source_norm).ratio() >= 0.78:
        return "fuzzy_high"
    return "manual_required"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def retrieval_summary(response: dict[str, Any]) -> dict[str, Any]:
    metadata = response.get("metadata") or {}
    stats = metadata.get("adaptive_k_stats") or {}
    return {
        "k_candidates": metadata.get("k_candidates"),
        "k_adaptive": metadata.get("k_adaptive"),
        "k_final": metadata.get("k_final"),
        "mode": stats.get("mode"),
        "density": stats.get("density"),
        "significant_gaps": stats.get("num_significant_gaps"),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.run_dir / "manifesto.json").read_text(encoding="utf-8"))
    questions = manifest["questions"]
    a_rows = load(args.a)
    b_rows = load(args.b)
    c_rows = load(args.c)
    b_by_key = {source_key(row["chat_id"], row["message_id"]): row for row in b_rows}
    b_by_hash = {str(row["text_sha256"]): row for row in b_rows}
    c_ids = {row["text_id"] for row in c_rows}

    reports_dir = args.run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    context_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    question_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for question in questions:
        response_path = args.run_dir / "json" / f"{question['label']}.json"
        response = json.loads(response_path.read_text(encoding="utf-8"))
        question_payloads.append((question, response))
        question_sources: set[str] = set()
        c_sources: set[str] = set()
        outside_c_sources: set[str] = set()
        long_sources: set[str] = set()
        matched = 0
        for context in response.get("context_docs") or []:
            metadata = context.get("metadata") or {}
            key = source_key(metadata.get("chat_id"), metadata.get("message_id"))
            source = b_by_key.get(key)
            match = source is not None
            if source is not None:
                matched += 1
                text_id = source["text_id"]
                in_c = text_id in c_ids
                text_chars = len(str(source.get("all_text") or ""))
                occurrence_count = int(source.get("text_occurrence_count") or 1)
                question_sources.add(text_id)
                (c_sources if in_c else outside_c_sources).add(text_id)
                if text_chars > 256:
                    long_sources.add(text_id)
            else:
                text_id = ""
                in_c = False
                text_chars = ""
                occurrence_count = ""
            context_rows.append(
                {
                    "axis": question["axis"],
                    "question_label": question["label"],
                    "question_title": question["title"],
                    "context_index": context.get("index"),
                    "content": context.get("content") or "",
                    "chat_id": metadata.get("chat_id") or "",
                    "chat_title": metadata.get("chat_title") or "",
                    "type": metadata.get("type") or "",
                    "strict_date": metadata.get("strict_date") or "",
                    "message_id": metadata.get("message_id") or "",
                    "source_match_b": match,
                    "text_id": text_id,
                    "in_corpus_c": in_c if match else "",
                    "text_chars": text_chars,
                    "occurrence_count": occurrence_count,
                    "validation_status": "fonte_validada_interpretacao_pendente" if match else "fonte_nao_localizada",
                }
            )
        comparison_rows.append(
            {
                "axis": question["axis"],
                "question_label": question["label"],
                "question_title": question["title"],
                "chunks": len(response.get("context_docs") or []),
                "matched_chunks": matched,
                "unique_sources": len(question_sources),
                "unique_sources_c": len(c_sources),
                "unique_sources_outside_c": len(outside_c_sources),
                "unique_sources_over_256_chars": len(long_sources),
                **retrieval_summary(response),
            }
        )

    write_csv(args.run_dir / "contextos_recuperados.csv", context_rows)
    write_csv(args.run_dir / "comparacao_b_c_por_pergunta.csv", comparison_rows)

    by_text: dict[str, dict[str, Any]] = {}
    for row in context_rows:
        if not row["text_id"]:
            continue
        text_id = str(row["text_id"])
        source = b_by_key[source_key(row["chat_id"], row["message_id"])]
        entry = by_text.setdefault(
            text_id,
            {
                "text_id": text_id,
                "in_corpus_c": row["in_corpus_c"],
                "text_chars": row["text_chars"],
                "occurrence_count": row["occurrence_count"],
                "chat_id": row["chat_id"],
                "chat_title": row["chat_title"],
                "type": row["type"],
                "strict_date": row["strict_date"],
                "message_id": row["message_id"],
                "all_text": source.get("all_text") or "",
                "retrieval_count": 0,
                "axes": set(),
                "questions": set(),
            },
        )
        entry["retrieval_count"] += 1
        entry["axes"].add(row["axis"])
        entry["questions"].add(row["question_label"])

    unique_rows: list[dict[str, Any]] = []
    for entry in by_text.values():
        unique_rows.append(
            {
                **{k: v for k, v in entry.items() if k not in {"axes", "questions"}},
                "axes": " | ".join(sorted(entry["axes"])),
                "questions": " | ".join(sorted(entry["questions"])),
                "validation_status": "fonte_validada_interpretacao_pendente",
            }
        )
    unique_rows.sort(key=lambda row: (-int(row["retrieval_count"]), -int(row["text_chars"]), row["text_id"]))
    write_csv(args.run_dir / "fontes_unicas_recuperadas.csv", unique_rows)

    candidates: list[dict[str, Any]] = []
    axes = sorted({question["axis"] for question in questions})
    for axis in axes:
        axis_sources = [
            row for row in unique_rows if axis in str(row["axes"]).split(" | ")
        ]
        outside = [row for row in axis_sources if str(row["in_corpus_c"]) == "False"]
        inside = [row for row in axis_sources if str(row["in_corpus_c"]) == "True"]
        key_fn = lambda row: (-int(row["retrieval_count"]), -int(row["text_chars"]), row["text_id"])
        selected = [("B_fora_C", row) for row in sorted(outside, key=key_fn)[:15]]
        selected += [("C", row) for row in sorted(inside, key=key_fn)[:5]]
        for rank, (scope, row) in enumerate(selected, start=1):
            candidates.append(
                {
                    "axis": axis,
                    "scope": scope,
                    "rank": rank,
                    **row,
                    "codigo": "",
                    "subcodigo": "",
                    "posicao_enunciativa": "pendente",
                    "uso_de_darwin": "pendente",
                    "confianca_analitica": "pendente",
                    "observacao": "Candidato por recuperação semântica; requer leitura humana antes de uso no artigo.",
                }
            )
    write_csv(args.run_dir / "matriz_candidatos_restrita.csv", candidates)

    historical_rows = list(csv.DictReader(args.historical_matrix.open(encoding="utf-8")))
    historical_audit: list[dict[str, Any]] = []
    for row in historical_rows:
        occurrence_matches = [
            item for item in a_rows
            if norm_id(item.get("message_id")) == norm_id(row.get("message_id"))
            and norm_text(item.get("chat_title")) == norm_text(row.get("chat_title"))
        ]
        occurrence = occurrence_matches[0] if len(occurrence_matches) == 1 else None
        representative = b_by_hash.get(str(occurrence.get("text_sha256"))) if occurrence else None
        # Fallback para matrizes antigas cujo metadado aponta a outra ocorrência do mesmo texto.
        if representative is None:
            same_id = [
                item for item in b_rows
                if norm_id(item.get("message_id")) == norm_id(row.get("message_id"))
                and norm_text(item.get("chat_title")) == norm_text(row.get("chat_title"))
            ]
            representative = same_id[0] if len(same_id) == 1 else None
        validation = validate_excerpt(
            row.get("excerto"),
            occurrence.get("all_text") if occurrence else representative.get("all_text") if representative else "",
        )
        historical_audit.append(
            {
                "eixo": row.get("eixo") or "",
                "codigo": row.get("codigo") or "",
                "historical_chat_title": row.get("chat_title") or "",
                "historical_message_id": row.get("message_id") or "",
                "occurrence_present_in_a": occurrence is not None,
                "text_present_in_b": representative is not None,
                "representative_chat_title": representative.get("chat_title") if representative else "",
                "representative_message_id": representative.get("message_id") if representative else "",
                "text_id": representative.get("text_id") if representative else "",
                "in_corpus_c": representative.get("text_id") in c_ids if representative else "",
                "excerpt_validation": validation,
                "text_matches_excerpt": validation != "manual_required",
            }
        )
    write_csv(args.run_dir / "auditoria_matriz_historica.csv", historical_audit)

    by_axis: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for question, response in question_payloads:
        by_axis[question["axis"]].append((question, response))
    for index, (axis, items) in enumerate(by_axis.items(), start=1):
        lines = [f"# {axis} — Social-RAG Corpus B", "", f"Tema: `{manifest['theme']}`  ", f"Modelo: `{manifest['model']}`", ""]
        for question, response in items:
            lines += [f"## {question['title']}", "", f"**Pergunta:** {question['question']}", "", "### Resposta", "", response.get("answer") or "", ""]
            retrieval = retrieval_summary(response)
            lines += [
                "### Recuperação",
                "",
                f"- modo: {retrieval['mode']}",
                f"- densidade: {retrieval['density']}",
                f"- K candidatos/adaptativo/final: {retrieval['k_candidates']} / {retrieval['k_adaptive']} / {retrieval['k_final']}",
                f"- chunks retornados: {len(response.get('context_docs') or [])}",
                "",
                "### Contextos recuperados",
                "",
            ]
            for context in response.get("context_docs") or []:
                metadata = context.get("metadata") or {}
                source = b_by_key.get(source_key(metadata.get("chat_id"), metadata.get("message_id")))
                in_c = source.get("text_id") in c_ids if source else None
                lines += [
                    f"#### Chunk {context.get('index')}",
                    "",
                    f"Fonte: `{metadata.get('chat_title')}` | `{metadata.get('type')}` | strict_date=`{metadata.get('strict_date')}` | message_id=`{metadata.get('message_id')}` | in_C=`{in_c}`",
                    "",
                    str(context.get("content") or ""),
                    "",
                ]
        (reports_dir / f"eixo{index}_consolidado.md").write_text("\n".join(lines), encoding="utf-8")

    total_chunks = len(context_rows)
    matched_chunks = sum(str(row["source_match_b"]) == "True" for row in context_rows)
    unique_c = sum(str(row["in_corpus_c"]) == "True" for row in unique_rows)
    unique_outside = sum(str(row["in_corpus_c"]) == "False" for row in unique_rows)
    by_axis_stats: dict[str, dict[str, int]] = {}
    for axis, items in by_axis.items():
        axis_rows = [row for row in context_rows if row["axis"] == axis and row["text_id"]]
        axis_ids = {str(row["text_id"]) for row in axis_rows}
        axis_c = {str(row["text_id"]) for row in axis_rows if str(row["in_corpus_c"]) == "True"}
        axis_outside = {str(row["text_id"]) for row in axis_rows if str(row["in_corpus_c"]) == "False"}
        by_axis_stats[axis] = {
            "questions": len(items),
            "chunks": len(axis_rows),
            "unique_sources": len(axis_ids),
            "unique_sources_c": len(axis_c),
            "unique_sources_outside_c": len(axis_outside),
        }
    summary = {
        "theme": manifest["theme"],
        "model": manifest["model"],
        "questions": len(questions),
        "chunks": total_chunks,
        "matched_chunks": matched_chunks,
        "unique_sources": len(unique_rows),
        "unique_sources_c": unique_c,
        "unique_sources_outside_c": unique_outside,
        "candidates": len(candidates),
        "historical_matrix_rows": len(historical_rows),
        "historical_matrix_occurrences_in_a": sum(str(row["occurrence_present_in_a"]) == "True" for row in historical_audit),
        "historical_matrix_present_in_b": sum(str(row["text_present_in_b"]) == "True" for row in historical_audit),
        "historical_matrix_excerpt_matches": sum(str(row["text_matches_excerpt"]) == "True" for row in historical_audit),
        "retrieval_modes": dict(Counter(str(row.get("mode")) for row in comparison_rows)),
        "by_axis": by_axis_stats,
    }
    (args.run_dir / "resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Social-RAG — Corpus B

## Execução

- tema: `{summary['theme']}`
- modelo: `{summary['model']}`
- perguntas: {summary['questions']}
- chunks recuperados: {summary['chunks']}
- chunks localizados no Corpus B: {summary['matched_chunks']}
- fontes únicas: {summary['unique_sources']}
- fontes pertencentes a C: {summary['unique_sources_c']}
- fontes de B ausentes de C: {summary['unique_sources_outside_c']}
- candidatos para leitura humana: {summary['candidates']}

## Comparação B–C por eixo

| Eixo | Perguntas | Chunks | Fontes únicas | Em C | Fora de C |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f"| {axis} | {stats['questions']} | {stats['chunks']} | {stats['unique_sources']} | {stats['unique_sources_c']} | {stats['unique_sources_outside_c']} |" for axis, stats in summary['by_axis'].items())}

## Auditoria da matriz histórica

- linhas: {summary['historical_matrix_rows']}
- ocorrências históricas presentes em A: {summary['historical_matrix_occurrences_in_a']}
- textos correspondentes presentes em B: {summary['historical_matrix_present_in_b']}
- excertos validados no texto-fonte: {summary['historical_matrix_excerpt_matches']}

## Limites

- Recuperação semântica não equivale a classificação nem contagem do corpus.
- A presença entre os contextos não demonstra adesão ideológica.
- A matriz de candidatos preserva interpretação como pendente.
- O campo `in_corpus_c` permite distinguir evidências já disponíveis no recorte histórico das recuperadas somente na base completa.
"""
    (args.run_dir / "relatorio.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
