#!/usr/bin/env python3
"""Executa lote reproduzível de perguntas Social-RAG no Corpus B."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

METADATA_CLAUSE = (
    " Ao responder, cite explicitamente os metadados de cada mensagem recuperada: "
    "chat_title, tipo, data e message_id. Distinga afirmação, pergunta, crítica, ironia, "
    "citação de terceiro, link/título e resposta automatizada. Não infira adesão sem apoio textual."
)

EXTRA_QUESTIONS = [
    {
        "axis": "6 — Sensibilidade B–C",
        "label": "6a_textos_longos",
        "title": "Argumentos desenvolvidos em textos longos",
        "question": (
            "Quais mensagens desenvolvem argumentos extensos ou narrativas completas sobre Darwin, evolução, "
            "darwinismo social, raça, religião ou política, em vez de apenas rótulos e frases curtas? "
            "Identifique encadeamentos argumentativos, fontes mobilizadas e tensões internas."
        ),
    },
    {
        "axis": "6 — Sensibilidade B–C",
        "label": "6b_contestacoes_internas",
        "title": "Contestações e correções internas",
        "question": (
            "Quais mensagens contestam, corrigem ou problematizam associações entre evolução, darwinismo social, "
            "racismo, eugenia, nazismo, religião ou comunismo? Identifique casos que recusam simplificações e "
            "separe correção conceitual, pergunta, ironia e disputa entre participantes."
        ),
    },
    {
        "axis": "6 — Sensibilidade B–C",
        "label": "6c_marx_darwin_ampliado",
        "title": "Marx-Darwin em textos desenvolvidos",
        "question": (
            "Como textos mais desenvolvidos articulam Darwin, Marx, materialismo, ateísmo, comunismo e natureza "
            "humana? Procure argumentos completos, valências positivas e negativas e contestações da associação."
        ),
    },
    {
        "axis": "6 — Sensibilidade B–C",
        "label": "6d_antievolucionismo_ampliado",
        "title": "Antievolucionismo em textos desenvolvidos",
        "question": (
            "Quais mensagens apresentam argumentos desenvolvidos contra a evolução, para além de afirmar que ela é "
            "mentira ou fraude? Identifique apelos a religião, design inteligente, autoridades científicas, escola, "
            "família, conspiração e outros repertórios pseudocientíficos."
        ),
    },
    {
        "axis": "6 — Sensibilidade B–C",
        "label": "6e_politica_raca_ampliado",
        "title": "Raça e extrema direita em textos desenvolvidos",
        "question": (
            "Quais mensagens desenvolvem relações entre Darwin, raça, eugenia, nazismo, antissemitismo, hierarquia "
            "social ou extrema direita? Diferencie uso afirmativo, acusação contra terceiros, crítica, pergunta e "
            "contestação da relação."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8900")
    parser.add_argument("--theme", default="darwin_b_20260717")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def slug(text: str) -> str:
    normalized = text.lower()
    replacements = str.maketrans("áàâãéêíóôõúç–—", "aaaaeeioooucn-")
    normalized = normalized.translate(replacements)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def parse_questions(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    axis = ""
    pending_heading = ""
    output: list[dict[str, str]] = []
    for line in lines:
        if line.startswith("## Eixo "):
            axis = line[3:].strip()
        elif line.startswith("## Notas"):
            axis = ""
        elif axis and line.startswith("### "):
            pending_heading = line[4:].strip()
        elif axis and pending_heading and line.startswith("> "):
            match = re.match(r"(?P<label>\d+[a-z])\.\s*(?P<title>.+)", pending_heading)
            if not match:
                raise ValueError(f"Cabeçalho de pergunta inválido: {pending_heading}")
            output.append(
                {
                    "axis": axis,
                    "label": f"eixo{match.group('label')[0]}_{match.group('label')}_{slug(match.group('title'))}",
                    "short_label": match.group("label"),
                    "title": match.group("title"),
                    "question": line[2:].strip(),
                }
            )
            pending_heading = ""
    if len(output) != 21:
        raise ValueError(f"Esperadas 21 perguntas históricas; encontradas {len(output)}")
    for row in EXTRA_QUESTIONS:
        output.append({**row, "short_label": row["label"].split("_")[0]})
    return output


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def retrieval_fields(response: dict[str, Any]) -> dict[str, Any]:
    metadata = response.get("metadata") or {}
    stats = metadata.get("adaptive_k_stats") or {}
    return {
        "k_candidates": metadata.get("k_candidates"),
        "k_adaptive": metadata.get("k_adaptive"),
        "k_final": metadata.get("k_final"),
        "mode": stats.get("mode"),
        "density": stats.get("density"),
        "significant_gaps": stats.get("num_significant_gaps"),
        "context_docs": len(response.get("context_docs") or []),
    }


def write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "axis", "label", "title", "status", "model", "theme", "elapsed_seconds",
        "k_candidates", "k_adaptive", "k_final", "mode", "density",
        "significant_gaps", "context_docs", "output_file", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    questions = parse_questions(args.questions)
    json_dir = args.output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "execucao.csv"
    log_rows: list[dict[str, Any]] = []

    for position, question in enumerate(questions, start=1):
        output_path = json_dir / f"{question['label']}.json"
        if args.resume and output_path.exists():
            response = json.loads(output_path.read_text(encoding="utf-8"))
            log_rows.append(
                {
                    "axis": question["axis"], "label": question["label"], "title": question["title"],
                    "status": "reused", "model": args.model, "theme": args.theme, "elapsed_seconds": 0,
                    **retrieval_fields(response), "output_file": str(output_path), "error": "",
                }
            )
            continue

        payload = {
            "question": question["question"] + METADATA_CLAUSE,
            "theme": args.theme,
            "llm_model": args.model,
            "include_context": True,
            "max_context_docs": 200,
            "include_prompt": False,
            "include_scores": False,
        }
        start = time.monotonic()
        error = ""
        response: dict[str, Any] | None = None
        for attempt in range(1, 4):
            try:
                response = post_json(f"{args.api_url}/ask_rag_detailed", payload, args.timeout)
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                error = f"tentativa {attempt}: {exc}"
                if attempt < 3:
                    time.sleep(2 ** attempt)
        elapsed = round(time.monotonic() - start, 3)
        if response is None:
            log_rows.append(
                {
                    "axis": question["axis"], "label": question["label"], "title": question["title"],
                    "status": "error", "model": args.model, "theme": args.theme,
                    "elapsed_seconds": elapsed, "k_candidates": "", "k_adaptive": "",
                    "k_final": "", "mode": "", "density": "", "significant_gaps": "",
                    "context_docs": "", "output_file": str(output_path), "error": error,
                }
            )
            write_log(log_path, log_rows)
            raise RuntimeError(f"Falha em {question['label']}: {error}")

        assert response is not None
        output_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        log_rows.append(
            {
                "axis": question["axis"], "label": question["label"], "title": question["title"],
                "status": "ok", "model": args.model, "theme": args.theme,
                "elapsed_seconds": elapsed, **retrieval_fields(response),
                "output_file": str(output_path), "error": "",
            }
        )
        write_log(log_path, log_rows)
        print(f"[{position}/{len(questions)}] {question['label']} — {len(response.get('context_docs') or [])} chunks — {elapsed}s", flush=True)

    manifest = {
        "theme": args.theme,
        "model": args.model,
        "api_url": args.api_url,
        "historical_questions": 21,
        "additional_questions": len(EXTRA_QUESTIONS),
        "total_questions": len(questions),
        "metadata_clause": METADATA_CLAUSE.strip(),
        "questions": questions,
    }
    (args.output_dir / "manifesto.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Concluído: {len(questions)} perguntas em {args.output_dir}")


if __name__ == "__main__":
    main()
