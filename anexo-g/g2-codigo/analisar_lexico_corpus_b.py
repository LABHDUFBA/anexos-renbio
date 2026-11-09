#!/usr/bin/env python3
"""Regenera TF-IDF, associações locais G² e KWIC nos corpora B e C."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

from tfidf_bigrams import STOPWORDS

TZ = ZoneInfo("America/Sao_Paulo")
ANALYSIS_STOPWORDS = set(STOPWORDS) | {
    "foi", "vai", "são", "nem", "bem", "há", "te", "la", "el", "los", "las",
    "en", "del", "al", "es", "y", "un", "una", "su", "sus", "más", "of", "in", "to",
}
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
DARWIN_RE = re.compile(r"^darwin\w*$", re.IGNORECASE)
TFIDF_TOP = 100
G2_TOP = 100
KWIC_WIDTH = 5
LOCAL_WINDOW = 5
MIN_CONTEXT_FREQ = 3


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


def contains_darwin_term(term: str) -> bool:
    return any(DARWIN_RE.fullmatch(token) for token in term.split())


def run_tfidf(name: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    texts = [str(row.get("all_text") or "") for row in rows]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=sorted(ANALYSIS_STOPWORDS),
        token_pattern=r"(?u)\b\w\w+\b",
        max_df=0.85,
        min_df=3,
        norm="l2",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    averages = np.asarray(matrix.mean(axis=0)).ravel()
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    results: list[dict[str, Any]] = []
    for index, term in enumerate(terms):
        if contains_darwin_term(str(term)):
            continue
        results.append(
            {
                "corpus": name,
                "ngram_type": "bigram" if " " in str(term) else "unigram",
                "term": str(term),
                "tfidf_mean": float(averages[index]),
                "document_frequency": int(document_frequency[index]),
                "document_percent": int(document_frequency[index]) / len(rows) * 100,
            }
        )

    top_rows: list[dict[str, Any]] = []
    top_json: dict[str, list[dict[str, Any]]] = {}
    for kind in ("unigram", "bigram"):
        selected = [row for row in results if row["ngram_type"] == kind]
        selected.sort(key=lambda row: (-row["tfidf_mean"], -row["document_frequency"], row["term"]))
        top = selected[:TFIDF_TOP]
        for rank, row in enumerate(top, start=1):
            top_rows.append({"rank": rank, **row})
        top_json[kind] = top

    summary = {
        "corpus": name,
        "documents": len(rows),
        "vocabulary_size_after_filters": len(results),
        "parameters": {
            "ngram_range": [1, 2],
            "token_pattern": r"(?u)\b\w\w+\b",
            "max_df": 0.85,
            "min_df": 3,
            "norm": "l2",
            "sublinear_tf": True,
            "darwin_derivatives_removed": True,
            "stopwords": "lista PT/EN do projeto acrescida de tokens funcionais PT/EN/ES identificados na auditoria",
            "top_n": TFIDF_TOP,
        },
        "top": top_json,
    }
    return summary, top_rows


def g2_value(a: int, b: int, c: int, d: int) -> float:
    observations = (a, b, c, d)
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    total = row1 + row2
    expectations = (
        row1 * col1 / total,
        row1 * col2 / total,
        row2 * col1 / total,
        row2 * col2 / total,
    )
    return 2 * sum(obs * math.log(obs / exp) for obs, exp in zip(observations, expectations) if obs > 0 and exp > 0)


def run_local_g2(name: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    context_total = 0
    background_total = 0
    darwin_occurrences = 0

    for row in rows:
        row_tokens = [token.lower() for token in TOKEN_RE.findall(str(row.get("all_text") or ""))]
        darwin_positions = [index for index, token in enumerate(row_tokens) if DARWIN_RE.fullmatch(token)]
        darwin_occurrences += len(darwin_positions)
        context_positions: set[int] = set()
        for position in darwin_positions:
            start = max(0, position - LOCAL_WINDOW)
            end = min(len(row_tokens), position + LOCAL_WINDOW + 1)
            context_positions.update(index for index in range(start, end) if index != position)

        for index, token in enumerate(row_tokens):
            if DARWIN_RE.fullmatch(token):
                continue
            if index in context_positions:
                context_counts[token] += 1
                context_total += 1
            else:
                background_counts[token] += 1
                background_total += 1

    candidates = set(context_counts) | set(background_counts)
    results: list[dict[str, Any]] = []
    for word in candidates:
        if word in ANALYSIS_STOPWORDS or DARWIN_RE.fullmatch(word):
            continue
        a = context_counts[word]
        if a < MIN_CONTEXT_FREQ:
            continue
        c = background_counts[word]
        b = context_total - a
        d = background_total - c
        if min(a, b, c, d) < 0:
            continue
        context_rate = a / context_total if context_total else 0
        background_rate = c / background_total if background_total else 0
        if context_rate <= background_rate:
            continue
        g2 = g2_value(a, b, c, d)
        results.append(
            {
                "corpus": name,
                "word": word,
                "context_frequency": a,
                "background_frequency": c,
                "context_total_tokens": context_total,
                "background_total_tokens": background_total,
                "context_rate_per_1000": context_rate * 1000,
                "background_rate_per_1000": background_rate * 1000,
                "g2": g2,
                "p_value": float(chi2.sf(g2, df=1)),
                "window": LOCAL_WINDOW,
            }
        )
    results.sort(key=lambda row: (-row["g2"], -row["context_frequency"], row["word"]))
    top = [{"rank": rank, **row} for rank, row in enumerate(results[:G2_TOP], start=1)]
    summary = {
        "corpus": name,
        "documents": len(rows),
        "darwin_occurrences": darwin_occurrences,
        "context_total_tokens": context_total,
        "background_total_tokens": background_total,
        "parameters": {
            "metric": "G² log-likelihood",
            "window": LOCAL_WINDOW,
            "window_unit": "tokens originais",
            "context_positions": "união das posições a ±5 tokens de darwin* em cada texto",
            "minimum_context_frequency": MIN_CONTEXT_FREQ,
            "direction": "somente termos proporcionalmente mais frequentes no contexto local",
        },
        "top": top,
    }
    return summary, top


def run_kwic(rows: list[dict[str, Any]], c_hashes: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("all_text") or "")
        row_tokens = TOKEN_RE.findall(text)
        for index, token in enumerate(row_tokens):
            if not DARWIN_RE.fullmatch(token.lower()):
                continue
            timestamp = int(row.get("text_first_date") or row["strict_date"])
            output.append(
                {
                    "kwic_id": len(output) + 1,
                    "text_id": row["text_id"],
                    "in_corpus_c": row["text_id"] in c_hashes,
                    "occurrence_count": int(row.get("text_occurrence_count") or 1),
                    "representative_chat_id": row.get("chat_id"),
                    "representative_chat_title": row.get("chat_title") or "",
                    "representative_message_id": row.get("message_id"),
                    "first_date": timestamp,
                    "first_date_iso": datetime.fromtimestamp(timestamp, tz=TZ).isoformat(),
                    "left_context": " ".join(row_tokens[max(0, index - KWIC_WIDTH):index]),
                    "keyword": token,
                    "right_context": " ".join(row_tokens[index + 1:index + KWIC_WIDTH + 1]),
                    "window": KWIC_WIDTH,
                    "validation_status": "pendente_leitura_humana",
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_kwic_text(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"KWIC darwin* — Corpus B ({len(rows)} ocorrências)",
        f"Janela: ±{KWIC_WIDTH} tokens | candidatos pendentes de leitura humana",
        "=" * 120,
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['kwic_id']:>5} | {row['left_context']:>45}  **{row['keyword']}**  {row['right_context']:<45} "
            f"| text_id={row['text_id']} | C={row['in_corpus_c']} | {row['first_date_iso'][:10]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    b_rows = load(args.b)
    c_rows = load(args.c)
    c_hashes = {row["text_id"] for row in c_rows}
    b_hashes = {row["text_id"] for row in b_rows}
    if not c_hashes.issubset(b_hashes):
        raise ValueError("Corpus C não é subconjunto de B")

    tfidf_b, tfidf_b_rows = run_tfidf("B", b_rows)
    tfidf_c, tfidf_c_rows = run_tfidf("C", c_rows)
    g2_b, g2_b_rows = run_local_g2("B", b_rows)
    g2_c, g2_c_rows = run_local_g2("C", c_rows)
    kwic_rows = run_kwic(b_rows, c_hashes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"lexico_corpus_b_{args.tag}"
    json_path = args.output_dir / f"{prefix}.json"
    tfidf_path = args.output_dir / f"{prefix}_tfidf.csv"
    g2_path = args.output_dir / f"{prefix}_associacoes_g2.csv"
    kwic_path = args.output_dir / f"{prefix}_kwic_restrito.csv"
    kwic_text_path = args.output_dir / f"{prefix}_kwic_restrito.txt"
    md_path = args.output_dir / f"{prefix}.md"

    json_path.write_text(
        json.dumps(
            {
                "tfidf": {"B": tfidf_b, "C": tfidf_c},
                "local_associations": {"B": g2_b, "C": g2_c},
                "kwic": {
                    "corpus": "B",
                    "rows": len(kwic_rows),
                    "texts_with_kwic": len({row["text_id"] for row in kwic_rows}),
                    "rows_in_corpus_c": sum(row["in_corpus_c"] for row in kwic_rows),
                    "parameters": {"keyword_regex": r"^darwin\w*$", "window": KWIC_WIDTH},
                    "validation_status": "pendente_leitura_humana",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(tfidf_path, tfidf_b_rows + tfidf_c_rows)
    write_csv(g2_path, g2_b_rows + g2_c_rows)
    write_csv(kwic_path, kwic_rows)
    write_kwic_text(kwic_text_path, kwic_rows)

    report = f"""# Análise lexical do Corpus B

## Escopo

- Corpus principal: B, com {len(b_rows)} textos únicos completos.
- Sensibilidade: C, com {len(c_rows)} textos históricos.
- Social-RAG não foi executado nesta etapa.

## TF-IDF

Parâmetros: unigrams e bigrams; `min_df=3`; `max_df=0.85`; frequência sublinear; normalização L2; stopwords PT/EN; derivados de Darwin removidos do ranking.

| Corpus | Documentos | Vocabulário após filtros |
|---|---:|---:|
| B | {tfidf_b['documents']} | {tfidf_b['vocabulary_size_after_filters']} |
| C | {tfidf_c['documents']} | {tfidf_c['vocabulary_size_after_filters']} |

### Top 20 termos de B

| Rank | Tipo | Termo | TF-IDF médio | Frequência documental |
|---:|---|---|---:|---:|
"""
    for row in (tfidf_b_rows[:20] + tfidf_b_rows[TFIDF_TOP:TFIDF_TOP + 20]):
        report += f"| {row['rank']} | {row['ngram_type']} | {row['term']} | {row['tfidf_mean']:.6f} | {row['document_frequency']} |\n"

    report += f"""

## Associações locais por G²

A unidade é a posição de token. O contexto local reúne posições a até ±{LOCAL_WINDOW} tokens de uma ocorrência de `darwin*`; o restante do corpus constitui o fundo. São mantidos apenas termos proporcionalmente mais frequentes no contexto local e com pelo menos {MIN_CONTEXT_FREQ} ocorrências contextuais.

| Corpus | Ocorrências de darwin* | Tokens contextuais | Tokens de fundo |
|---|---:|---:|---:|
| B | {g2_b['darwin_occurrences']} | {g2_b['context_total_tokens']} | {g2_b['background_total_tokens']} |
| C | {g2_c['darwin_occurrences']} | {g2_c['context_total_tokens']} | {g2_c['background_total_tokens']} |

### Top 20 associações de B

| Rank | Palavra | G² | Freq. contexto | Freq. fundo |
|---:|---|---:|---:|---:|
"""
    for row in g2_b_rows[:20]:
        report += f"| {row['rank']} | {row['word']} | {row['g2']:.3f} | {row['context_frequency']} | {row['background_frequency']} |\n"

    report += f"""

## KWIC

- Concordâncias: {len(kwic_rows)}.
- Textos com ao menos uma concordância: {len({row['text_id'] for row in kwic_rows})}.
- Concordâncias pertencentes a C: {sum(row['in_corpus_c'] for row in kwic_rows)}.
- Janela: ±{KWIC_WIDTH} tokens.
- Status: pendente de leitura humana.

## Limites

- TF-IDF indica saliência lexical, não sentido, posição política ou adesão.
- G² registra associação local de tokens; não substitui leitura contextual.
- KWIC preserva rastreabilidade, mas o representante de B corresponde à primeira ocorrência observada do texto.
- Resultados qualitativos só podem ser usados após leitura das concordâncias e retorno à mensagem.

## Produtos

- `{json_path.name}` — parâmetros e resultados estruturados.
- `{tfidf_path.name}` — rankings TF-IDF de B e C.
- `{g2_path.name}` — associações locais por G².
- `{kwic_path.name}` — concordâncias rastreáveis, arquivo restrito.
- `{kwic_text_path.name}` — versão legível das concordâncias, arquivo restrito.
- `{md_path.name}` — este relatório.
"""
    md_path.write_text(report, encoding="utf-8")
    print(f"B={len(b_rows)} C={len(c_rows)} KWIC={len(kwic_rows)}")
    print(f"relatorio={md_path}")


if __name__ == "__main__":
    main()
