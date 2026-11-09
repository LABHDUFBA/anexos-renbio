#!/usr/bin/env python3
"""Formaliza os corpora textuais B e C a partir das ocorrências canônicas.

Corpus B contém um registro por texto exato no universo completo. Corpus C é o
subconjunto dos textos usados no artigo histórico. Corpus C-recorrências contém
todas as ocorrências canônicas desses textos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical", type=Path, help="Corpus A-canônico em JSON")
    parser.add_argument("historical", type=Path, help="Corpus histórico em JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="Identificador comum dos produtos")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_present(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, dict):
        return any(is_present(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(is_present(item) for item in value)
    return True


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} não contém um array JSON não vazio")
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} contém elementos que não são objetos")
    return rows


def validate_canonical(rows: list[dict[str, Any]]) -> None:
    required = ("chat_id", "message_id", "all_text", "strict_date", "occurrence_id", "text_sha256")
    for index, row in enumerate(rows):
        missing = [field for field in required if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"Ocorrência canônica {index} sem {missing}")
        expected_occurrence = f"{row['chat_id']}:{row['message_id']}"
        if row["occurrence_id"] != expected_occurrence:
            raise ValueError(f"occurrence_id inconsistente na linha {index}")
        if row["text_sha256"] != sha256_text(str(row["all_text"])):
            raise ValueError(f"text_sha256 inconsistente na linha {index}")
    occurrence_ids = [row["occurrence_id"] for row in rows]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError("Corpus canônico contém occurrence_id duplicado")


def build_text_corpus(
    canonical: list[dict[str, Any]], historical_texts: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    texts_by_hash: dict[str, str] = {}
    for row in canonical:
        text_hash = row["text_sha256"]
        text = str(row["all_text"])
        previous = texts_by_hash.setdefault(text_hash, text)
        if previous != text:
            raise ValueError(f"Colisão SHA-256 detectada: {text_hash}")
        groups[text_hash].append(row)

    corpus_b: list[dict[str, Any]] = []
    for text_hash, occurrences in groups.items():
        occurrences = sorted(
            occurrences,
            key=lambda row: (int(row["strict_date"]), int(row["chat_id"]), int(row["message_id"])),
        )
        representative = dict(occurrences[0])
        chat_counts = Counter(row["chat_id"] for row in occurrences)
        dates = [int(row["strict_date"]) for row in occurrences]
        representative.update(
            {
                "text_id": text_hash,
                "representative_occurrence_id": occurrences[0]["occurrence_id"],
                "text_occurrence_count": len(occurrences),
                "text_chat_count": len(chat_counts),
                "text_title_count": len({row.get("chat_title") for row in occurrences}),
                "text_first_date": min(dates),
                "text_last_date": max(dates),
                "text_span_seconds": max(dates) - min(dates),
                "text_repeated_intrachat": any(count > 1 for count in chat_counts.values()),
                "text_replicated_interchat": len(chat_counts) > 1,
                "text_any_forward_metadata": any(
                    is_present(row.get("forward_from_chat"))
                    or is_present(row.get("forward_chat_id"))
                    for row in occurrences
                ),
                "in_historical_corpus": representative["all_text"] in historical_texts,
                "occurrence_ids": [row["occurrence_id"] for row in occurrences],
                "chat_ids": sorted(chat_counts),
            }
        )
        corpus_b.append(representative)

    corpus_b.sort(key=lambda row: (row["text_first_date"], row["text_id"]))
    corpus_c = [row for row in corpus_b if row["in_historical_corpus"]]
    return corpus_b, corpus_c


def product_entry(path: Path, rows: int, sensitivity: str, description: str) -> dict[str, Any]:
    return {
        "file": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "sensitivity": sensitivity,
        "description": description,
    }


def main() -> None:
    args = parse_args()
    canonical = load_json_array(args.canonical)
    historical = load_json_array(args.historical)
    validate_canonical(canonical)

    historical_texts = {str(row.get("all_text")) for row in historical if row.get("all_text") is not None}
    if len(historical_texts) != len(historical):
        raise ValueError("Corpus histórico não possui um texto distinto por registro")

    corpus_b, corpus_c = build_text_corpus(canonical, historical_texts)
    b_texts = {str(row["all_text"]) for row in corpus_b}
    missing_historical = historical_texts.difference(b_texts)
    if missing_historical:
        raise ValueError(f"Há {len(missing_historical)} textos históricos ausentes do Corpus B")

    historical_hashes = {sha256_text(text) for text in historical_texts}
    corpus_c_occurrences = [row for row in canonical if row["text_sha256"] in historical_hashes]
    corpus_c_occurrences.sort(key=lambda row: (row["strict_date"], row["chat_id"], row["message_id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "b_json": args.output_dir / f"darwin_corpus_b_textos_unicos_{args.tag}.json",
        "b_csv": args.output_dir / f"darwin_corpus_b_textos_unicos_{args.tag}.csv",
        "c_json": args.output_dir / f"darwin_corpus_c_historico_{args.tag}.json",
        "c_csv": args.output_dir / f"darwin_corpus_c_historico_{args.tag}.csv",
        "c_occ_json": args.output_dir / f"darwin_corpus_c_ocorrencias_{args.tag}.json",
        "c_occ_csv": args.output_dir / f"darwin_corpus_c_ocorrencias_{args.tag}.csv",
        "manifest_json": args.output_dir / f"darwin_manifesto_corpora_{args.tag}.json",
        "manifest_md": args.output_dir / f"darwin_manifesto_corpora_{args.tag}.md",
    }

    write_json(paths["b_json"], corpus_b)
    write_csv(paths["b_csv"], corpus_b)
    write_json(paths["c_json"], corpus_c)
    write_csv(paths["c_csv"], corpus_c)
    write_json(paths["c_occ_json"], corpus_c_occurrences)
    write_csv(paths["c_occ_csv"], corpus_c_occurrences)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "definitions": {
            "A_bruto": "Linhas retornadas pela Scroll API; pode conter duplicação técnica de ingestão.",
            "A_canonico": "Uma ocorrência por (chat_id, message_id).",
            "B": "Um registro por all_text exato sobre A-canônico; representante é a ocorrência mais antiga.",
            "C": "Textos de B que pertencem ao conjunto histórico de 1.269 textos.",
            "C_ocorrencias": "Todas as ocorrências canônicas dos textos de C.",
        },
        "representative_rule": "Menor strict_date; desempate por chat_id e message_id.",
        "sources": [
            {
                "file": args.canonical.name,
                "rows": len(canonical),
                "sha256": sha256_file(args.canonical),
                "role": "Corpus A-canônico",
            },
            {
                "file": args.historical.name,
                "rows": len(historical),
                "sha256": sha256_file(args.historical),
                "role": "Corpus histórico preservado",
            },
        ],
        "counts": {
            "A_canonico_ocorrencias": len(canonical),
            "B_textos_unicos": len(corpus_b),
            "C_textos_historicos": len(corpus_c),
            "C_ocorrencias": len(corpus_c_occurrences),
            "B_fora_de_C": len(corpus_b) - len(corpus_c),
        },
        "products": [],
        "versioning_policy": {
            "restricted": "Arquivos com all_text, chat_id, message_id, títulos ou metadados reidentificáveis; não enviar ao remoto sem autorização.",
            "review_before_versioning": "Agregados por hash e auditorias sem texto; conferir risco residual antes do commit.",
            "versionable": "Scripts, planejamento e manifestos sem conteúdo de mensagens.",
        },
    }

    manifest["products"] = [
        product_entry(paths["b_json"], len(corpus_b), "restricted", "Corpus B em JSON"),
        product_entry(paths["b_csv"], len(corpus_b), "restricted", "Corpus B em CSV"),
        product_entry(paths["c_json"], len(corpus_c), "restricted", "Corpus C em JSON"),
        product_entry(paths["c_csv"], len(corpus_c), "restricted", "Corpus C em CSV"),
        product_entry(paths["c_occ_json"], len(corpus_c_occurrences), "restricted", "Ocorrências do Corpus C em JSON"),
        product_entry(paths["c_occ_csv"], len(corpus_c_occurrences), "restricted", "Ocorrências do Corpus C em CSV"),
    ]
    paths["manifest_json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Manifesto dos corpora Darwin

**Gerado em:** {manifest['generated_at']}

## Linhagem

| Corpus | Unidade | Registros |
|---|---|---:|
| A-canônico | mensagem do Telegram por `(chat_id, message_id)` | {len(canonical)} |
| B | texto exato único | {len(corpus_b)} |
| C | texto histórico presente em B | {len(corpus_c)} |
| C-recorrências | mensagem canônica cujo texto pertence a C | {len(corpus_c_occurrences)} |

## Regras

- Corpus B é derivado do A-canônico por igualdade exata de `all_text`.
- O representante de cada texto é a ocorrência com menor `strict_date`; empates usam `chat_id` e `message_id`.
- Contagens agregadas de ocorrências e chats permanecem anexadas a cada texto do Corpus B.
- Corpus C preserva exatamente os {len(historical_texts)} textos do arquivo histórico.
- Nenhum texto histórico ficou ausente do Corpus B.

## Proveniência

- A-canônico: `{args.canonical.name}` — `{sha256_file(args.canonical)}`.
- Histórico: `{args.historical.name}` — `{sha256_file(args.historical)}`.

## Produtos e sensibilidade

| Arquivo | Linhas | Classe | SHA-256 |
|---|---:|---|---|
"""
    for product in manifest["products"]:
        report += f"| `{product['file']}` | {product['rows']} | {product['sensitivity']} | `{product['sha256']}` |\n"
    report += """

Todos os produtos tabulares acima são **restritos**, pois contêm texto e metadados reidentificáveis. Não devem ser enviados ao remoto sem autorização explícita e revisão ética. Scripts, arquivos de planejamento e este manifesto não reproduzem mensagens e podem ser avaliados separadamente para versionamento.
"""
    paths["manifest_md"].write_text(report, encoding="utf-8")

    print(f"A_canonico={len(canonical)}")
    print(f"B={len(corpus_b)}")
    print(f"C={len(corpus_c)}")
    print(f"C_ocorrencias={len(corpus_c_occurrences)}")
    print(f"manifesto={paths['manifest_md']}")


if __name__ == "__main__":
    main()
