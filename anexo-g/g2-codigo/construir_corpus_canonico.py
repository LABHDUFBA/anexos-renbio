#!/usr/bin/env python3
"""Constrói um corpus canônico com uma linha por mensagem do Telegram.

A chave de ocorrência é (chat_id, message_id). Duplicações técnicas só são
removidas quando texto e data concordam. Entre cópias concordantes, preserva-se
o registro com maior completude informacional.
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

REQUIRED_FIELDS = ("chat_id", "message_id", "all_text", "strict_date")
INVARIANT_FIELDS = ("chat_id", "message_id", "all_text", "strict_date")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON array exportado do Elasticsearch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="darwin_ocorrencias_canonicas")
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


def information_score(value: Any) -> int:
    """Conta folhas informativas, ignorando None, strings vazias e coleções vazias."""
    if value is None or value == "":
        return 0
    if isinstance(value, dict):
        return sum(information_score(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(information_score(v) for v in value)
    return 1


def record_score(record: dict[str, Any]) -> int:
    return information_score(record)


def is_present(value: Any) -> bool:
    return information_score(value) > 0


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


def validate_records(records: list[dict[str, Any]]) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("O arquivo deve conter um array JSON não vazio")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Registro {index} não é um objeto JSON")
        missing = [field for field in REQUIRED_FIELDS if record.get(field) in (None, "")]
        if missing:
            raise ValueError(f"Registro {index} sem campos obrigatórios: {missing}")


def choose_canonical(group: list[tuple[int, dict[str, Any]]]) -> tuple[int, dict[str, Any], list[str]]:
    divergent_fields: list[str] = []
    all_fields = sorted({field for _, record in group for field in record})
    for field in all_fields:
        values = {canonical_json(record.get(field)) for _, record in group}
        if len(values) > 1:
            divergent_fields.append(field)

    invariant_conflicts = [field for field in INVARIANT_FIELDS if field in divergent_fields]
    if invariant_conflicts:
        key = (group[0][1].get("chat_id"), group[0][1].get("message_id"))
        raise ValueError(f"Colisão na chave {key}: campos invariantes divergentes {invariant_conflicts}")

    # max() garante maior completude; a serialização resolve empates de forma estável.
    chosen_index, chosen = max(
        group,
        key=lambda item: (record_score(item[1]), canonical_json(item[1])),
    )
    return chosen_index, chosen, divergent_fields


def build_canonical(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, Any], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record["chat_id"], record["message_id"])].append((index, record))

    canonical: list[dict[str, Any]] = []
    duplicate_audit: list[dict[str, Any]] = []

    for (chat_id, message_id), group in groups.items():
        chosen_index, chosen, divergent_fields = choose_canonical(group)
        output = dict(chosen)
        output["occurrence_id"] = f"{chat_id}:{message_id}"
        output["text_sha256"] = sha256_text(str(chosen["all_text"]))
        output["ingest_multiplicity"] = len(group)
        canonical.append(output)

        if len(group) > 1:
            duplicate_audit.append(
                {
                    "occurrence_id": output["occurrence_id"],
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text_sha256": output["text_sha256"],
                    "multiplicity": len(group),
                    "input_rows": ",".join(str(index) for index, _ in group),
                    "chosen_input_row": chosen_index,
                    "information_scores": ",".join(str(record_score(record)) for _, record in group),
                    "api_ids": ",".join(sorted({str(record.get("__api_id__")) for _, record in group})),
                    "divergent_fields": ",".join(divergent_fields),
                }
            )

    canonical.sort(key=lambda row: (row["strict_date"], row["chat_id"], row["message_id"]))
    duplicate_audit.sort(key=lambda row: (row["chat_id"], row["message_id"]))
    return canonical, duplicate_audit


def build_clusters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["text_sha256"]].append(record)

    rows: list[dict[str, Any]] = []
    for text_hash, group in groups.items():
        chat_counts = Counter(record["chat_id"] for record in group)
        dates = [int(record["strict_date"]) for record in group]
        rows.append(
            {
                "text_sha256": text_hash,
                "ocorrencias": len(group),
                "chats": len(chat_counts),
                "titulos": len({record.get("chat_title") for record in group}),
                "primeira_data": min(dates),
                "ultima_data": max(dates),
                "amplitude_segundos": max(dates) - min(dates),
                "repeticao_intrachat": any(count > 1 for count in chat_counts.values()),
                "replicacao_interchat": len(chat_counts) > 1,
                "encaminhamento_em_alguma_ocorrencia": any(
                    is_present(record.get("forward_from_chat"))
                    or is_present(record.get("forward_chat_id"))
                    for record in group
                ),
            }
        )
    rows.sort(key=lambda row: (-row["ocorrencias"], row["text_sha256"]))
    return rows


def main() -> None:
    args = parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    validate_records(records)
    canonical, duplicate_audit = build_canonical(records)
    clusters = build_clusters(canonical)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.prefix}.json"
    csv_path = args.output_dir / f"{args.prefix}.csv"
    duplicates_path = args.output_dir / f"{args.prefix}_duplicacoes_tecnicas.csv"
    clusters_path = args.output_dir / f"{args.prefix}_clusters_textuais.csv"
    report_path = args.output_dir / f"{args.prefix}_relatorio.md"

    json_path.write_text(json.dumps(canonical, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, canonical)
    write_csv(duplicates_path, duplicate_audit)
    write_csv(clusters_path, clusters)

    repeated_clusters = [row for row in clusters if row["ocorrencias"] > 1]
    intrachat_only = [row for row in repeated_clusters if row["repeticao_intrachat"] and not row["replicacao_interchat"]]
    interchat_only = [row for row in repeated_clusters if row["replicacao_interchat"] and not row["repeticao_intrachat"]]
    both = [row for row in repeated_clusters if row["replicacao_interchat"] and row["repeticao_intrachat"]]
    forwarded_rows = sum(
        is_present(record.get("forward_from_chat")) or is_present(record.get("forward_chat_id"))
        for record in canonical
    )

    report = f"""# Relatório — corpus canônico de ocorrências

**Gerado em:** {datetime.now().isoformat(timespec='seconds')}

## Entrada

- Arquivo: `{args.input.name}`
- SHA-256: `{sha256_file(args.input)}`
- Linhas: {len(records)}

## Regra canônica

- Chave: `(chat_id, message_id)`.
- Campos obrigatórios: {', '.join(REQUIRED_FIELDS)}.
- Texto e data devem concordar em chaves repetidas.
- Entre cópias concordantes, preserva-se o registro com maior completude informacional.
- Empates são resolvidos pela representação JSON canônica, garantindo resultado estável.

## Resultado

- Mensagens canônicas: {len(canonical)}
- Linhas técnicas removidas: {len(records) - len(canonical)}
- Chaves com multiplicidade maior que 1: {len(duplicate_audit)}
- Textos únicos: {len(clusters)}
- Cópias textuais adicionais: {len(canonical) - len(clusters)}
- Mensagens com metadado de encaminhamento: {forwarded_rows}

## Clusters textuais repetidos

- Textos com duas ou mais ocorrências: {len(repeated_clusters)}
- Somente repetição intrachat: {len(intrachat_only)}
- Somente replicação interchat: {len(interchat_only)}
- Repetição intrachat e replicação interchat: {len(both)}

## Arquivos

- `{json_path.name}` — corpus canônico completo.
- `{csv_path.name}` — versão tabular.
- `{duplicates_path.name}` — auditoria das chaves duplicadas.
- `{clusters_path.name}` — agregação por hash textual, sem reproduzir o texto.

## Integridade dos produtos

- JSON canônico: `{sha256_file(json_path)}`
- CSV canônico: `{sha256_file(csv_path)}`
- Duplicações: `{sha256_file(duplicates_path)}`
- Clusters: `{sha256_file(clusters_path)}`
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"entrada={len(records)}")
    print(f"canonicas={len(canonical)}")
    print(f"removidas={len(records) - len(canonical)}")
    print(f"textos_unicos={len(clusters)}")
    print(f"relatorio={report_path}")


if __name__ == "__main__":
    main()
