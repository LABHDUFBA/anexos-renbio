#!/usr/bin/env python3
"""Analisa recorrência textual e circulação observável no corpus canônico."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Sao_Paulo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Corpus A-canônico")
    parser.add_argument("--b", type=Path, required=True, help="Corpus B")
    parser.add_argument("--c", type=Path, required=True, help="Corpus C")
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


def source_chat(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("forward_from_chat")
    if not isinstance(value, dict):
        return None
    source_id = value.get("id")
    if not present(source_id):
        return None
    try:
        source_chat_id = int(str(source_id))
    except (TypeError, ValueError):
        return None
    return {
        "source_chat_id": source_chat_id,
        "source_chat_title": value.get("title") or value.get("username") or "",
        "source_chat_type": value.get("type") or "",
    }


def has_forward_metadata(record: dict[str, Any]) -> bool:
    return source_chat(record) is not None


def to_iso(ts: Any) -> str:
    if ts in (None, ""):
        return ""
    return datetime.fromtimestamp(float(ts), tz=TZ).isoformat()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def classify_cluster(rows: list[dict[str, Any]]) -> str:
    chat_counts = Counter(row["chat_id"] for row in rows)
    intrachat = any(count > 1 for count in chat_counts.values())
    interchat = len(chat_counts) > 1
    if intrachat and interchat:
        return "intra_e_interchat"
    if intrachat:
        return "somente_intrachat"
    if interchat:
        return "somente_interchat"
    return "sem_repeticao"


def build_cluster_rows(a_rows: list[dict[str, Any]], c_hashes: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in a_rows:
        grouped[row["text_sha256"]].append(row)

    cluster_rows: list[dict[str, Any]] = []
    for text_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: (int(row["strict_date"]), int(row["chat_id"]), int(row["message_id"])))
        chat_counts = Counter(row["chat_id"] for row in rows)
        title_counts = Counter(row.get("chat_title") or "" for row in rows)
        dates = [int(row["strict_date"]) for row in rows]
        source_edges = []
        for row in rows:
            source = source_chat(row)
            if source is None:
                continue
            source_edges.append((source["source_chat_id"], row["chat_id"]))
        cluster_rows.append(
            {
                "text_id": text_id,
                "in_corpus_c": text_id in c_hashes,
                "classe_repeticao": classify_cluster(rows),
                "ocorrencias": len(rows),
                "chats": len(chat_counts),
                "titulos": len(title_counts),
                "primeira_data": min(dates),
                "primeira_data_iso": to_iso(min(dates)),
                "ultima_data": max(dates),
                "ultima_data_iso": to_iso(max(dates)),
                "amplitude_segundos": max(dates) - min(dates),
                "primeiro_occurrence_id": rows[0]["occurrence_id"],
                "primeiro_chat_id": rows[0]["chat_id"],
                "primeiro_chat_title": rows[0].get("chat_title") or "",
                "primeiro_type": rows[0].get("type") or "",
                "ocorrencias_com_forward_from_chat_id": sum(has_forward_metadata(row) for row in rows),
                "ocorrencias_com_forward_chat_id": sum(present(row.get("forward_chat_id")) for row in rows),
                "source_chat_edges_distintas": len(set(source_edges)),
            }
        )
    return sorted(cluster_rows, key=lambda row: (-row["ocorrencias"], row["text_id"]))


def build_edges(a_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edge_counter: Counter[tuple[Any, Any, str, str, str, str]] = Counter()
    for row in a_rows:
        source = source_chat(row)
        if source is None:
            continue
        key = (
            source["source_chat_id"],
            row["chat_id"],
            source["source_chat_title"],
            row.get("chat_title") or "",
            source["source_chat_type"],
            row.get("type") or "",
        )
        edge_counter[key] += 1
    edges = [
        {
            "source_chat_id": key[0],
            "target_chat_id": key[1],
            "source_chat_title": key[2],
            "target_chat_title": key[3],
            "source_chat_type": key[4],
            "target_chat_type": key[5],
            "ocorrencias": count,
        }
        for key, count in edge_counter.items()
    ]
    return sorted(edges, key=lambda row: (-row["ocorrencias"], str(row["source_chat_id"]), str(row["target_chat_id"])))


def build_temporal_edges(cluster_rows: list[dict[str, Any]], grouped_a: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Cria arestas inferidas entre a primeira ocorrência e ocorrências posteriores do mesmo texto.

    Essas arestas NÃO provam direção de difusão. Elas registram apenas precedência temporal observada.
    """
    edges_counter: Counter[tuple[Any, Any, str, str]] = Counter()
    for cluster in cluster_rows:
        if cluster["classe_repeticao"] not in {"somente_interchat", "intra_e_interchat"}:
            continue
        rows = sorted(grouped_a[cluster["text_id"]], key=lambda row: (int(row["strict_date"]), int(row["chat_id"]), int(row["message_id"])))
        first = rows[0]
        for row in rows[1:]:
            if row["chat_id"] == first["chat_id"]:
                continue
            key = (first["chat_id"], row["chat_id"], first.get("chat_title") or "", row.get("chat_title") or "")
            edges_counter[key] += 1
    return sorted(
        [
            {
                "source_chat_id_inferido": key[0],
                "target_chat_id": key[1],
                "source_chat_title_inferido": key[2],
                "target_chat_title": key[3],
                "textos": count,
                "interpretacao": "precedencia_temporal_observada_nao_prova_difusao",
            }
            for key, count in edges_counter.items()
        ],
        key=lambda row: (-row["textos"], str(row["source_chat_id_inferido"]), str(row["target_chat_id"])),
    )


def build_trajectory_rows(cluster_rows: list[dict[str, Any]], grouped_a: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Registra a sequência temporal observada dos textos repetidos."""
    cluster_class = {row["text_id"]: row["classe_repeticao"] for row in cluster_rows}
    trajectory_rows: list[dict[str, Any]] = []
    for text_id, rows in grouped_a.items():
        if cluster_class[text_id] == "sem_repeticao":
            continue
        ordered = sorted(rows, key=lambda row: (int(row["strict_date"]), int(row["chat_id"]), int(row["message_id"])))
        first_ts = int(ordered[0]["strict_date"])
        for index, row in enumerate(ordered, start=1):
            source = source_chat(row) or {}
            trajectory_rows.append(
                {
                    "text_id": text_id,
                    "classe_repeticao": cluster_class[text_id],
                    "ordem_observada": index,
                    "delta_primeira_segundos": int(row["strict_date"]) - first_ts,
                    "occurrence_id": row["occurrence_id"],
                    "chat_id": row["chat_id"],
                    "chat_title": row.get("chat_title") or "",
                    "type": row.get("type") or "",
                    "strict_date": row["strict_date"],
                    "strict_date_iso": to_iso(row["strict_date"]),
                    "forward_source_chat_id": source.get("source_chat_id", ""),
                    "forward_source_chat_title": source.get("source_chat_title", ""),
                    "forward_source_chat_type": source.get("source_chat_type", ""),
                    "forward_message_id_or_legacy_field": row.get("forward_chat_id") or "",
                    "interpretacao": "sequencia_observada_nao_prova_difusao",
                }
            )
    return sorted(trajectory_rows, key=lambda row: (row["text_id"], int(row["ordem_observada"])))


def build_forward_nodes(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Constrói nós agregados da rede confirmada por metadado."""
    nodes: dict[Any, dict[str, Any]] = {}
    for edge in edges:
        for side in ("source", "target"):
            node_id = edge[f"{side}_chat_id"]
            node = nodes.setdefault(
                node_id,
                {
                    "chat_id": node_id,
                    "chat_title": edge.get(f"{side}_chat_title") or "",
                    "chat_type": edge.get(f"{side}_chat_type") or "",
                    "out_degree_confirmado": 0,
                    "in_degree_confirmado": 0,
                    "out_ocorrencias_confirmadas": 0,
                    "in_ocorrencias_confirmadas": 0,
                },
            )
            if not node["chat_title"] and edge.get(f"{side}_chat_title"):
                node["chat_title"] = edge[f"{side}_chat_title"]
            if not node["chat_type"] and edge.get(f"{side}_chat_type"):
                node["chat_type"] = edge[f"{side}_chat_type"]
        count = int(edge["ocorrencias"])
        nodes[edge["source_chat_id"]]["out_degree_confirmado"] += 1
        nodes[edge["source_chat_id"]]["out_ocorrencias_confirmadas"] += count
        nodes[edge["target_chat_id"]]["in_degree_confirmado"] += 1
        nodes[edge["target_chat_id"]]["in_ocorrencias_confirmadas"] += count
    return sorted(
        nodes.values(),
        key=lambda row: (-(int(row["out_ocorrencias_confirmadas"]) + int(row["in_ocorrencias_confirmadas"])), str(row["chat_id"])),
    )


def summarize(
    cluster_rows: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    temporal_edges: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    forward_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    total_texts = len(cluster_rows)
    repeated = [row for row in cluster_rows if row["classe_repeticao"] != "sem_repeticao"]
    repeated_c = [row for row in repeated if row["in_corpus_c"]]
    class_counts = Counter(row["classe_repeticao"] for row in cluster_rows)
    class_counts_c = Counter(row["classe_repeticao"] for row in cluster_rows if row["in_corpus_c"])
    amplitudes = [float(row["amplitude_segundos"]) for row in repeated]
    occurrence_counts = [float(row["ocorrencias"]) for row in repeated]
    interchat_edges = [row for row in edges if row["source_chat_id"] != row["target_chat_id"]]
    self_edges = [row for row in edges if row["source_chat_id"] == row["target_chat_id"]]
    edge_type_counts = Counter(
        (row.get("source_chat_type") or "unknown", row.get("target_chat_type") or "unknown")
        for row in interchat_edges
    )
    edge_type_occurrences = Counter()
    for row in interchat_edges:
        edge_type_occurrences[(row.get("source_chat_type") or "unknown", row.get("target_chat_type") or "unknown")] += int(row["ocorrencias"])
    return {
        "textos_total": total_texts,
        "textos_repetidos": len(repeated),
        "textos_repetidos_corpus_c": len(repeated_c),
        "classes": dict(class_counts),
        "classes_corpus_c": dict(class_counts_c),
        "ocorrencias_em_textos_repetidos": sum(int(row["ocorrencias"]) for row in repeated),
        "amplitude_segundos_textos_repetidos": stats(amplitudes),
        "ocorrencias_por_texto_repetido": stats(occurrence_counts),
        "arestas_forward_confirmadas": len(edges),
        "arestas_forward_confirmadas_interchat": len(interchat_edges),
        "arestas_forward_confirmadas_mesmo_chat": len(self_edges),
        "ocorrencias_em_arestas_forward_confirmadas": sum(int(row["ocorrencias"]) for row in edges),
        "ocorrencias_em_arestas_forward_interchat": sum(int(row["ocorrencias"]) for row in interchat_edges),
        "ocorrencias_em_arestas_forward_mesmo_chat": sum(int(row["ocorrencias"]) for row in self_edges),
        "arestas_forward_interchat_por_tipo": {f"{a}->{b}": c for (a, b), c in sorted(edge_type_counts.items())},
        "ocorrencias_forward_interchat_por_tipo": {f"{a}->{b}": c for (a, b), c in sorted(edge_type_occurrences.items())},
        "arestas_temporais_inferidas": len(temporal_edges),
        "textos_em_arestas_temporais_inferidas": sum(int(row["textos"]) for row in temporal_edges),
        "linhas_trajetorias_textos_repetidos": len(trajectory_rows),
        "nos_rede_forward_confirmada": len(forward_nodes),
        "top_edges_forward": interchat_edges[:20],
        "top_edges_temporais": temporal_edges[:20],
        "top_nodes_forward_confirmada": forward_nodes[:20],
    }


def main() -> None:
    args = parse_args()
    a_rows = load(args.a)
    b_rows = load(args.b)
    c_rows = load(args.c)
    c_hashes = {row["text_id"] for row in c_rows}
    b_hashes = {row["text_id"] for row in b_rows}
    a_hashes = {row["text_sha256"] for row in a_rows}
    if b_hashes != a_hashes:
        raise ValueError("Corpus B não cobre todos os hashes do A-canônico")
    if not c_hashes.issubset(b_hashes):
        raise ValueError("Corpus C não é subconjunto de B")

    grouped_a: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in a_rows:
        grouped_a[row["text_sha256"]].append(row)

    cluster_rows = build_cluster_rows(a_rows, c_hashes)
    edges = build_edges(a_rows)
    temporal_edges = build_temporal_edges(cluster_rows, grouped_a)
    trajectory_rows = build_trajectory_rows(cluster_rows, grouped_a)
    forward_nodes = build_forward_nodes(edges)
    summary = summarize(cluster_rows, edges, temporal_edges, trajectory_rows, forward_nodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = args.output_dir / f"circulacao_clusters_textuais_{args.tag}.csv"
    edges_path = args.output_dir / f"circulacao_arestas_forward_confirmadas_{args.tag}.csv"
    nodes_path = args.output_dir / f"circulacao_nos_forward_confirmados_{args.tag}.csv"
    temporal_edges_path = args.output_dir / f"circulacao_arestas_temporais_inferidas_{args.tag}.csv"
    trajectory_path = args.output_dir / f"circulacao_trajetorias_textos_repetidos_{args.tag}.csv"
    json_path = args.output_dir / f"circulacao_resumo_{args.tag}.json"
    md_path = args.output_dir / f"circulacao_resumo_{args.tag}.md"
    write_csv(clusters_path, cluster_rows)
    write_csv(edges_path, edges)
    write_csv(nodes_path, forward_nodes)
    write_csv(temporal_edges_path, temporal_edges)
    write_csv(trajectory_path, trajectory_rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# Recorrência e circulação — corpus Darwin

## Unidade e limites

- Unidade de ocorrência: mensagem canônica por `(chat_id, message_id)`.
- Unidade de recorrência: texto idêntico por SHA-256 de `all_text`.
- Encaminhamento confirmado entre chats exige `forward_from_chat.id` presente.
- `forward_chat_id` é tratado como possível ID da mensagem encaminhada na origem, não como chat de origem.
- Arestas temporais inferidas partem da primeira ocorrência observada de um texto e não provam direção de difusão.

## Recorrência textual

| Classe | Textos |
|---|---:|
| Sem repetição | {summary['classes'].get('sem_repeticao', 0)} |
| Somente intrachat | {summary['classes'].get('somente_intrachat', 0)} |
| Somente interchat | {summary['classes'].get('somente_interchat', 0)} |
| Intra e interchat | {summary['classes'].get('intra_e_interchat', 0)} |

- Textos totais: {summary['textos_total']}.
- Textos repetidos: {summary['textos_repetidos']}.
- Textos repetidos pertencentes ao corpus histórico C: {summary['textos_repetidos_corpus_c']}.
- Ocorrências em textos repetidos: {summary['ocorrencias_em_textos_repetidos']}.
- Mediana de ocorrências por texto repetido: {summary['ocorrencias_por_texto_repetido']['median']}.
- Mediana da amplitude temporal dos textos repetidos: {summary['amplitude_segundos_textos_repetidos']['median']} segundos.

## Encaminhamentos confirmados por metadado

- Arestas origem→destino distintas: {summary['arestas_forward_confirmadas']}.
- Ocorrências em arestas confirmadas: {summary['ocorrencias_em_arestas_forward_confirmadas']}.
- Nós distintos na rede confirmada: {summary['nos_rede_forward_confirmada']}.
- Arestas interchat distintas: {summary['arestas_forward_confirmadas_interchat']}.
- Ocorrências interchat confirmadas: {summary['ocorrencias_em_arestas_forward_interchat']}.
- Arestas em que origem e destino têm o mesmo `chat_id`: {summary['arestas_forward_confirmadas_mesmo_chat']} ({summary['ocorrencias_em_arestas_forward_mesmo_chat']} ocorrências). Elas indicam encaminhamento ou republicação interna, não circulação entre chats.

### Tipos de aresta interchat confirmada

| Origem→destino | Arestas | Ocorrências |
|---|---:|---:|
"""
    for label, count in summary["arestas_forward_interchat_por_tipo"].items():
        md += f"| {label} | {count} | {summary['ocorrencias_forward_interchat_por_tipo'].get(label, 0)} |\n"

    md += f"""

## Arestas temporais inferidas

- Arestas distintas: {summary['arestas_temporais_inferidas']}.
- Textos envolvidos: {summary['textos_em_arestas_temporais_inferidas']}.
- Linhas de trajetória observada para textos repetidos: {summary['linhas_trajetorias_textos_repetidos']}.
- Interpretação: precedência temporal observada, não prova de difusão.

## Arquivos

- `{clusters_path.name}` — classes de recorrência por texto.
- `{edges_path.name}` — arestas confirmadas por `forward_from_chat.id`.
- `{nodes_path.name}` — nós agregados da rede confirmada por metadado.
- `{temporal_edges_path.name}` — arestas inferidas por primeira ocorrência observada.
- `{trajectory_path.name}` — sequências temporais observadas de textos repetidos.
- `{json_path.name}` — resumo estruturado.
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"clusters={len(cluster_rows)} edges={len(edges)} nodes={len(forward_nodes)} temporal_edges={len(temporal_edges)} trajectories={len(trajectory_rows)}")
    print(f"relatorio={md_path}")


if __name__ == "__main__":
    main()
