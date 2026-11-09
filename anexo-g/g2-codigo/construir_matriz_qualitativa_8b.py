#!/usr/bin/env python3
"""Constrói a matriz qualitativa da Fase 8B a partir dos 55 candidatos únicos.

As decisões abaixo registram pré-codificação acadêmica auditável. Elas não
substituem a responsabilidade interpretativa final dos autores.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

# candidato: (codigo, subcodigo, posicao, uso_darwin, confianca, relevancia,
#             prioridade_artigo, observacao)
CODING: dict[str, tuple[str, str, str, str, str, str, str, str]] = {
    "C01": ("correcao_conceitual", "darwinismo_social_nazismo", "citação de reportagem crítica", "objeto de distinção histórica", "alta", "alta", "sim", "Texto longo externo reproduzido; contesta a derivação direta Darwin→nazismo."),
    "C02": ("darwin_racismo", "genealogia_acusatoria", "divulgação promocional de documentário", "origem atribuída do racismo", "média", "alta", "sim", "A legenda endossa o enquadramento promocional, mas o documentário não foi analisado integralmente."),
    "C03": ("darwin_autoridade", "marxismo_pro_ciencia", "afirmação programática com citações", "autoridade científica positiva", "alta", "alta", "sim", "Contraexemplo de apropriação positiva de Darwin e da evolução por canal marxista."),
    "C04": ("apropriacao_racial", "evolucionismo_catolico_racialista", "ensaio afirmativo", "fundamento de hierarquia racial", "alta", "alta", "sim", "Aceita evolução e compatibilidade religiosa, mas a converte em explicação racial hierárquica."),
    "C05": ("antievolucionismo", "escola_conversao_de_darwin", "polemica religiosa afirmativa", "alvo científico e moral", "alta", "alta", "sim", "Articula professora, universidade, falta de provas, conversão de Darwin e criação bíblica."),
    "C06": ("darwinismo_social", "politicas_sociais_eua", "texto declarado como gerado por IA", "analogia social crítica", "alta", "média", "não", "Relevante para circulação de texto sintético; não usar como voz espontânea de participante."),
    "C07": ("marx_darwin", "carta_e_religiao", "reprodução de narrativa e carta", "contraponto moral a Marx", "média", "média", "não", "Mistura interpretação acusatória e transcrição documental; requer auditoria histórica externa."),
    "C08": ("apropriacao_direita", "anti_igualitarismo", "citação de Darwin seguida de comentário", "autoridade naturalista para direita anti-igualitária", "alta", "alta", "sim", "A legenda conecta seleção natural, materialismo e projeto de renovação da direita."),
    "C09": ("critica_filosofica", "altruismo_e_materialismo", "citação de coluna", "problema moral atribuído ao darwinismo", "média", "média", "não", "A posição imediata é de texto reproduzido; autoria indicada na própria mensagem."),
    "C10": ("antievolucionismo", "mobilizacao_educacional", "chamada promocional afirmativa", "farsa e ameaça a valores", "alta", "alta", "sim", "Convoca jovens de escolas e universidades a combater o darwinismo."),
    "C11": ("correcao_conceitual", "biologia_versus_dominação", "resposta declarada como gerada por IA", "objeto de distinção conceitual", "alta", "média", "não", "Correção explícita, mas marcada por hashtag ChatGPT; tratar como conteúdo sintético circulante."),
    "C12": ("critica_extrema_direita", "yockey_capitalismo_materialismo", "ensaio ideológico com citação", "alvo materialista", "alta", "alta", "sim", "Critica o darwinismo jornalístico sem rejeitar adaptação; associa-o a capitalismo e animalização."),
    "C13": ("antievolucionismo", "darwin_blefe", "reprodução promocional de livro", "fraudador e autor inseguro", "alta", "alta", "sim", "Texto longo atribuído a Evolution News/Discovery Institute; posição é de circulação promocional e as atribuições requerem cotejo externo."),
    "C14": ("darwin_nazismo", "higiene_racial", "reprodução de texto de terceiro", "origem atribuída de propaganda racial", "média", "média", "não", "Reúne antissemitismo nazista e suposta inspiração darwiniana; fonte imediata não está clara."),
    "C15": ("design_inteligente", "bioquimica_contra_evolucao", "sinopse promocional", "teoria científica insuficiente", "alta", "média", "não", "Sinopse de A caixa preta de Darwin; não equivale a argumento original do canal."),
    "C16": ("darwin_racismo", "titulo_e_links", "título promocional", "rótulo temático", "alta", "baixa", "não", "Sem desenvolvimento textual; insuficiente para inferir posição."),
    "C17": ("antievolucionismo", "curriculo_ortodoxo", "citação atribuída a Alexander Dugin", "ciência anticristã a excluir da escola", "alta", "alta", "sim", "Propõe retirar evolução da educação escolar em nome de valores religiosos e tradicionais."),
    "C18": ("racialismo_direita", "dark_enlightenment", "citação extensa de Nick Land", "base naturalista para desigualdade", "alta", "alta", "sim", "A circulação positiva do trecho é observável; adesão integral do canal requer cautela."),
    "C19": ("historia_darwinismo", "malthus_capitalismo", "reprodução de ensaio histórico", "teoria com implicações políticas", "alta", "média", "não", "Análise conceitual longa; não contém posição antissemita apesar do chat de origem."),
    "C20": ("marx_darwin", "hegel_materialismo", "afirmação conspiratória", "elo entre Hegel e marxismo", "alta", "alta", "sim", "Constrói equivalência entre dialética, seleção natural e marxismo."),
    "C21": ("design_inteligente", "indice_de_links", "lista de referências", "rótulo bibliográfico", "alta", "baixa", "não", "Aponta para outro post; sem argumento recuperado."),
    "C22": ("compatibilidade_religiosa", "cristianismo_evolucao", "título de vídeo", "objeto de conciliação", "alta", "baixa", "não", "Caso negativo relevante, mas conteúdo do vídeo não integra a mensagem."),
    "C23": ("design_inteligente", "argumentacao_desenvolvida", "ensaio afirmativo", "teoria rival insuficiente", "alta", "alta", "sim", "Mobiliza complexidade irredutível, fósseis, DNA, biogênese, ajuste fino e moralidade."),
    "C24": ("correcao_politica", "nazismo_materialismo", "argumento em debate", "apropriação nazista criticada", "alta", "alta", "sim", "Distingue dignidade ontológica de hierarquia racial e descreve apropriação nazista do darwinismo."),
    "C25": ("negacao_historica", "holocausto_e_darwinismo", "afirmação extremista", "influência no nazismo recusada", "alta", "alta", "sim", "Contesta darwinismo social nazista e incorpora negação do Holocausto; não tratar como correção histórica neutra."),
    "C26": ("darwin_hitler", "luta_e_direito", "citação de livro", "genealogia intelectual do nazismo", "alta", "média", "não", "Trecho atribuído a Richard Weikart; posição imediata é citação editorializada."),
    "C27": ("conspiracao_antissemita", "darwin_hitler_talmud_maconaria", "ataque direto em debate", "agente maçônico e fundamento racial", "alta", "alta", "sim", "Enunciado antissemita, religioso e insultuoso; posição afirmativa inequívoca."),
    "C28": ("antievolucionismo", "video_ray_comfort", "sinopse promocional", "autor de confusão científica", "alta", "média", "não", "A mensagem resume um vídeo; não permite verificar as entrevistas citadas."),
    "C29": ("darwin_racialismo", "descendencia_do_homem", "afirmação expositiva", "autor racialista e origem de usos sociais", "média", "média", "não", "Generaliza trechos históricos sem citação precisa; exige confronto com fonte primária."),
    "C30": ("antievolucionismo", "mitos_do_darwinismo", "teaser com link", "ortodoxia a ser quebrada", "alta", "baixa", "não", "Conteúdo principal está fora da mensagem."),
    "C31": ("antievolucionismo", "sobrevivencia_tautologia", "resumo com link", "teoria circular", "média", "média", "não", "Apresenta uma objeção, mas depende do texto externo para avaliação completa."),
    "C32": ("correcao_marxista", "critica_darwinismo_social", "artigo doutrinário com citação", "ciência distinguida de extrapolação social", "alta", "alta", "sim", "Canal marxista rejeita darwinismo social como extensão idealista das leis biológicas."),
    "C33": ("marx_darwin", "plinio_salgado_materialismo", "citação atribuída", "matriz materialista de Marx", "alta", "média", "não", "Reprodução de Plínio Salgado; útil como circulação de genealogia ideológica."),
    "C34": ("conspiracao_religiosa", "maconaria_satanismo_ateismo", "afirmação conspiratória", "agente materialista e satânico", "alta", "alta", "sim", "Articula família, maçonaria, ateísmo, ocultismo e perda de fé."),
    "C35": ("determinismo_biologico", "lysenko_classe_genero_migracao", "ensaio político afirmativo", "ciência biológica contra marxismo", "alta", "alta", "sim", "Usa Lysenko para naturalizar classe, sexo e diferenças populacionais; inferências históricas e causais não foram verificadas."),
    "C36": ("darwin_awards", "eliminacao_moral", "afirmação jocosa", "seleção social moralizada", "alta", "alta", "sim", "Aplica seleção natural à morte de três pessoas e celebra a remoção de suas características."),
    "C37": ("metadiscurso", "questionamento_contextual", "pergunta", "referência contestada", "alta", "baixa", "não", "Sem contexto adjacente, não permite saber qual associação está sendo questionada."),
    "C38": ("darwin_nazismo", "titulo_de_video", "título promocional", "origem atribuída do racismo", "alta", "baixa", "não", "Título de vídeo sem desenvolvimento."),
    "C39": ("marx_darwin", "link_marxista", "link e título", "afinidade intelectual", "alta", "média", "não", "Valência positiva, mas a argumentação está na página externa."),
    "C40": ("antievolucionismo", "evolucao_mentira", "chamada e link", "aberração e mentira", "alta", "média", "não", "Afirmação direta curta; já representada por casos desenvolvidos."),
    "C41": ("darwin_awards", "competicao_jocosa", "humor", "premiação negativa", "alta", "alta", "sim", "Exemplo histórico claro do enquadramento competitivo."),
    "C42": ("darwin_awards", "enquete_interna", "comentário jocoso", "rótulo memético", "baixa", "baixa", "não", "Contexto insuficiente para interpretar a enquete."),
    "C43": ("conspiracao", "sionismo", "pergunta", "identidade conspiratória possível", "alta", "baixa", "não", "Pergunta isolada; não prova adesão à associação."),
    "C44": ("darwin_autoridade", "titulo_generico", "título com link", "referência científica", "alta", "baixa", "não", "Sem conteúdo suficiente."),
    "C45": ("marx_darwin", "ateismo_religioso", "afirmação", "causa do ateísmo via marxismo", "alta", "alta", "não", "Exemplo histórico conciso; casos longos agora permitem maior contextualização."),
    "C46": ("metadiscurso", "desafio", "imperativo/pergunta", "autoridade a ser contestada", "alta", "baixa", "não", "Enunciado isolado sem alvo identificável."),
    "C47": ("darwinismo_social", "fragmento", "fragmento", "rótulo", "alta", "baixa", "não", "Sem posição enunciativa recuperável."),
    "C48": ("metadiscurso", "participante_darwinista", "pergunta", "identidade atribuída a participante", "alta", "baixa", "não", "Sem contexto adjacente."),
    "C49": ("compatibilidade_religiosa", "debate_publico", "anúncio interrogativo", "objeto de conciliação", "alta", "média", "não", "Evidencia debate, não posição final dos interlocutores."),
    "C50": ("retracao_de_darwin", "checagem_da_narrativa", "título interrogativo de vídeo", "alvo de verificação", "alta", "média", "não", "Caso negativo para a narrativa de retratação; conteúdo está no vídeo."),
    "C51": ("tensao_interna", "redpill_mgtow", "observação metadiscursiva", "signo apropriado e negado", "alta", "alta", "sim", "Formula diretamente a contradição entre uso de Darwin e negação da evolução."),
    "C52": ("compatibilidade_religiosa", "video_marcos_eberlin", "pergunta e chamada", "objeto de controvérsia", "alta", "baixa", "não", "Sem conteúdo argumentativo na mensagem."),
    "C53": ("materialismo", "link_externo", "link sem legenda", "tema externo", "alta", "baixa", "não", "Não permite inferir enquadramento além do URL."),
    "C54": ("contestacao_darwin_nazismo", "pedido_de_provas", "pergunta contestatória com insulto", "associação rejeitada", "alta", "alta", "sim", "Contraexemplo histórico claro; não equivale a defesa acadêmica da teoria."),
    "C55": ("marx_darwin", "afinidade_intelectual", "afirmação expositiva", "referência admirada por Marx", "média", "média", "não", "Resumo sem fonte explícita na própria mensagem."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    by_text: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_text.setdefault(
            row["text_id"],
            {**row, "candidate_axes_set": set(), "scopes_set": set(), "best_rank": 999},
        )
        entry["candidate_axes_set"].add(row["axis"])
        entry["scopes_set"].add(row["scope"])
        entry["best_rank"] = min(int(entry["best_rank"]), int(row["rank"]))
    ordered = sorted(
        by_text.values(),
        key=lambda row: (
            str(row["in_corpus_c"]) == "True",
            -int(row["retrieval_count"]),
            -int(row["text_chars"]),
            row["text_id"],
        ),
    )
    if len(ordered) != 55:
        raise ValueError(f"Esperados 55 textos únicos; encontrados {len(ordered)}")
    if set(CODING) != {f"C{i:02d}" for i in range(1, 56)}:
        raise ValueError("A tabela de codificação não cobre C01–C55")

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        candidate = f"C{index:02d}"
        codigo, subcodigo, posicao, uso, confianca, relevancia, prioridade, observacao = CODING[candidate]
        output_rows.append(
            {
                "candidate_id": candidate,
                "text_id": row["text_id"],
                "in_corpus_c": row["in_corpus_c"],
                "candidate_axes": " | ".join(sorted(row["candidate_axes_set"])),
                "retrieval_count": row["retrieval_count"],
                "best_rank": row["best_rank"],
                "text_chars": row["text_chars"],
                "chat_id": row["chat_id"],
                "chat_title": row["chat_title"],
                "type": row["type"],
                "strict_date": row["strict_date"],
                "message_id": row["message_id"],
                "codigo": codigo,
                "subcodigo": subcodigo,
                "posicao_enunciativa": posicao,
                "uso_de_darwin": uso,
                "confianca_fonte_posicao": confianca,
                "relevancia_analitica": relevancia,
                "prioridade_artigo": prioridade,
                "status_revisao": "fonte_validada_precodificacao_academica",
                "observacao": observacao,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Matriz criada: {len(output_rows)} textos; prioridade artigo: {sum(r['prioridade_artigo'] == 'sim' for r in output_rows)}")


if __name__ == "__main__":
    main()
