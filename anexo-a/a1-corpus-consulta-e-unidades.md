# Corpus, consulta e unidades de análise

## Consulta

- Campo pesquisado: `all_text`.
- Consulta: `darwin* AND NOT darwinbrandao`.
- Janela nominal: 1º de janeiro de 2016 a 26 de maio de 2026.
- Período efetivamente observado: 9 de outubro de 2016 a 25 de maio de 2026.

## Constituição do corpus canônico

A extração completa retornou 4.064 linhas. A chave `(chat_id, message_id)` identificou quatro duplicações técnicas concordantes, removidas por regra determinística. O corpus canônico contém **4.060 mensagens**.

O agrupamento local por igualdade exata de `all_text` produziu **2.718 textos distintos**. Versões parcialmente modificadas permanecem como textos diferentes.

## Unidades analíticas

- **Mensagem:** frequência temporal, tipos de chat, visualizações, encaminhamentos, repetição e circulação.
- **Texto distinto:** idioma, TF-IDF, G², KWIC, eixos temáticos, links e análise qualitativa.

## Deduplicação preliminar incompleta

Uma extração anterior por `all_text.keyword` retornou 1.269 textos porque o subcampo não indexava textos acima do limite configurado. Esse conjunto foi preservado apenas como registro de auditoria e não integra o argumento principal.

## Limites

- O painel foi constituído por bola de neve digital e variou ao longo do tempo.
- A série temporal descreve o corpus observado; não mede prevalência relativa no Telegram.
- Texto idêntico em chats distintos demonstra replicação, não direção de difusão.
- Direção confirmada exige `forward_from_chat.id`; precedência temporal isolada não prova encaminhamento.
