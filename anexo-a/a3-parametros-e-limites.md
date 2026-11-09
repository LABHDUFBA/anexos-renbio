# Parâmetros, funções analíticas e limites

## TF-IDF

- unidade: 2.718 textos distintos;
- `ngram_range=(1, 2)`;
- `min_df=3`;
- `max_df=0.85`;
- frequência sublinear;
- normalização L2;
- palavras funcionais em português, inglês e espanhol removidas;
- derivados de `darwin*` retirados dos rankings;
- Anexo C: 100 primeiros unigramas e 100 primeiros bigramas, com ranking independente por tipo.

## Associações locais G² e KWIC

- alvo: ocorrências de `darwin*`;
- janela: cinco tokens à esquerda e cinco à direita;
- fundo: tokens fora das janelas no corpus;
- medida: razão de verossimilhança G²;
- KWIC: cinco tokens de cada lado;
- função: localizar proximidades lexicais, sem inferir adesão, causalidade ou posição política.

## Eixos exploratórios

Cinco eixos não excludentes foram aplicados por regras lexicais reproduzíveis. Eles localizam candidatos e não substituem a leitura contextual. As categorias combinam temas e repertórios discursivos; não constituem tipologia exaustiva.

## Social-RAG

O Social-RAG foi usado como mecanismo de recuperação. As respostas geradas não foram tratadas como evidência. Todos os casos analisados retornaram ao `all_text` restrito e aos metadados de origem.

## Circularidade metodológica

A análise lexical orientou os eixos, e os eixos orientaram parte das perguntas do Social-RAG. Isso cria risco de recuperar material que confirme o esquema inicial. A mitigação incluiu perguntas adicionais sobre textos longos, contestações, casos negativos, antievolucionismo e relações entre política e raça; preservação de posições divergentes; e validação direta no texto-fonte. O procedimento não elimina a circularidade e os resultados qualitativos não estimam prevalência.
