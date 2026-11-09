# Regras lexicais dos cinco eixos

As expressões abaixo são transcritas da configuração usada na análise. As categorias são não excludentes e funcionam como localizadores de candidatos.

```python
EIXOS = {
    'Biográfico-teórico': re.compile(
        r'(teoria.*evolu[cç][aã]o|sele[cç][aã]o natural|origem.*esp[eé]cies|'
        r'evolu[cç][aã]o.*esp[eé]cies|evolu[cç][aã]o darwiniana|'
        r'charles.*darwin|darwin.*evolu[cç][aã]o|esp[eé]cies.*darwin)',
        re.I
    ),
    'Darwin Awards': re.compile(
        r'(darwin award|pr[eê]mio darwin|trof[eé]u.*darwin|trof[eé]u.*charles|'
        r'awards.*pgm|forte candidato|candidato.*award|darwin awards|'
        r'award news|concorrente.*award)',
        re.I
    ),
    'Criacionista/pseudocientífico': re.compile(
        r'(design inteligente|intelligent design|t[uú]mulo.*londres|'
        r'abadia.*westminster|mentira.*charles|charles.*mentira|'
        r'estava errado|darwin.*mentira|evolu[cç][aã]o.*fake|'
        r'big bang|elo perdido|darwin.*arrependeu|leito.*morte|'
        r'darwin.*deus|terra.*10\.?000|dinossauro.*n[aã]o|'
        r'esoterismo.*gn[oó]stico|criacionismo|criacionista)',
        re.I
    ),
    'Extrema direita/antissemitismo': re.compile(
        r'(dailystormer|daily stormer|soap merchant|'
        r'ra[cç]a superior|ra[cç]a.*inferior|veio.*macaco|viemos.*macaco|'
        r'charles.*judeu|darwin.*judeu|judeu.*darwin|'
        r'hitler.*darwin|darwin.*hitler|nazi.*darwin|darwin.*nazi|'
        r'nazismo.*darwin|darwin.*nazismo|'
        r'darwinismo social|social darwinism|darwinismo racial)',
        re.I
    ),
    'Marx-Darwin': re.compile(
        r'(karl marx|marx.*darwin|darwin.*marx|marxismo.*darwin|darwin.*marxismo|'
        r'marx.*stalin|stalin.*darwin|darwin.*stalin|'
        r'darwinismo.*comunis|comunis.*darwin)',
        re.I
    ),
}
```
