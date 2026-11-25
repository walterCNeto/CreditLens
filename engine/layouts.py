"""
Layouts dos arquivos de entrada do CreditLens: dicionário de colunas e
arquivos-modelo preenchidos (pasta templates/). Fonte única para o
aplicativo e para a documentação.

Convenções gerais: CSV separado por vírgula, decimal com ponto, UTF-8;
valores monetários em reais (R$), sem escala, exceto as colunas cujo nome
termina em `_mi` (R$ milhões); taxas e frações em decimal (5% = 0.05).
"""
import pandas as pd

SEGMENTOS_TXT = ("corporate, sme, hvcre, if, if_s1s2, varejo_residencial, "
                 "varejo_rotativo, varejo_demais")

# (coluna, obrigatória, formato/unidade, descrição)
CARTEIRA = [
    ("id_contraparte", "sim", "texto, único",
     "Identificador da contraparte (uma linha por contraparte)."),
    ("exposicao", "sim", "R$, > 0",
     "EAD: exposição no default, em reais, com fatores de conversão de "
     "crédito já aplicados."),
    ("pd", "sim", "fração em (0, 1)",
     "Probabilidade de default em 12 meses."),
    ("lgd", "não", "fração em (0, 1] — padrão 1,0",
     "Perda dado o default; severidade c = EAD × LGD."),
    ("prazo_anos", "não", "anos — padrão 2,5",
     "Prazo efetivo; truncado a [1, 5] no ajuste de maturidade do IRB."),
    ("rating", "não", "texto",
     "Grau de rating; obrigatório para migração/ECL e deve existir na "
     "matriz de transição."),
    ("setor", "não", "texto",
     "Setor econômico; habilita o modelo multifatorial."),
    ("nome", "não", "texto", "Nome da contraparte (apenas exibição)."),
    ("pais", "não", "texto",
     "País do risco; seletor de choques no estresse customizado (p.ex. "
     "transição climática por país)."),
    ("regiao", "não", "texto",
     "Região (p.ex. UF ou macrorregião); seletor de choques no estresse "
     "customizado (p.ex. risco climático físico)."),
    ("segmento_irb", "não", SEGMENTOS_TXT + " — padrão corporate",
     "Categoria IRB (Res. BCB 303) e base do fator padronizado "
     "(Res. BCB 229)."),
    ("receita_bruta_mi", "não", "R$ milhões",
     "Receita bruta anual: ajuste de porte da SME no IRB; PME (< 300) e "
     "pequena empresa do varejo (< 15) no padronizado."),
    ("ativo_total_mi", "não", "R$ milhões",
     "Ativo total: critério da PME (< 240) no padronizado (art. 36)."),
    ("grande_baixo_risco", "não", "0/1",
     "Grande empresa de baixo risco (art. 35: FPR 65%); critérios "
     "verificados pelo usuário."),
    ("ltv", "não", "fração (ex.: 0.65)",
     "Saldo / valor de avaliação do imóvel; obrigatório para o "
     "imobiliário no padronizado (sem LTV: faixa acima de 100%)."),
    ("imovel_dependente_fluxo", "não", "0/1",
     "Pagamento depende do fluxo do imóvel (arts. 51 e 53)."),
    ("transactor", "não", "0/1",
     "Rotativo sem atraso, parcelamento ou saque em 360 dias (art. 47: "
     "FPR 45%)."),
    ("ativo_problematico", "não", "0/1",
     "Ativo problemático (art. 66); alternativa: dias_atraso >= 90. "
     "Marca a exposição como em default."),
    ("dias_atraso", "não", "dias",
     "Dias de atraso. Acima de 30: estágio 2 do ECL; a partir de 90: "
     "exposição em default (tratamento próprio, fora dos modelos de "
     "simulação)."),
    ("meses_em_default", "não", "meses, >= 0",
     "Tempo desde o default; preenchido = exposição em default. Posiciona "
     "o contrato na curva de LGD em default. Ausente: estimado por "
     "(dias_atraso − 90) / 30,4."),
    ("provisao", "não", "R$",
     "Provisão da exposição: deduzida da EAD do padronizado e usada nas "
     "faixas do art. 66; nas exposições em default, comparada à ELBE."),
    ("categoria_if", "não", "A, B ou C — padrão A",
     "Categoria de risco da instituição financeira (arts. 29 a 33)."),
    ("prazo_original_dias", "não", "dias",
     "Prazo original (IF: até 90 dias tem fator reduzido)."),
    ("fpr", "não", "fração",
     "Fator de ponderação informado pelo usuário; prevalece sobre o "
     "calculado."),
]

ANALITICA_EXTRA = [
    ("id_contraparte", "sim", "texto (pode repetir)",
     "Identificador do cliente; contratos do mesmo cliente compartilham o "
     "id — base dos limites de varejo do art. 46."),
    ("id_contrato", "não", "texto",
     "Identificador do contrato; usado quando um cliente tem mais de um "
     "contrato entre as grandes exposições."),
    ("gh", "não", "texto — padrão: rating",
     "Grupo homogêneo de risco usado na agregação (GH × setor × "
     "categoria IRB, e país e região quando escolhidos para preservação)."),
]

MATRIZ = [
    ("1ª coluna (sem título)", "sim", "texto",
     "Rating de origem, um por linha, do melhor ao pior; a última linha é "
     "o estado de default."),
    ("demais colunas", "sim", "fração em [0, 1]",
     "Probabilidade de migrar, em um ano, do rating da linha para o "
     "rating da coluna; cabeçalho com os mesmos ratings, na mesma ordem "
     "das linhas."),
    ("soma de cada linha", "regra", "= 1",
     "Cada linha soma 1 (tolerância de 0,001; pequenos resíduos são "
     "renormalizados)."),
    ("estado de default", "regra", "última linha e coluna",
     "Absorvente: a última linha é [0, …, 0, 1]."),
    ("ratings", "regra", "mesmos rótulos da carteira",
     "Todo rating da coluna `rating` da carteira precisa existir na "
     "matriz."),
]


CURVA = [
    ("segmento_irb", "não", "categoria IRB ou 'todos'",
     "Segmento a que a curva se aplica; 'todos' (ou vazio) vale para os "
     "segmentos sem curva própria."),
    ("meses", "sim", "meses em default, >= 0",
     "Pontos da curva; entre pontos, interpolação linear; após o último, "
     "vale o último valor."),
    ("lgd_elbe", "sim", "fração em [0, 1], não decrescente",
     "ELBE: melhor estimativa da perda dado o tempo em default; o último "
     "ponto (100%) marca o prazo de baixa (write-off)."),
]


CENARIOS = [
    ("cenario", "sim", "texto", "Nome do cenário ou da sensibilidade; as "
     "linhas com o mesmo nome formam um cenário. O nome 'Base' é a "
     "referência das variações."),
    ("tipo", "não", "'cenario' ou 'sensibilidade'", "Cenário plurianual "
     "(análise de cenários) ou choque de um parâmetro em 1 ano (análise de "
     "sensibilidade). Vazio = 'cenario'."),
    ("ano", "sim", "inteiro de 1 a 5", "Ano da projeção."),
    ("setor", "não", "nome do setor ou 'todos'", "Setor da carteira a que "
     "o choque se aplica; 'todos' vale para a carteira inteira."),
    ("z", "não", "número; negativo = adverso", "Índice de crédito do ano: "
     "PD(z) = Φ(Φ⁻¹(PD) − z·√(ρ/(1 − ρ))); z = 0 reproduz a PD de 12 meses."),
    ("mult_pd", "não", "multiplicador > 0", "Multiplica a PD depois do "
     "choque em z (padrão 1)."),
    ("choque_lgd_pp", "não", "pontos percentuais", "Somado à LGD da "
     "carteira não default e à LGD de entrada do estoque em default."),
    ("mult_k_default", "não", "multiplicador > 0", "Multiplica a margem k "
     "da LGD em default (recuperação mais incerta)."),
    ("pib_pct, desemprego_pct, selic_pct, ipca_pct", "não", "percentuais",
     "Narrativa macroeconômica do cenário; documentação, sem uso no "
     "cálculo."),
    ("descricao", "não", "texto", "Descrição do cenário."),
]


CUSTOMIZADO = [
    ("cenario", "sim", "texto", "Nome do cenário; as linhas com o mesmo "
     "nome formam um cenário."),
    ("modo", "não", "'instantaneo' ou 'projecao'", "Instantâneo: cada "
     "horizonte é um choque sobre a carteira atual (convém a horizontes "
     "longos, p.ex. 2030 e 2050). Projeção: anos 1 a 5 em sequência, com "
     "balanço constante. Vazio = 'instantaneo'."),
    ("ano", "sim", "inteiro", "Horizonte (p.ex. 2030) no modo instantâneo; "
     "ano de 1 a 5 no modo de projeção."),
    ("setor, pais, regiao, segmento_irb, rating", "não",
     "valor da coluna ou 'todos'", "Seletores: a linha atinge as exposições "
     "com esses valores; vazio ou 'todos' = qualquer valor. Quando duas "
     "linhas atingem a mesma exposição, prevalece a mais específica, que "
     "herda os campos vazios."),
    ("z", "não", "número; negativo = adverso", "Índice de crédito: "
     "PD(z) = Φ(Φ⁻¹(PD) − z·√(ρ/(1 − ρ)))."),
    ("mult_pd", "não", "multiplicador > 0", "Multiplica a PD (canal "
     "típico do risco de transição)."),
    ("choque_lgd_pp", "não", "pontos percentuais", "Somado à LGD (canal "
     "típico do risco físico: dano às garantias)."),
    ("choque_ead_pct", "não", "percentual", "Variação da EAD (p.ex. saque "
     "de limites sob estresse)."),
    ("mult_k_default", "não", "multiplicador > 0", "Multiplica a margem k "
     "da LGD em default."),
    ("objetivo, fonte, metodo_depara, premissas, descricao", "não", "texto",
     "Documentação do exercício; vai para o paper. Basta preencher na "
     "primeira linha de cada cenário."),
]


def dicionario(itens):
    return pd.DataFrame(itens, columns=["Coluna", "Obrigatória",
                                        "Formato / unidade", "Descrição"])


def dicionario_analitica():
    obrig = [c for c in CARTEIRA if c[0] not in ("id_contraparte", "nome")]
    return dicionario(ANALITICA_EXTRA + obrig)
