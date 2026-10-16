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
     "Ativo problemático (art. 66); alternativa: dias_atraso >= 90."),
    ("dias_atraso", "não", "dias", "Dias de atraso."),
    ("provisao", "não", "R$",
     "Provisão da exposição: deduzida da EAD do padronizado e usada nas "
     "faixas do art. 66."),
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
     "categoria IRB)."),
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


def dicionario(itens):
    return pd.DataFrame(itens, columns=["Coluna", "Obrigatória",
                                        "Formato / unidade", "Descrição"])


def dicionario_analitica():
    obrig = [c for c in CARTEIRA if c[0] not in ("id_contraparte", "nome")]
    return dicionario(ANALITICA_EXTRA + obrig)
