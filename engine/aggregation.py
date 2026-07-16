"""
Camada de agregação do CreditLens: da base analítica à carteira de modelagem.

Recebe uma base analítica (contrato a contrato, potencialmente milhões de
linhas, no mesmo padrão de colunas do CreditLens) e produz uma carteira
compacta em duas partes:

1. GRANDES EXPOSIÇÕES: contratos acima de um corte de materialidade (por
   valor e/ou top-K) passam individualmente, preservando a informação de
   concentração de nomes que domina a cauda.

2. GRUPOS HOMOGÊNEOS (GH): o resíduo pulverizado é agregado por
   GH x setor (o GH pode vir de uma coluna `gh` do usuário; na ausência,
   usa-se o rating como proxy de banda de risco), com parâmetros
   ponderados de forma a preservar a perda esperada EXATAMENTE:

       C_g   = soma(EAD_i x LGD_i)                (severidade do bucket)
       pd_g  = soma(pd_i x EAD_i x LGD_i) / C_g   (PD ponderada por severidade)
       lgd_g = C_g / soma(EAD_i)
       prazo_g = ponderado por severidade
       n_g   = número de contratos do bucket

   Com esses pesos, EL_g = pd_g x C_g = soma(pd_i x EAD_i x LGD_i): a EL
   da carteira agregada bate ao centavo com a analítica (teste de sanidade
   incluído no relatório).

A coluna nova `n_contratos` informa aos modelos que a linha é um pool:
a simulação usa Binomial(n_g, p condicional) para pools moderados e a
aproximação de carteira grande (perda condicional = C_g x p condicional,
lei dos grandes números de Vasicek) para pools grandes — o que reduz a
dimensão da simulação ao número de buckets e torna viável a base de
milhões de linhas.

O relatório de qualidade reporta a dispersão de PD dentro de cada GH
(viés de Jensen potencial nas funções não lineares de capital): buckets
com coeficiente de variação alto sugerem refinar a formação dos GHs.
"""
import numpy as np
import pandas as pd

from .portfolio import Portfolio, load_portfolio, REQUIRED_COLS


def aggregate_portfolio(df, corte_grandes: float | None = None,
                        max_grandes: int = 200,
                        col_gh: str = "gh"):
    """Agrega a base analítica em grandes exposições + buckets GH x setor.

    Parâmetros: `corte_grandes` (contratos com exposicao >= corte passam
    individualmente; se None, usa apenas o top `max_grandes`), `max_grandes`
    (teto de nomes individuais) e `col_gh` (coluna de grupo homogêneo; se
    ausente, usa `rating`).

    Devolve (df_agregado, relatorio)."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    faltam = [c for c in REQUIRED_COLS if c not in df.columns]
    if faltam:
        raise ValueError(f"Base analítica sem colunas obrigatórias: {faltam}")
    for c, dflt in [("lgd", 1.0), ("prazo_anos", 2.5),
                    ("rating", "NA"), ("setor", "GERAL")]:
        if c not in df.columns:
            df[c] = dflt
    gh_col = col_gh if col_gh in df.columns else "rating"

    df["severidade"] = df["exposicao"] * df["lgd"]
    el_analitica = float((df["pd"] * df["severidade"]).sum())

    # ---- corte de grandes exposições ----
    grandes_mask = pd.Series(False, index=df.index)
    if corte_grandes is not None:
        grandes_mask |= df["exposicao"] >= corte_grandes
    if grandes_mask.sum() > max_grandes or corte_grandes is None:
        top_idx = df["exposicao"].nlargest(max_grandes).index
        grandes_mask = df.index.isin(top_idx) if corte_grandes is None \
            else grandes_mask & df.index.isin(
                df.loc[grandes_mask, "exposicao"].nlargest(max_grandes).index)
    grandes = df[grandes_mask].copy()
    resto = df[~grandes_mask]

    # ---- agregação GH x setor com pesos de severidade ----
    g = resto.groupby([gh_col, "setor"], observed=True)
    agg = pd.DataFrame({
        "exposicao": g["exposicao"].sum(),
        "severidade": g["severidade"].sum(),
        "pd": g.apply(lambda x: np.average(x["pd"], weights=x["severidade"]),
                      include_groups=False),
        "prazo_anos": g.apply(lambda x: np.average(x["prazo_anos"],
                                                   weights=x["severidade"]),
                              include_groups=False),
        "n_contratos": g.size(),
        "pd_cv": g.apply(lambda x: (np.std(x["pd"]) / np.mean(x["pd"])
                                    if np.mean(x["pd"]) > 0 else 0.0),
                         include_groups=False),
    }).reset_index()
    agg["lgd"] = agg["severidade"] / agg["exposicao"]
    agg["rating"] = agg[gh_col].astype(str)
    agg["id_contraparte"] = ("GH_" + agg[gh_col].astype(str) + "_"
                             + agg["setor"].astype(str))
    agg["nome"] = ("Pool " + agg[gh_col].astype(str) + " / "
                   + agg["setor"].astype(str))

    grandes["n_contratos"] = 1
    grandes["pd_cv"] = 0.0
    if "nome" not in grandes.columns:
        grandes["nome"] = grandes["id_contraparte"]
    cols = ["id_contraparte", "nome", "rating", "setor", "exposicao",
            "pd", "lgd", "prazo_anos", "n_contratos", "pd_cv"]
    if "rating_origem" in grandes.columns:
        cols.append("rating_origem")
        agg["rating_origem"] = agg["rating"]
    out = pd.concat([grandes[cols], agg[cols]], ignore_index=True)

    el_agregada = float((out["pd"] * out["exposicao"] * out["lgd"]).sum())
    relatorio = {
        "linhas_analitica": int(len(df)),
        "linhas_agregada": int(len(out)),
        "n_grandes": int(len(grandes)),
        "n_buckets": int(len(agg)),
        "exposicao_grandes_pct": float(grandes["exposicao"].sum()
                                       / df["exposicao"].sum()),
        "el_analitica": el_analitica,
        "el_agregada": el_agregada,
        "el_diferenca_pct": abs(el_agregada / el_analitica - 1),
        "pd_cv_max_bucket": float(agg["pd_cv"].max()) if len(agg) else 0.0,
        "pd_cv_medio_bucket": float(agg["pd_cv"].mean()) if len(agg) else 0.0,
        "buckets_cv_alto": int((agg["pd_cv"] > 0.5).sum()),
    }
    return out, relatorio


def load_analytic_and_aggregate(path_or_buffer, **kwargs):
    """Conveniência: lê CSV analítico, agrega e devolve (Portfolio, relatório)."""
    df = pd.read_csv(path_or_buffer)
    out, rel = aggregate_portfolio(df, **kwargs)
    # valida via pipeline padrão (reusa piso de PD, truncamento de prazo etc.)
    import io
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    port = load_portfolio(buf)
    return port, rel
