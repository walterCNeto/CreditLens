"""Alocação de capital por modelo: tabela-resumo do apêndice B.

Para cada modelo de perda, o capital econômico em cada nível de confiança é
a perda inesperada, VaR − EL e ES − EL, somada ao capital determinístico
das exposições em default (EAD · (LGD_D − ELBE)), quando houver. As linhas
de referência trazem o IRB (K*, com e sem ajuste de granularidade) e o
capital regulatório final, após o piso de saída, que só existem a 99,9%.
"""
import numpy as np
import pandas as pd

ALPHAS = ["95.00%", "99.00%", "99.90%", "99.97%"]
ORDEM = "ES 99.90%"          # coluna que ordena os modelos


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def tabela_alocacao(results):
    """DataFrame com colunas modelo, tipo ('modelo' ou 'referencia') e
    VaR/ES − EL em cada nível (em reais), e a EAD total usada no percentual.
    Retorna (df_reais, df_pct, ead_total, k_default)."""
    med = results.get("medidas")
    if med is None or not len(med):
        return None, None, np.nan, 0.0
    ad = results.get("inadimplidos") or {}
    rd = ad.get("resumo") or {}
    k_def = _f(rd.get("capital_irb")) if rd else 0.0
    k_def = 0.0 if k_def != k_def else k_def
    ead_def = _f(rd.get("ead")) if rd else 0.0
    ead_def = 0.0 if ead_def != ead_def else ead_def
    ead = _f(results.get("resumo", {}).get("ead_total")) + ead_def

    cols = [f"{m} {a}" for m in ("VaR", "ES") for a in ALPHAS]
    linhas = []
    for r in med.itertuples(index=False):
        nome = r[0]
        if str(nome).startswith("IRB"):
            continue
        d = dict(zip(med.columns, r))
        el = _f(d.get("EL"))
        lin = {"modelo": nome, "tipo": "modelo"}
        for c in cols:
            v = _f(d.get(c))
            lin[c] = v - el + k_def if v == v and el == el else np.nan
        linhas.append(lin)
    # do maior para o menor capital pelo ES 99,9% − EL (sem ES: ao final)
    linhas.sort(key=lambda l: -l[ORDEM] if l[ORDEM] == l[ORDEM] else np.inf)

    def ref(nome, valor):
        lin = {"modelo": nome, "tipo": "referencia"}
        lin.update({c: np.nan for c in cols})
        lin["VaR 99.90%"] = valor
        linhas.append(lin)

    irb = results.get("irb") or {}
    suf = " + default" if k_def else ""
    if irb.get("capital_total_999") is not None:
        ref("IRB, K* (Res. BCB 303)" + suf,
            _f(irb["capital_total_999"]) + k_def)
    if irb.get("capital_com_ga") is not None:
        ref("IRB com ajuste de granularidade" + suf,
            _f(irb["capital_com_ga"]) + k_def)
    ps = (results.get("padronizado") or {}).get("piso")
    if ps and ps.get("capital_final") is not None:
        ref("Regulatório final, com piso de saída (8% do RWA)",
            _f(ps["capital_final"]))
    df = pd.DataFrame(linhas, columns=["modelo", "tipo"] + cols)
    pct = df.copy()
    if ead and ead == ead:
        pct[cols] = df[cols] / ead
    return df, pct, ead, k_def
