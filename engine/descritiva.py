"""
Análise descritiva da base de entrada (apêndice do paper de validação).

Opera sobre a base ORIGINAL — a base analítica contrato a contrato, antes
da agregação, ou a carteira granular enviada —, para que o validador veja
os dados que sustentam a execução: completude e domínio das colunas,
distribuição das variáveis de risco, concentração das exposições e
composição por rating, setor e categoria regulatória.
"""
import numpy as np
import pandas as pd

NUMERICAS = [("exposicao", "Exposição (EAD)"), ("pd", "PD"),
             ("lgd", "LGD"), ("prazo_anos", "Prazo (anos)"),
             ("ltv", "LTV"), ("receita_bruta_mi", "Receita bruta (R$ mi)")]
CATEGORICAS = [("rating", "Rating"), ("setor", "Setor"),
               ("segmento_irb", "Categoria IRB")]
OBRIGATORIAS = ["id_contraparte", "exposicao", "pd"]
OPCIONAIS = ["lgd", "prazo_anos", "rating", "setor", "gh", "segmento_irb",
             "receita_bruta_mi", "ltv", "transactor", "ativo_problematico",
             "dias_atraso", "provisao", "categoria_if", "fpr"]
_Q = [0.01, 0.25, 0.50, 0.75, 0.99]


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return np.nan
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def descrever(df_in):
    """Estatísticas descritivas da base. Devolve um dicionário com
    escalares e DataFrames, pronto para o app e para o paper."""
    df = df_in.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    n = len(df)
    ead = pd.to_numeric(df["exposicao"], errors="coerce")
    pdv = pd.to_numeric(df["pd"], errors="coerce")
    lgd = (pd.to_numeric(df["lgd"], errors="coerce")
           if "lgd" in df.columns else pd.Series(1.0, index=df.index))
    sev = ead * lgd.fillna(1.0)
    el = pdv * sev
    tot = float(ead.sum())

    # ---- visão geral e qualidade ----
    geral = {
        "linhas": n,
        "contrapartes": int(df["id_contraparte"].nunique())
        if "id_contraparte" in df.columns else n,
        "ead_total": tot,
        "severidade_total": float(sev.sum()),
        "el_total": float(el.sum()),
        "el_pct": float(el.sum() / sev.sum()) if sev.sum() > 0 else np.nan,
        "pd_media_ponderada": float((pdv * sev).sum() / sev.sum()),
        "lgd_media_ponderada": float(sev.sum() / tot) if tot > 0 else np.nan,
        "ead_media": float(ead.mean()),
    }
    qual = []
    for c in OBRIGATORIAS + [c for c in OPCIONAIS if c in df.columns]:
        s = df[c]
        falt = int(s.isna().sum())
        obs = ""
        if c == "exposicao":
            obs = f"{int((ead <= 0).sum())} não positiva(s)"
        elif c == "pd":
            obs = f"{int(((pdv <= 0) | (pdv >= 1)).sum())} fora de (0, 1)"
        elif c == "lgd":
            obs = f"{int(((lgd <= 0) | (lgd > 1)).sum())} fora de (0, 1]"
        elif c == "id_contraparte":
            obs = f"{n - int(s.nunique())} repetido(s)"
        elif c == "ltv" and "segmento_irb" in df.columns:
            ap = df["segmento_irb"].astype(str).str.lower().isin(
                ["varejo_residencial", "hvcre"])
            if ap.any():
                obs = (f"{1 - s[ap].isna().mean():.1%} preenchido no "
                       "imobiliário").replace(".", ",")
        elif c == "receita_bruta_mi" and "segmento_irb" in df.columns:
            ap = df["segmento_irb"].astype(str).str.lower() == "sme"
            if ap.any():
                obs = (f"{1 - s[ap].isna().mean():.1%} preenchido nas "
                       "SME").replace(".", ",")
        qual.append({"coluna": c,
                     "tipo": "obrigatória" if c in OBRIGATORIAS else "opcional",
                     "preenchimento": 1.0 - falt / n if n else np.nan,
                     "faltantes": falt, "observacao": obs})
    ausentes = [c for c in OPCIONAIS if c not in df.columns]

    # ---- variáveis numéricas ----
    linhas = []
    for c, rot in NUMERICAS:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(x):
            continue
        q = x.quantile(_Q).to_numpy()
        linhas.append({"variavel": rot, "n": int(len(x)), "media": x.mean(),
                       "desvio": x.std(), "min": x.min(), "p01": q[0],
                       "p25": q[1], "mediana": q[2], "p75": q[3],
                       "p99": q[4], "max": x.max()})
    numericas = pd.DataFrame(linhas)

    # ---- concentração ----
    e = np.sort(ead.dropna().to_numpy())[::-1]
    w = e / e.sum()
    conc = {
        "hhi": float(np.square(w).sum()),
        "n_efetivo": float(1.0 / np.square(w).sum()),
        "top10": float(w[:10].sum()), "top100": float(w[:100].sum()),
        "top1pct": float(w[:max(1, int(np.ceil(0.01 * len(w))))].sum()),
        "gini": _gini(e),
    }
    # curva de Lorenz (subamostrada para o gráfico)
    asc = e[::-1]
    pts = np.unique(np.linspace(0, len(asc), min(len(asc), 400) + 1)
                    .astype(int))
    cum = np.concatenate([[0.0], np.cumsum(asc)]) / asc.sum()
    lorenz = (pts / len(asc), cum[pts])

    # ---- distribuições categóricas ----
    cats = {}
    for c, rot in CATEGORICAS:
        if c not in df.columns:
            continue
        g = pd.DataFrame({"cat": df[c].astype(str), "ead": ead, "sev": sev,
                          "el": el, "pdsev": pdv * sev}).groupby("cat")
        t = pd.DataFrame({"linhas": g.size(), "ead": g["ead"].sum(),
                          "sev": g["sev"].sum(), "el": g["el"].sum(),
                          "pdsev": g["pdsev"].sum()})
        t["ead_pct"] = t["ead"] / tot
        t["pd_media"] = t["pdsev"] / t["sev"]
        t["lgd_media"] = t["sev"] / t["ead"]
        t["el_pct"] = t["el"] / t["el"].sum()
        t = t.drop(columns=["sev", "pdsev"]).reset_index().rename(
            columns={"cat": c})
        if c == "rating":
            t = t.sort_values("pd_media")
        else:
            t = t.sort_values("ead", ascending=False)
        cats[c] = (rot, t)

    # PD por rating para o gráfico (amostra limitada)
    pd_por_rating = None
    if "rating" in df.columns:
        r = cats["rating"][1]["rating"].tolist()
        amostra = df[["rating"]].assign(pd=pdv).sample(
            min(n, 60_000), random_state=0) if n > 60_000 else \
            df[["rating"]].assign(pd=pdv)
        pd_por_rating = [(k, amostra.loc[amostra["rating"].astype(str) == k,
                                         "pd"].dropna().to_numpy())
                         for k in r]

    # matriz setor x rating (% da EAD)
    cruz = None
    if "setor" in df.columns and "rating" in df.columns:
        cruz = pd.crosstab(df["setor"].astype(str), df["rating"].astype(str),
                           values=ead, aggfunc="sum").fillna(0.0) / tot
        if "rating" in cats:
            cruz = cruz[[k for k in cats["rating"][1]["rating"]
                         if k in cruz.columns]]

    return {"geral": geral, "qualidade": pd.DataFrame(qual),
            "ausentes": ausentes, "numericas": numericas,
            "concentracao": conc, "lorenz": lorenz, "categoricas": cats,
            "pd_por_rating": pd_por_rating, "setor_rating": cruz,
            "ead_amostra": ead.dropna().sample(min(n, 100_000),
                                               random_state=0).to_numpy()}
