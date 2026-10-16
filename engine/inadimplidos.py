"""
Exposições em default (inadimplidas) na carteira ativa, sem write-off.

Tratamento determinístico, em linha com a abordagem IRB para exposições
inadimplidas (Res. BCB 303/2023; Basileia, CRE 36):

  * ELBE(t) — melhor estimativa da perda dado o tempo em default t (meses):
    a curva de LGD em default. Sem curva informada, usa-se a proxy linear
        ELBE(t) = LGD_0 + (1 − LGD_0) · min(t / T_baixa, 1),
    em que LGD_0 é a LGD da carteira não default do próprio contrato e
    T_baixa o prazo até a baixa (write-off), quando a perda chega a 100%.
  * LGD_D(t) — LGD em default com margem para a incerteza da recuperação.
    Proxy: LGD_D = ELBE + k · (1 − ELBE), que se anula na baixa.
  * Capital: K = max(0; LGD_D − ELBE) = k · (1 − ELBE); capital = EAD · K
    e RWA = 12,5 · capital. A perda esperada (ELBE) é coberta pela provisão
    de estágio 3, não por capital.
  * Perda em um ano: incremento determinístico da provisão,
        ΔL = EAD · [ELBE(t + 12) − ELBE(t)],
    que se soma à perda de qualquer modelo da carteira não default (é
    constante, não altera a perda inesperada). O estoque já provisionado é
    perda passada e não entra.

Contratos em atraso que não atingiram o default seguem na carteira não
default, com a LGD não default.

Critério de default (qualquer um): `meses_em_default` preenchido,
`dias_atraso` >= 90, `ativo_problematico` = 1 ou `pd` >= 1. Sem
`meses_em_default`, o tempo em default é estimado por
(dias_atraso − 90) / 30,4375.
"""
import numpy as np
import pandas as pd

MES_DIAS = 30.4375


def _num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    return np.full(len(df), np.nan)


def mascara_default(df):
    """Booleano por linha: exposição em default."""
    meses = _num(df, "meses_em_default")
    atraso = _num(df, "dias_atraso")
    probl = _num(df, "ativo_problematico")
    p = _num(df, "pd")
    return (~np.isnan(meses) & (meses >= 0)) | (np.nan_to_num(atraso) >= 90) \
        | (np.nan_to_num(probl) > 0.5) | (np.nan_to_num(p) >= 1.0)


def meses_em_default(df):
    meses = _num(df, "meses_em_default")
    atraso = _num(df, "dias_atraso")
    est = np.maximum(np.nan_to_num(atraso) - 90.0, 0.0) / MES_DIAS
    return np.where(np.isnan(meses), est, np.maximum(meses, 0.0))


def separar(df):
    """(df_performing, df_default). O índice original é preservado."""
    m = mascara_default(df)
    return df.loc[~m].copy(), df.loc[m].copy()


# ---------------------------------------------------------------- curvas
def ler_curva(fonte):
    """Curva de LGD em default: colunas `meses`, `lgd_elbe` e, opcional,
    `segmento_irb` (vazio ou 'todos' = vale para todos os segmentos)."""
    c = pd.read_csv(fonte)
    c.columns = [x.strip().lower() for x in c.columns]
    falta = {"meses", "lgd_elbe"} - set(c.columns)
    if falta:
        raise ValueError(f"Curva de LGD sem as colunas: {sorted(falta)}.")
    if "segmento_irb" not in c.columns:
        c["segmento_irb"] = "todos"
    c["segmento_irb"] = (c["segmento_irb"].fillna("todos").astype(str)
                         .str.strip().str.lower().replace("", "todos"))
    c["meses"] = pd.to_numeric(c["meses"], errors="coerce")
    c["lgd_elbe"] = pd.to_numeric(c["lgd_elbe"], errors="coerce")
    c = c.dropna(subset=["meses", "lgd_elbe"]).sort_values(
        ["segmento_irb", "meses"])
    if ((c["lgd_elbe"] < 0) | (c["lgd_elbe"] > 1)).any():
        raise ValueError("A curva de LGD deve estar em [0, 1].")
    for s, g in c.groupby("segmento_irb"):
        if (np.diff(g["lgd_elbe"].to_numpy()) < -1e-12).any():
            raise ValueError(f"A curva de LGD do segmento '{s}' deve ser "
                             "não decrescente no tempo em default.")
    return c.reset_index(drop=True)


def elbe(t, lgd0, seg, curva=None, prazo_baixa=24.0):
    """ELBE para tempos t (meses), vetorizado por contrato."""
    t = np.asarray(t, float)
    lgd0 = np.asarray(lgd0, float)
    if curva is None:
        return lgd0 + (1.0 - lgd0) * np.clip(t / prazo_baixa, 0.0, 1.0)
    out = np.full(t.shape, np.nan)
    seg = np.asarray(seg).astype(str)
    segs_curva = set(curva["segmento_irb"])
    geral = curva[curva["segmento_irb"] == "todos"]
    for s in np.unique(seg):
        m = seg == s
        g = curva[curva["segmento_irb"] == s] if s in segs_curva else geral
        if g.empty:
            out[m] = lgd0[m] + (1.0 - lgd0[m]) * np.clip(
                t[m] / prazo_baixa, 0.0, 1.0)
            continue
        out[m] = np.interp(t[m], g["meses"].to_numpy(),
                           g["lgd_elbe"].to_numpy())
    return out


# ---------------------------------------------------------------- avaliação
def avaliar(df_def, curva=None, prazo_baixa=24.0, k=0.10, escala_reais=1.0,
            horizonte=5, taxa_desc=0.0):
    """Provisão (ELBE), capital (LGD_D − ELBE), RWA IRB e padronizado, e
    perdas incrementais em 1 ano e no horizonte multiperíodo."""
    if df_def is None or len(df_def) == 0:
        return None
    d = df_def.copy()
    n = len(d)
    ead = pd.to_numeric(d["exposicao"], errors="coerce").to_numpy(float)
    lgd0 = (pd.to_numeric(d["lgd"], errors="coerce").fillna(1.0).to_numpy(float)
            if "lgd" in d.columns else np.ones(n))
    seg = (d["segmento_irb"].astype(str).str.strip().str.lower().to_numpy()
           if "segmento_irb" in d.columns else np.full(n, "corporate"))
    t = meses_em_default(d)
    e0 = elbe(t, lgd0, seg, curva, prazo_baixa)
    e12 = elbe(t + 12.0, lgd0, seg, curva, prazo_baixa)
    lgd_d = e0 + k * (1.0 - e0)
    K = np.maximum(lgd_d - e0, 0.0)
    prov_inf = _num(d, "provisao")

    # padronizado (Res. BCB 229, art. 66): a exposição é ativo problemático
    from .padronizado_bcb import fpr_por_exposicao
    d_cp = d.copy()
    d_cp["ativo_problematico"] = 1
    fpr, art, ead_cpad, _ = fpr_por_exposicao(d_cp, escala_reais)

    tab = pd.DataFrame({
        "id_contraparte": d["id_contraparte"].astype(str).to_numpy()
        if "id_contraparte" in d.columns else np.arange(n).astype(str),
        "segmento_irb": seg, "exposicao": ead, "lgd_nao_default": lgd0,
        "meses_em_default": t, "elbe": e0, "elbe_12m": e12, "lgd_d": lgd_d,
        "K": K, "provisao_elbe": ead * e0, "provisao_informada": prov_inf,
        "perda_incremental_12m": ead * (e12 - e0), "capital_irb": ead * K,
        "rwa_irb": 12.5 * ead * K, "fpr_cpad": fpr, "artigo_cpad": art,
        "ead_cpad": ead_cpad, "rwa_cpad": ead_cpad * fpr,
    })
    if "id_contrato" in d.columns:
        tab.insert(1, "id_contrato", d["id_contrato"].astype(str).to_numpy())

    # perdas incrementais por ano do horizonte multiperíodo (determinísticas)
    inc = np.array([float((ead * (elbe(t + 12.0 * y, lgd0, seg, curva,
                                        prazo_baixa)
                                   - elbe(t + 12.0 * (y - 1), lgd0, seg,
                                          curva, prazo_baixa))).sum())
                    for y in range(1, int(horizonte) + 1)])
    disc = (1.0 + taxa_desc) ** -np.arange(1, len(inc) + 1)

    tot = lambda c: float(tab[c].sum())
    prov_tot = float(np.nansum(prov_inf)) if not np.all(np.isnan(prov_inf)) \
        else np.nan
    w = ead / max(ead.sum(), 1e-12)
    resumo = {
        "contratos": n, "ead": float(ead.sum()),
        "meses_medio": float((w * t).sum()),
        "elbe_media": float((w * e0).sum()),
        "lgd_d_media": float((w * lgd_d).sum()),
        "provisao_elbe": tot("provisao_elbe"),
        "provisao_informada": prov_tot,
        "insuficiencia_provisao": float(np.nansum(
            np.maximum(ead * e0 - np.nan_to_num(prov_inf, nan=0.0), 0.0)))
        if not np.isnan(prov_tot) else np.nan,
        "perda_incremental_12m": tot("perda_incremental_12m"),
        "capital_irb": tot("capital_irb"), "rwa_irb": tot("rwa_irb"),
        "rwa_cpad": tot("rwa_cpad"),
        "perda_incremental_horizonte": float((inc * disc).sum()),
        "incremento_por_ano": inc,
        "curva": "informada" if curva is not None else "proxy linear",
        "prazo_baixa": float(prazo_baixa), "k": float(k),
        "ja_baixados": int((e0 >= 1.0 - 1e-12).sum()),
    }
    por_seg = tab.groupby("segmento_irb").agg(
        contratos=("exposicao", "size"), ead=("exposicao", "sum"),
        provisao_elbe=("provisao_elbe", "sum"),
        perda_incremental_12m=("perda_incremental_12m", "sum"),
        capital_irb=("capital_irb", "sum"), rwa_irb=("rwa_irb", "sum"),
        rwa_cpad=("rwa_cpad", "sum")).reset_index()
    por_seg["elbe_media"] = por_seg["provisao_elbe"] / por_seg["ead"]
    por_faixa = _por_faixa(tab)
    return {"tabela": tab, "resumo": resumo, "por_segmento": por_seg,
            "por_faixa": por_faixa,
            "curva_ilustrativa": curva_ilustrativa(
                float((w * lgd0).sum()), curva, prazo_baixa, k)}


def _por_faixa(tab):
    faixas = [0, 3, 6, 12, 18, 24, 36, np.inf]
    rot = ["0–3", "3–6", "6–12", "12–18", "18–24", "24–36", "36+"]
    f = pd.cut(tab["meses_em_default"], faixas, right=False, labels=rot)
    g = tab.groupby(f, observed=True).agg(
        contratos=("exposicao", "size"), ead=("exposicao", "sum"),
        provisao_elbe=("provisao_elbe", "sum"),
        capital_irb=("capital_irb", "sum")).reset_index()
    g = g.rename(columns={"meses_em_default": "meses"})
    g["elbe"] = g["provisao_elbe"] / g["ead"]
    g["K"] = g["capital_irb"] / g["ead"]
    return g


def curva_ilustrativa(lgd0, curva, prazo_baixa, k, ate=None):
    """ELBE, LGD_D e K ao longo do tempo em default (para gráficos)."""
    if curva is not None:
        ate = ate or float(curva["meses"].max()) * 1.15
        g = curva[curva["segmento_irb"] == "todos"]
        if g.empty:
            g = curva[curva["segmento_irb"] == curva["segmento_irb"].iloc[0]]
        tt = np.linspace(0, ate, 121)
        e = np.interp(tt, g["meses"], g["lgd_elbe"])
    else:
        ate = ate or prazo_baixa * 1.25
        tt = np.linspace(0, ate, 121)
        e = lgd0 + (1 - lgd0) * np.clip(tt / prazo_baixa, 0, 1)
    ld = e + k * (1 - e)
    return pd.DataFrame({"meses": tt, "elbe": e, "lgd_d": ld, "K": ld - e})


# ---------------------------------------------------------------- integração
def integrar(results, aval):
    """Acrescenta as exposições em default ao capital padronizado e ao piso
    de saída (mesmas exposições nos dois lados: art. 34, Res. BCB 303). Os
    modelos de simulação e o IRB da carteira não default não mudam."""
    if aval is None:
        return results
    from .padronizado_bcb import piso_saida, F, ARTIGOS
    r = aval["resumo"]
    results["inadimplidos"] = aval
    cp = results.get("padronizado")
    if not cp:
        return results
    cp["rwa_total_nao_default"] = cp["rwa_total"]
    cp["rwa_total"] = cp["rwa_total"] + r["rwa_cpad"]
    cp["capital"] = cp["rwa_total"] * F
    tab = aval["tabela"]
    pa_d = tab.groupby("artigo_cpad").agg(
        linhas=("exposicao", "size"), ead=("ead_cpad", "sum"),
        rwa=("rwa_cpad", "sum")).reset_index().rename(
        columns={"artigo_cpad": "artigo"})
    if cp.get("por_artigo") is not None:
        pa = pd.concat([cp["por_artigo"][["artigo", "linhas", "ead", "rwa"]],
                        pa_d]).groupby("artigo", as_index=False).sum()
        pa["fpr_medio"] = pa["rwa"] / pa["ead"].where(pa["ead"] > 0)
        pa["descricao"] = pa["artigo"].map(lambda a: ARTIGOS.get(a, (a,))[0])
        cp["por_artigo"] = pa.sort_values("rwa", ascending=False)
    rwa_irb_nd = cp["piso"]["rwa_irb"]
    cp["piso"] = piso_saida(rwa_irb_nd + r["rwa_irb"], cp["rwa_total"])
    cp["piso"]["rwa_irb_nao_default"] = rwa_irb_nd
    cp["piso"]["rwa_irb_default"] = r["rwa_irb"]
    comp = cp.get("comparacao")
    linha = pd.DataFrame([{
        "segmento": "default", "descricao": "Exposições em default",
        "ead_irb": r["ead"], "rwa_irb": r["rwa_irb"], "ead": r["ead"],
        "rwa_cpad": r["rwa_cpad"],
        "razao_irb_cpad": r["rwa_irb"] / r["rwa_cpad"]
        if r["rwa_cpad"] > 0 else np.nan}])
    cp["comparacao"] = linha if comp is None else pd.concat(
        [comp, linha[[c for c in linha.columns if c in comp.columns]]],
        ignore_index=True)
    return results


def ecl_estagio3(aval):
    """Linhas de ECL das exposições em default (estágio 3: provisão =
    EAD × ELBE, a perda esperada já condicional ao default)."""
    t = aval["tabela"]
    return pd.DataFrame({
        "id_contraparte": t["id_contraparte"], "rating": "Default",
        "exposicao": t["exposicao"], "pd": 1.0, "lgd": t["elbe"],
        "prazo_anos": np.nan, "pd_matriz_1a": 1.0,
        "ECL_12m": t["provisao_elbe"], "ECL_lifetime": t["provisao_elbe"],
        "razao_lifetime_12m": 1.0, "estagio": 3,
        "provisao": t["provisao_elbe"]})


def ajustar_multiperiodo(sim, aval, taxa_desc=0.0):
    """Soma às perdas simuladas o incremento determinístico das exposições
    já em default ao longo do horizonte (desloca EL, VaR e ES)."""
    if aval is None or sim is None:
        return sim
    inc = aval["resumo"]["incremento_por_ano"][: sim["T"]]
    disc = (1.0 + taxa_desc) ** -np.arange(1, len(inc) + 1)
    d = float((inc * disc).sum())
    out = dict(sim)
    out["perdas"] = np.asarray(sim["perdas"]) + d
    out["EL"] = sim["EL"] + d
    out["VaR"] = {a: v + d for a, v in sim["VaR"].items()}
    out["ES"] = {a: v + d for a, v in sim["ES"].items()}
    out["incremento_default"] = d
    return out
