"""
Capital IRB segundo a Resolução BCB nº 303/2023 (abordagem IRB do Banco
Central do Brasil; revogou a Circular nº 3.648/2013; em vigor desde
1º/7/2023).

O requerimento por exposição, em fração da severidade c = EAD x LGD, é

    K = { N[ (N^-1(PD) + sqrt(R) N^-1(0,999)) / sqrt(1-R) ] - PD } x MA

em que N é a normal padrão acumulada, R a correlação da categoria e MA o
ajuste de maturidade, aplicado SOMENTE ao atacado e às instituições
financeiras (art. 39); o varejo não tem ajuste de maturidade (art. 46):

    b  = (0,11852 - 0,05478 ln PD)^2               (art. 39, VI)
    MA = (1 + (M - 2,5) b) / (1 - 1,5 b),  1 <= M <= 5 (art. 81)

Correlações por categoria/subcategoria (arts. 7º a 9º, 39 e 46), com
f(PD) = (1 - e^{-50 PD}) / (1 - e^{-50}) e g(PD) = (1 - e^{-35 PD}) / (1 - e^{-35}):

    corporate        demais atacado         0,12 f + 0,24 (1-f)
    sme              atacado SME            0,12 f + 0,24 (1-f)
                                            - 0,04 (1 - (S-15)/(X-15))
                     S = receita bruta anual em R$ milhões, S >= 15;
                     X = 300 (teto de receita do porte SME, art. 36
                     da Resolução BCB nº 229/2022)
    hvcre            imóveis comerciais de alta volatilidade
                                            0,12 f + 0,30 (1-f)
    if               instituição financeira 0,12 f + 0,24 (1-f)
    if_s1s2          IF dos segmentos S1/S2 1,25 x [0,12 f + 0,24 (1-f)]
    varejo_residencial                      0,15
    varejo_rotativo  rotativo qualificado   0,04
    varejo_demais    demais varejo          0,03 g + 0,16 (1-g)

Pisos de PD (art. 61): 0,05%; no rotativo qualificado, 0,10% quando a
exposição se enquadra no art. 47 da Resolução BCB nº 229 — na ausência
de identificação, aplica-se 0,10% a toda a subcategoria (conservador).
O RWA equivalente é K x EAD / F, com F = 8% (fator 12,5).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

SEGMENTOS = {
    "corporate": ("Atacado — demais", True),
    "sme": ("Atacado — SME", True),
    "hvcre": ("Atacado — HVCRE", True),
    "if": ("Instituição financeira", True),
    "if_s1s2": ("Instituição financeira S1/S2", True),
    "varejo_residencial": ("Varejo — residencial", False),
    "varejo_rotativo": ("Varejo — rotativo qualificado", False),
    "varejo_demais": ("Varejo — demais", False),
}
SEGMENTO_PADRAO = "corporate"
SME_S_MIN, SME_X = 15.0, 300.0          # R$ milhões
PD_PISO, PD_PISO_ROTATIVO = 0.0005, 0.0010
B0, B1 = 0.11852, 0.05478               # coeficientes oficiais (art. 39, VI)
F = 0.08                                # RWA = K x EAD / F


def _f(pd_, k):
    return (1.0 - np.exp(-k * pd_)) / (1.0 - np.exp(-k))


def correlacao(pd_, segmento, receita_mi=None):
    """Correlação R por exposição, segundo a categoria (vetorizado)."""
    pd_ = np.asarray(pd_, dtype=float)
    seg = np.asarray(segmento, dtype=object)
    f = _f(pd_, 50.0)
    base = 0.12 * f + 0.24 * (1.0 - f)
    R = base.copy()
    sme = seg == "sme"
    if sme.any():
        if receita_mi is None:
            S = np.full(pd_.shape, SME_X)          # sem receita: sem redução
        else:
            S = np.asarray(receita_mi, dtype=float).copy()
            S = np.where(np.isnan(S), SME_X, S)
        S = np.clip(S, SME_S_MIN, SME_X)
        R = np.where(sme, base - 0.04 * (1.0 - (S - SME_S_MIN)
                                         / (SME_X - SME_S_MIN)), R)
    R = np.where(seg == "hvcre", 0.12 * f + 0.30 * (1.0 - f), R)
    R = np.where(seg == "if_s1s2", 1.25 * base, R)
    R = np.where(seg == "varejo_residencial", 0.15, R)
    R = np.where(seg == "varejo_rotativo", 0.04, R)
    g = _f(pd_, 35.0)
    R = np.where(seg == "varejo_demais", 0.03 * g + 0.16 * (1.0 - g), R)
    return R


def pd_com_piso(pd_, segmento):
    seg = np.asarray(segmento, dtype=object)
    piso = np.where(seg == "varejo_rotativo", PD_PISO_ROTATIVO, PD_PISO)
    return np.maximum(np.asarray(pd_, dtype=float), piso)


def coeficiente_k(pd_, prazo, segmento, receita_mi=None, alpha=0.999):
    """K por unidade de severidade (EAD x LGD), com pisos de PD.
    Devolve (K, R, pd_usada)."""
    seg = np.asarray(segmento, dtype=object)
    p = pd_com_piso(pd_, seg)
    R = correlacao(p, seg, receita_mi)
    cond = norm.cdf((norm.ppf(p) + np.sqrt(R) * norm.ppf(alpha))
                    / np.sqrt(1.0 - R))
    b = np.square(B0 - B1 * np.log(p))
    M = np.clip(np.asarray(prazo, dtype=float), 1.0, 5.0)
    ma = (1.0 + (M - 2.5) * b) / (1.0 - 1.5 * b)
    atacado = np.array([SEGMENTOS[s][1] for s in seg])
    K = (cond - p) * np.where(atacado, ma, 1.0)
    return K, R, p


def segmentos_da_carteira(df):
    if "segmento_irb" in df.columns:
        return df["segmento_irb"].astype(str).to_numpy()
    return np.full(len(df), SEGMENTO_PADRAO, dtype=object)


def receita_da_carteira(df):
    if "receita_bruta_mi" in df.columns:
        return pd.to_numeric(df["receita_bruta_mi"],
                             errors="coerce").to_numpy(dtype=float)
    return None


def calcular(port, alpha=0.999):
    """Capital IRB Bacen da carteira: vetores por linha e quadro por
    segmento."""
    df = port.df
    seg = segmentos_da_carteira(df)
    rec = receita_da_carteira(df)
    K, R, p_usada = coeficiente_k(port.p, port.tenor, seg, rec, alpha)
    c, ead = port.c, port.ead
    cap = c * K
    linhas = []
    for s in [s for s in SEGMENTOS if s in set(seg)]:
        m = seg == s
        linhas.append({
            "segmento": s,
            "descricao": SEGMENTOS[s][0],
            "linhas": int(m.sum()),
            "ead": float(ead[m].sum()),
            "exposicao_liquida": float(c[m].sum()),
            "R_ponderado": float(np.dot(R[m], c[m]) / c[m].sum()),
            "ajuste_maturidade": SEGMENTOS[s][1],
            "capital": float(cap[m].sum()),
            "capital_pct": float(cap[m].sum() / c[m].sum()),
            "rwa": float(cap[m].sum() / F),
        })
    sme_sem_receita = int(((seg == "sme") & (
        np.isnan(rec) if rec is not None else True)).sum())
    return {
        "k": K, "R": R, "capital": cap,
        "por_segmento": pd.DataFrame(linhas),
        "n_pd_piso": int((p_usada > port.p).sum()),
        "sme_sem_receita": sme_sem_receita,
        "segmentado": "segmento_irb" in df.columns,
    }
