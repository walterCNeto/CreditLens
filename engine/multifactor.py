"""
Extensão multifatorial setorial do CreditLens.

Modelo de limiar com dois níveis de fatores, na estrutura de correlação
do próprio Bolder (cap. 4, buildAssetCorrelationMatrix, com carga
setorial uniforme):

    Y_n = sqrt(rho_G) * G + sqrt(rho_S - rho_G) * F_{s(n)} + sqrt(1 - rho_S) * eps_n

em que G é o fator global, F_s o fator do setor s(n) da contraparte n,
eps_n o choque idiossincrático (todos N(0,1) iid), rho_G a correlação
de ativos entre setores distintos e rho_S (>= rho_G) a correlação de
ativos dentro do mesmo setor. Default quando Y_n < Phi^{-1}(p_n).
Variante t-Student: Y multiplicado por sqrt(nu/V), V ~ qui-quadrado(nu),
com limiar t_nu^{-1}(p_n).

A calibração segue a mesma estratégia do modelo de um fator: rho_G e
rho_S são obtidos numericamente para que as correlações de default
implícitas (via normal bivariada) atinjam os alvos inter e intra-setor
definidos pelo usuário.
"""
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import norm, t as myT

import cmUtilities as util
import thresholdModels as th


def calibrate_rho(pbar: float, rho_d_target: float) -> float:
    """Correlação de ativos que reproduz a correlação de default alvo."""
    res = scipy.optimize.minimize_scalar(
        th.calibrateGaussian, bounds=(1e-4, 0.99), args=(pbar, rho_d_target),
        method="bounded")
    return float(res.x)


def get_y_sector(N, M, p, rho_g, rho_s, sector_idx, nu, is_t=0, rng=np.random):
    """Gera a matriz latente Y (M x N) do modelo de dois níveis."""
    n_sec = int(sector_idx.max()) + 1
    G = rng.normal(0, 1, (M, 1))                      # fator global
    F = rng.normal(0, 1, (M, n_sec))                  # fatores setoriais
    eps = rng.normal(0, 1, (M, N))                    # idiossincrático
    Y = (np.sqrt(rho_g) * G
         + np.sqrt(max(rho_s - rho_g, 0.0)) * F[:, sector_idx]
         + np.sqrt(1.0 - rho_s) * eps)
    if is_t == 1:
        W = np.sqrt(nu / rng.chisquare(nu, (M, 1)))
        Y = W * Y
    return Y


def simulate_losses(N, M, p, c, rho_g, rho_s, sector_idx, nu, is_t=0):
    """Distribuição de perdas e matriz de perdas por nome (para contribuições)."""
    Y = get_y_sector(N, M, p, rho_g, rho_s, sector_idx, nu, is_t)
    K = (myT.ppf(p, nu) if is_t == 1 else norm.ppf(p)) * np.ones((M, 1))
    default_ind = 1 * np.less(Y, K)                   # M x N
    loss_by_name = default_ind * c                    # M x N
    loss = loss_by_name.sum(axis=1)                   # M
    return loss, loss_by_name


def run_multifactor_analysis(
    port,
    M: int = 100_000,
    alphas: np.ndarray = np.array([0.95, 0.99, 0.999, 0.9997]),
    rho_intra_target: float = 0.08,
    rho_inter_target: float = 0.03,
    nu: float = 10.0,
    contrib_alpha: float = 0.99,
    contrib_S: int = 10,
) -> dict:
    """Executa o modelo multifatorial setorial (gaussiano e t) e devolve
    medidas de risco, calibração e contribuições por setor e por nome."""
    if "setor" not in port.df.columns:
        raise ValueError("A carteira não possui a coluna 'setor' — "
                         "necessária para o modelo multifatorial.")
    if rho_inter_target > rho_intra_target:
        raise ValueError("O alvo de correlação inter-setor não pode exceder o intra-setor.")

    N, p, c = port.N, port.p, port.c
    sectors = port.df["setor"].astype(str)
    sector_names, sector_idx = np.unique(sectors, return_inverse=True)
    pbar = float(np.dot(c / c.sum(), p))

    # ---- calibração: correlações de ativos que atingem os alvos de default ----
    rho_s = calibrate_rho(pbar, rho_intra_target)   # intra-setor
    rho_g = calibrate_rho(pbar, rho_inter_target)   # inter-setor (fator global)

    out = {
        "setores": list(sector_names),
        "calibracao": {
            "rho_default_intra_alvo": rho_intra_target,
            "rho_default_inter_alvo": rho_inter_target,
            "rho_ativos_intra": rho_s,
            "rho_ativos_inter": rho_g,
            "pbar": pbar,
        },
        "medidas": [],
    }

    # ---- medidas de risco: gaussiano e t multifatoriais ----
    dists = {}
    for is_t, nome in [(0, "Threshold Gaussiano multifatorial (setores)"),
                       (1, f"Threshold t-Student multifatorial (setores, nu={nu:g})")]:
        loss, _ = simulate_losses(N, M, p, c, rho_g, rho_s, sector_idx, nu, is_t)
        el, ul, var, es = util.computeRiskMeasures(M, np.sort(loss), alphas)
        row = {"modelo": nome, "EL": float(el), "UL": float(ul)}
        for a, v, e in zip(alphas, np.atleast_1d(var), np.atleast_1d(es)):
            row[f"VaR {a:.2%}"] = float(v)
            row[f"ES {a:.2%}"] = float(e)
        out["medidas"].append(row)
        dists[nome] = np.sort(loss)
    out["distribuicoes"] = dists

    # ---- contribuições de risco por nome e por setor (gaussiano MF) ----
    contrib_name = np.zeros((contrib_S, N))
    var_s = np.zeros(contrib_S)
    es_s = np.zeros(contrib_S)
    Mc = min(M, 100_000)
    for s in range(contrib_S):
        loss, loss_by_name = simulate_losses(N, Mc, p, c, rho_g, rho_s,
                                             sector_idx, nu, 0)
        var_a = np.quantile(loss, contrib_alpha)
        tail = loss >= var_a
        contrib_name[s] = loss_by_name[tail].mean(axis=0)   # E[c_n 1_Dn | L >= VaR]
        var_s[s], es_s[s] = var_a, loss[tail].mean()
    contrib_es = contrib_name.mean(axis=0)

    df_sec = pd.DataFrame({
        "setor": sectors.values,
        "exposicao_liquida": c,
        "perda_esperada": p * c,
        "contrib_ES": contrib_es,
    }).groupby("setor").sum()
    df_sec["n_contrapartes"] = sectors.value_counts()
    df_sec["exposicao_pct"] = df_sec["exposicao_liquida"] / c.sum()
    df_sec["contrib_ES_pct"] = df_sec["contrib_ES"] / contrib_es.sum()
    df_sec["razao_risco_exposicao"] = df_sec["contrib_ES_pct"] / df_sec["exposicao_pct"]
    out["contrib_por_setor"] = df_sec.sort_values("contrib_ES", ascending=False).reset_index()

    w_sec = df_sec["exposicao_liquida"] / df_sec["exposicao_liquida"].sum()
    out["hhi_setorial"] = float((w_sec ** 2).sum())
    out["contrib_por_nome_ES"] = contrib_es
    out["contrib_alpha"] = contrib_alpha
    out["es_medio"] = float(es_s.mean())
    out["var_medio"] = float(var_s.mean())
    return out
