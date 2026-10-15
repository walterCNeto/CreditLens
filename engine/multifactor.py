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

Cargas por setor (opcional). Com uma tabela de alvos por setor, cada setor
s recebe as próprias correlações de ativos, rho_G,s e rho_S,s:

    Y_n = a_s G + b_s F_s + d_s eps_n,
    a_s = sqrt(rho_G,s),  b_s = sqrt(rho_S,s - rho_G,s),  d_s = sqrt(1 - rho_S,s)

A correlação de ativos dentro do setor s é rho_S,s; entre os setores s e
t, a_s a_t = sqrt(rho_G,s rho_G,t). A estrutura entre setores é de posto 1
(um fator global): pares diferem pela carga de cada setor, mas a razão
corr(s,t)/corr(s,u) = a_t/a_u é a mesma para todo s. Cada setor é
calibrado com a sua PD média ponderada. Com escalares, o modelo é o de
carga uniforme acima, e os resultados são idênticos.
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


def cargas(rho_g, rho_s, n_sec):
    """Cargas por setor (a, b, d): Y = a_s G + b_s F_s + d_s eps.
    rho_g e rho_s podem ser escalares (carga uniforme) ou vetores com uma
    entrada por setor, na ordem de np.unique dos nomes de setor."""
    g = np.broadcast_to(np.asarray(rho_g, dtype=float), (n_sec,)).copy()
    s = np.broadcast_to(np.asarray(rho_s, dtype=float), (n_sec,)).copy()
    return np.sqrt(g), np.sqrt(np.maximum(s - g, 0.0)), np.sqrt(1.0 - s)


def rho_default_par(rho_a, p1, p2):
    """Correlação de default implícita entre dois devedores com PDs p1 e p2
    e correlação de ativos rho_a (normal bivariada)."""
    from scipy.stats import multivariate_normal
    k1, k2 = norm.ppf(p1), norm.ppf(p2)
    p12 = multivariate_normal(mean=[0, 0], cov=[[1, rho_a], [rho_a, 1]]).cdf(
        [k1, k2])
    return float((p12 - p1 * p2) / np.sqrt(p1 * (1 - p1) * p2 * (1 - p2)))


def calibrar_por_setor(setores, c, p, rho_intra, rho_inter, rho_setor):
    """Calibra rho_S,s e rho_G,s setor a setor.

    `rho_setor`: DataFrame com colunas setor, rho_intra, rho_inter (alvos de
    correlação de DEFAULT). Cada setor informado é calibrado na sua PD média
    ponderada pela exposição líquida; os setores ausentes da tabela mantêm
    as correlações de ativos da carga uniforme (alvos globais calibrados na
    PD média da carteira), de modo que uma tabela parcial só altera os
    setores que ela lista."""
    nomes, idx = np.unique(np.asarray(setores).astype(str), return_inverse=True)
    tab = {}
    if rho_setor is not None and len(rho_setor):
        for _, r in pd.DataFrame(rho_setor).iterrows():
            tab[str(r["setor"])] = (float(r["rho_intra"]), float(r["rho_inter"]))
    pbar = float(np.dot(c, p) / c.sum())
    rs0, rg0 = calibrate_rho(pbar, rho_intra), calibrate_rho(pbar, rho_inter)
    linhas, vg, vs, pb = [], [], [], []
    for k, nome in enumerate(nomes):
        m = idx == k
        p_s = float(np.dot(c[m], p[m]) / c[m].sum())
        intra, inter = tab.get(nome, (rho_intra, rho_inter))
        if inter > intra:
            raise ValueError(f"Setor {nome}: o alvo inter-setor ({inter:.3f}) "
                             f"excede o intra-setor ({intra:.3f}).")
        if nome in tab:
            rs, rg = calibrate_rho(p_s, intra), calibrate_rho(p_s, inter)
        else:
            rs, rg = rs0, rg0
        vg.append(rg); vs.append(rs); pb.append(p_s)
        linhas.append({"setor": nome, "pd_media": p_s,
                       "exposicao_liquida": float(c[m].sum()),
                       "rho_default_intra_alvo": intra,
                       "rho_default_inter_alvo": inter,
                       "pd_calibracao": p_s if nome in tab else pbar,
                       "rho_ativos_intra": rs, "rho_ativos_inter": rg,
                       "carga_global": float(np.sqrt(rg)),
                       "carga_setorial": float(np.sqrt(max(rs - rg, 0.0))),
                       "informado": nome in tab})
    vg, vs, pb = np.array(vg), np.array(vs), np.array(pb)
    w = np.array([l["exposicao_liquida"] for l in linhas]); w = w / w.sum()
    # matriz de correlação de ativos entre setores (diagonal: intra)
    mat = np.sqrt(np.outer(vg, vg))
    np.fill_diagonal(mat, vs)
    # correlação de default implícita, par a par, nas PDs dos setores
    n = len(nomes)
    md = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            md[i, j] = md[j, i] = rho_default_par(mat[i, j], pb[i], pb[j])
    return {
        "por_setor": pd.DataFrame(linhas),
        "rho_ativos_inter_vec": vg.tolist(),
        "rho_ativos_intra_vec": vs.tolist(),
        "matriz_ativos": pd.DataFrame(mat, index=nomes, columns=nomes),
        "matriz_default": pd.DataFrame(md, index=nomes, columns=nomes),
        "rho_ativos_inter": float(np.dot(w, vg)),
        "rho_ativos_intra": float(np.dot(w, vs)),
        "cargas_por_setor": True,
    }


def _pct(x):
    return f"{100 * x:.3f}%".replace(".", ",")


def _num(x):
    return f"{x:.3f}".replace(".", ",")


def leitura_setores(cal, rho_intra_ref, rho_inter_ref, rs_ref=None, rg_ref=None):
    """Interpretação, em linguagem de negócio, das correlações por setor.

    Para cada setor: o nível da dependência intra-setor (alvo de default
    comparado à referência uniforme) e a sua origem — a fração da
    correlação de ativos intra-setor que vem do fator global,
    rho_G,s / rho_S,s. Perto de 1, o setor é CÍCLICO (os defaults andam
    juntos porque a economia os move juntos, e o setor contagia os demais);
    perto de 0, é CONCENTRADO (um choque próprio do setor derruba muitos
    nomes de uma vez, mas pouco se espalha). Devolve (quadro, pares, resumo)."""
    ps = cal["por_setor"].copy()
    ps["fracao_global"] = ps["rho_ativos_inter"] / ps["rho_ativos_intra"]

    def nivel(x):
        if x < 0.75 * rho_intra_ref:
            return "baixa"
        if x > 1.25 * rho_intra_ref:
            return "alta"
        return "na referência"

    def perfil(f):
        if f >= 0.75:
            return "cíclico"
        if f <= 0.40:
            return "concentrado"
        return "misto"

    def frase(r):
        txt = {"cíclico": "quase toda a dependência vem do ciclo econômico; "
                          "quebra junto com a economia e contagia os demais "
                          "setores",
               "concentrado": "o choque próprio do setor domina; muitos nomes "
                              "quebram juntos, mas o risco pouco se espalha",
               "misto": "dependência repartida entre o choque do setor e o "
                        "ciclo econômico"}[r["perfil"]]
        return (f"dependência intra-setor {r['nivel']} "
                f"({_pct(r['rho_default_intra_alvo'])}); {txt}")

    ps["nivel"] = ps["rho_default_intra_alvo"].map(nivel)
    ps["perfil"] = ps["fracao_global"].map(perfil)
    ps["leitura"] = ps.apply(frase, axis=1)

    md = cal["matriz_default"]
    nomes = list(md.index)
    pares = sorted(((md.iloc[i, j], nomes[i], nomes[j])
                    for i in range(len(nomes)) for j in range(i + 1, len(nomes))),
                   reverse=True)
    w = ps["exposicao_liquida"] / ps["exposicao_liquida"].sum()
    rs_m = float((w * ps["rho_ativos_intra"]).sum())
    rg_m = float((w * ps["rho_ativos_inter"]).sum())
    resumo = {"rho_s_medio": rs_m, "rho_g_medio": rg_m,
              "rho_s_ref": rs_ref, "rho_g_ref": rg_ref}
    if rs_ref is not None and rg_ref is not None:
        d = (rs_m - rs_ref) + (rg_m - rg_ref)
        resumo["direcao"] = ("acima" if d > 0.005 else
                             "abaixo" if d < -0.005 else "em linha com")
    return ps, pares, resumo


def rhos_para_motor(cal):
    """(rho_g, rho_s) para os motores: vetores por setor, se calibrados,
    ou os escalares da carga uniforme."""
    if cal.get("cargas_por_setor"):
        return (np.asarray(cal["rho_ativos_inter_vec"]),
                np.asarray(cal["rho_ativos_intra_vec"]))
    return cal["rho_ativos_inter"], cal["rho_ativos_intra"]


def get_y_sector(N, M, p, rho_g, rho_s, sector_idx, nu, is_t=0, rng=np.random):
    """Gera a matriz latente Y (M x N) do modelo de dois níveis."""
    n_sec = int(sector_idx.max()) + 1
    a, b, d = cargas(rho_g, rho_s, n_sec)
    G = rng.normal(0, 1, (M, 1))                      # fator global
    F = rng.normal(0, 1, (M, n_sec))                  # fatores setoriais
    eps = rng.normal(0, 1, (M, N))                    # idiossincrático
    Y = (a[sector_idx] * G
         + b[sector_idx] * F[:, sector_idx]
         + d[sector_idx] * eps)
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
    rho_setor=None,
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
    calib = {
        "rho_default_intra_alvo": rho_intra_target,
        "rho_default_inter_alvo": rho_inter_target,
        "pbar": pbar,
    }
    if rho_setor is not None and len(rho_setor):
        calib.update(calibrar_por_setor(sectors, c, p, rho_intra_target,
                                        rho_inter_target, rho_setor))
        rho_g, rho_s = rhos_para_motor(calib)
    else:
        rho_s = calibrate_rho(pbar, rho_intra_target)   # intra-setor
        rho_g = calibrate_rho(pbar, rho_inter_target)   # inter-setor (global)
        calib.update({"rho_ativos_intra": rho_s, "rho_ativos_inter": rho_g})

    out = {
        "setores": list(sector_names),
        "calibracao": calib,
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
