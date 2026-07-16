"""
Orquestração dos modelos do Bolder (2018) sobre uma carteira CreditLens.

Suíte de modelos (referência: capítulos do livro):
  Cap. 2  Binomial independente (analítico + Monte Carlo)
  Cap. 3  Mixture models: beta-binomial, CreditRisk+ (Poisson-gama 1 fator)
  Cap. 4  Threshold models: Gaussiano 1 fator, t-Student 1 fator, ASRF analítico
  Cap. 6  IRB de Basileia (capital regulatório) + ajuste de granularidade
  Cap. 7  Contribuições de VaR/ES (decomposição Monte Carlo)

Todos os modelos são calibrados, quando aplicável, ao mesmo par
(pbar, rho_alvo): PD média ponderada por exposição e correlação de
default alvo — seguindo a estratégia de comparação do próprio livro.
"""
import time

import numpy as np
import pandas as pd
import scipy.optimize

from . import _BOLDER_DIR  # garante path
import cmUtilities as util
import binomialPoissonModels as bp
import mixtureModels as mix
import thresholdModels as th
import irbModel as irb
import varContributions as vc

DEFAULT_ALPHAS = np.array([0.95, 0.99, 0.999, 0.9997])


def _fmt_alphas(alphas):
    return [f"{a:.2%}".replace(".00%", "%") for a in alphas]


def calibrate_gaussian_rho(pbar: float, rho_target: float) -> float:
    """Correlação de ativos rho_A do modelo gaussiano que reproduz a
    correlação de default alvo (Bolder, cap. 4)."""
    res = scipy.optimize.minimize_scalar(
        th.calibrateGaussian, bounds=(1e-4, 0.99), args=(pbar, rho_target),
        method="bounded",
    )
    return float(res.x)


def run_full_analysis(
    port,
    M: int = 100_000,
    alphas: np.ndarray = DEFAULT_ALPHAS,
    rho_target: float = 0.05,
    nu: float = 10.0,
    crp_w: float = 0.35,
    crp_v: float = 1.2,
    run_contributions: bool = True,
    run_multifactor: bool = False,
    rho_intra: float = 0.08,
    rho_inter: float = 0.03,
    contrib_S: int = 10,
    contrib_alpha: float = 0.99,
    seed: int | None = 42,
    progress=None,
) -> dict:
    """Executa a suíte completa e devolve dicionário de resultados."""
    if seed is not None:
        np.random.seed(seed)

    def _tick(msg):
        if progress:
            progress(msg)

    N, p, c, tenor = port.N, port.p, port.c, port.tenor
    summary = port.summary()
    pbar = summary["pd_media_ponderada"]
    total = summary["exposicao_liquida_total"]
    results = {"resumo": summary, "parametros": {
        "M": M, "alphas": list(map(float, alphas)), "rho_alvo": rho_target,
        "nu_t": nu, "crplus_w": crp_w, "crplus_v": crp_v, "seed": seed,
    }}

    rows = []

    def add_row(modelo, el, ul, var, es, tempo):
        row = {"modelo": modelo, "EL": el, "UL": ul, "tempo_s": tempo}
        for a, v, e in zip(alphas, np.atleast_1d(var), np.atleast_1d(es)):
            row[f"VaR {a:.2%}"] = v
            row[f"ES {a:.2%}"] = e
        rows.append(row)

    # ---------------- Cap. 2: independentes ----------------
    _tick("Binomial independente (analítico, homogêneo)...")
    t0 = time.time()
    # Aproximação homogênea do livro: p = pbar, c = exposição média
    cbar = total / N
    _, _, var, es = bp.independentBinomialAnalytic(N, pbar, cbar, alphas)
    el = pbar * total
    ul = np.sqrt(N * pbar * (1 - pbar)) * cbar
    add_row("Binomial independente (analítico, homog.)", el, ul, var, es, time.time() - t0)

    _tick("Binomial independente (Monte Carlo)...")
    t0 = time.time()
    el, ul, var, es = bp.independentBinomialSimulation(N, M, p, c, alphas)
    add_row("Binomial independente (MC)", el, ul, var, es, time.time() - t0)

    # ---------------- Cap. 3: misturas ----------------
    _tick("Beta-binomial...")
    t0 = time.time()
    a_beta, b_beta = mix.betaCalibrate(pbar, rho_target)
    el, ul, var, es = mix.betaBinomialSimulation(N, M, c, a_beta, b_beta, alphas)
    add_row("Beta-binomial (mixture)", el, ul, var, es, time.time() - t0)
    results["calibracao_beta"] = {"a": float(a_beta), "b": float(b_beta)}

    _tick("CreditRisk+ (1 fator)...")
    t0 = time.time()
    el, ul, var, es = mix.crPlusOneFactor(N, M, crp_w, p, c, crp_v, alphas)
    add_row("CreditRisk+ (Poisson-gama 1F)", el, ul, var, es, time.time() - t0)

    # ---------------- Cap. 4: limiar ----------------
    _tick("Calibrando correlação de ativos...")
    rho_asset = calibrate_gaussian_rho(pbar, rho_target)
    results["calibracao_gaussiana"] = {
        "rho_default_alvo": rho_target, "rho_ativos": rho_asset, "pbar": pbar,
    }

    _tick("Threshold gaussiano 1 fator...")
    t0 = time.time()
    el, ul, var, es = th.oneFactorThresholdModel(N, M, p, c, rho_asset, nu, alphas, isT=0)
    add_row("Threshold Gaussiano 1F", el, ul, var, es, time.time() - t0)

    _tick("Threshold t-Student 1 fator...")
    t0 = time.time()
    el, ul, var, es = th.oneFactorThresholdModel(N, M, p, c, rho_asset, nu, alphas, isT=1)
    add_row(f"Threshold t-Student 1F (nu={nu:g})", el, ul, var, es, time.time() - t0)

    _tick("ASRF analítico...")
    t0 = time.time()
    _, _, var_asrf, es_asrf = th.asrfModel(pbar, rho_asset, c, alphas)
    add_row("ASRF (analítico, carteira infinita)", pbar * total, np.nan,
            var_asrf, es_asrf, time.time() - t0)

    # ---------------- Cap. 6: IRB de Basileia ----------------
    _tick("Capital IRB de Basileia...")
    t0 = time.time()
    k_n = irb.getBaselK(p, tenor, 0.999)  # vetorizado por contraparte
    cap_n = c * k_n
    rho_basel = irb.getRho(p)
    # Ajuste de granularidade Gordy–Lütkebohmert via CreditRisk+ (Bolder, ex. cap. 6)
    rho_g = float(np.dot(rho_basel, c) / c.sum())
    ga_a, ga_xi, ga_gbar = 0.25, 0.25, 1.0
    ga_w = irb.getW(pbar, ga_a, rho_g, 0.999)
    ga_total = float(irb.granularityAdjustmentCR(ga_a, ga_w, ga_gbar, ga_xi, p, c, 0.999))
    results["irb"] = {
        "capital_total_999": float(cap_n.sum()),
        "capital_pct_exposicao": float(cap_n.sum() / total),
        "ajuste_granularidade": ga_total,
        "capital_com_ga": float(cap_n.sum() + ga_total),
        "rwa_equivalente": float(cap_n.sum() * 12.5),
        "capital_por_contraparte": cap_n,
        "coeficiente_k": k_n,
        "rho_basel_medio": float(np.mean(rho_basel)),
        "rho_basel_ponderado": rho_g,
        "tempo_s": time.time() - t0,
    }
    add_row("IRB Basileia (VaR 99,90% regulatório)",
            pbar * total, np.nan,
            [np.nan if abs(a - 0.999) > 1e-9 else cap_n.sum() + pbar * total for a in alphas],
            [np.nan] * len(alphas), time.time() - t0)

    # ---------------- distribuição de perdas p/ gráficos ----------------
    _tick("Gerando distribuições de perda para gráficos...")
    Mg = min(M, 50_000)
    dists = {}
    Yg = th.getY(N, Mg, p, rho_asset, nu, 0)
    K = th.norm.ppf(p) * np.ones((Mg, 1))
    dists["Threshold Gaussiano 1F"] = np.sort(np.dot(1 * np.less(Yg, K), c), axis=None)
    Yt = th.getY(N, Mg, p, rho_asset, nu, 1)
    Kt = th.myT.ppf(p, nu) * np.ones((Mg, 1))
    dists["Threshold t-Student 1F"] = np.sort(np.dot(1 * np.less(Yt, Kt), c), axis=None)
    dists["Binomial independente"] = bp.independentBinomialLossDistribution(
        N, Mg, p, c, alphas)
    results["distribuicoes"] = dists

    # ---------------- Cap. 7: contribuições de risco ----------------
    if run_contributions:
        _tick(f"Contribuições de VaR/ES (decomposição MC, S={contrib_S})...")
        t0 = time.time()
        Mc = min(M, 100_000)
        contr, var_s, es_s = vc.mcThresholdGDecomposition(
            N, Mc, contrib_S, p, c, rho_asset, nu, 0, contrib_alpha)
        contrib_var = contr[:, :, 0].mean(axis=1)
        contrib_es = contr[:, :, 1].mean(axis=1)
        results["contribuicoes"] = {
            "alpha": contrib_alpha,
            "var_contrib": contrib_var,
            "es_contrib": contrib_es,
            "var_medio": float(var_s.mean()),
            "es_medio": float(es_s.mean()),
            "tempo_s": time.time() - t0,
        }

    # ---------------- Extensão multifatorial setorial ----------------
    if run_multifactor and "setor" in port.df.columns:
        _tick("Modelo multifatorial setorial...")
        from .multifactor import run_multifactor_analysis
        mf = run_multifactor_analysis(
            port, M=M, alphas=alphas, rho_intra_target=rho_intra,
            rho_inter_target=rho_inter, nu=nu,
            contrib_alpha=contrib_alpha, contrib_S=contrib_S)
        for row in mf["medidas"]:
            row["tempo_s"] = np.nan
            rows.append(row)
        results["multifator"] = mf
        results["distribuicoes"].update(
            {k: v[::max(1, len(v) // 50_000)]  # subamostra sistemática do array ordenado
             for k, v in mf["distribuicoes"].items()})

    results["medidas"] = pd.DataFrame(rows)
    return results


def results_to_excel(results: dict, port, path: str):
    """Exporta resultados para Excel (dados calculados, sem fórmulas)."""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        # carteira
        port.df.to_excel(xw, sheet_name="Carteira", index=False)

        # resumo
        pd.DataFrame(
            [{"métrica": k, "valor": v} for k, v in results["resumo"].items()]
        ).to_excel(xw, sheet_name="Resumo", index=False)

        # medidas por modelo
        results["medidas"].to_excel(xw, sheet_name="Medidas de Risco", index=False)

        # IRB por contraparte
        irb_res = results["irb"]
        df_irb = port.df[["id_contraparte", "exposicao", "pd", "lgd", "prazo_anos"]].copy()
        df_irb["coeficiente_K"] = irb_res["coeficiente_k"]
        df_irb["capital_IRB"] = irb_res["capital_por_contraparte"]
        df_irb["capital_pct_EAD"] = df_irb["capital_IRB"] / df_irb["exposicao"]
        df_irb.sort_values("capital_IRB", ascending=False).to_excel(
            xw, sheet_name="Capital IRB", index=False)

        # contribuições
        if "contribuicoes" in results:
            cc = results["contribuicoes"]
            df_c = port.df[["id_contraparte", "exposicao", "pd"]].copy()
            df_c["contrib_VaR"] = cc["var_contrib"]
            df_c["contrib_ES"] = cc["es_contrib"]
            df_c["contrib_ES_pct"] = df_c["contrib_ES"] / cc["es_medio"]
            df_c.sort_values("contrib_ES", ascending=False).to_excel(
                xw, sheet_name="Contribuições de Risco", index=False)

        # setores (multifatorial)
        if "multifator" in results:
            results["multifator"]["contrib_por_setor"].to_excel(
                xw, sheet_name="Setores", index=False)

        # parâmetros e calibração
        params = dict(results["parametros"])
        params.update({f"beta_{k}": v for k, v in results.get("calibracao_beta", {}).items()})
        params.update({f"gauss_{k}": v for k, v in results.get("calibracao_gaussiana", {}).items()})
        if "multifator" in results:
            params.update({f"mf_{k}": v for k, v in results["multifator"]["calibracao"].items()})
        pd.DataFrame([{"parâmetro": k, "valor": str(v)} for k, v in params.items()]
                     ).to_excel(xw, sheet_name="Parâmetros", index=False)


def run_pooled_analysis(
    port,
    M: int = 100_000,
    alphas: np.ndarray = DEFAULT_ALPHAS,
    rho_target: float = 0.05,
    rho_intra: float = 0.08,
    rho_inter: float = 0.03,
    nu: float = 10.0,
    contrib_alpha: float = 0.99,
    seed: int | None = 42,
    progress=None,
) -> dict:
    """Análise para carteiras AGREGADAS (pools + grandes exposições).

    Estrutura de resultados compatível com run_full_analysis para reuso
    das abas do aplicativo; modelos executados no motor pooled (ramos
    binomial condicional / large pool), IRB por bucket e ajuste de
    granularidade com o termo c^2 corrigido para C_g^2/n_g."""
    from .pooled import simulate_pooled
    from .multifactor import calibrate_rho
    import irbModel as irb

    df = port.df
    n = df["n_contratos"].to_numpy(int) if "n_contratos" in df.columns \
        else np.ones(port.N, int)
    C, p, tenor = port.c, port.p, port.tenor
    summary = port.summary()
    pbar = summary["pd_media_ponderada"]
    total = summary["exposicao_liquida_total"]

    rho1 = calibrate_rho(pbar, rho_target)
    rho_s = calibrate_rho(pbar, rho_intra)
    rho_g = calibrate_rho(pbar, rho_inter)
    use_mf = "setor" in df.columns

    res_sim = simulate_pooled(
        port, M=M, alphas=alphas, rho=rho1,
        rho_g=rho_g if use_mf else None, rho_s=rho_s if use_mf else None,
        nu=nu, contrib_alpha=contrib_alpha, seed=seed, progress=progress)

    medidas = res_sim["medidas"]

    # ---- ASRF e IRB (analíticos, por bucket) ----
    import thresholdModels as th
    _, _, var_asrf, es_asrf = th.asrfModel(pbar, rho1, C, alphas)
    row = {"modelo": "ASRF (analítico, carteira infinita)",
           "EL": pbar * total, "UL": np.nan}
    for a, v, e in zip(alphas, np.atleast_1d(var_asrf), np.atleast_1d(es_asrf)):
        row[f"VaR {a:.2%}"] = float(v); row[f"ES {a:.2%}"] = float(e)
    medidas = pd.concat([medidas, pd.DataFrame([row])], ignore_index=True)

    k_n = irb.getBaselK(p, tenor, 0.999)
    cap_n = C * k_n
    # GA com c^2 -> C^2/n (bucket homogêneo); grandes tem n=1
    ga_a, ga_xi, ga_gbar = 0.25, 0.25, 1.0
    rho_basel = irb.getRho(p)
    rho_g_w = float(np.dot(rho_basel, C) / C.sum())
    ga_w = irb.getW(pbar, ga_a, rho_g_w, 0.999)
    myDelta = irb.getDelta(ga_a, 0.999)
    Cn = irb.getC(ga_gbar, ga_xi)
    RKn = irb.getRK(ga_gbar, ga_a, ga_w, p, 0.999)
    Kn_cr = irb.getK(ga_gbar, ga_a, ga_w, p, 0.999)
    KStar = float(np.dot(C, Kn_cr))
    ratio = irb.myLGDRatio(ga_gbar, ga_xi)
    t1 = myDelta * (Cn * RKn + np.power(RKn, 2) * ratio)
    t2 = Kn_cr * (Cn + 2 * RKn * ratio)
    csq = np.power(C, 2) / n
    ga_total = float(np.dot(csq, t1 - t2) / (2 * KStar))

    results = {
        "resumo": summary,
        "pooled": True,
        "parametros": {"M": M, "alphas": list(map(float, alphas)),
                       "rho_alvo": rho_target, "rho_intra": rho_intra,
                       "rho_inter": rho_inter, "nu_t": nu, "seed": seed},
        "calibracao_gaussiana": {"rho_default_alvo": rho_target,
                                 "rho_ativos": rho1, "pbar": pbar},
        "medidas": medidas,
        "distribuicoes": res_sim["distribuicoes"],
        "irb": {
            "capital_total_999": float(cap_n.sum()),
            "capital_pct_exposicao": float(cap_n.sum() / total),
            "ajuste_granularidade": ga_total,
            "capital_com_ga": float(cap_n.sum() + ga_total),
            "rwa_equivalente": float(cap_n.sum() * 12.5),
            "capital_por_contraparte": cap_n,
            "coeficiente_k": k_n,
            "rho_basel_medio": float(np.mean(rho_basel)),
            "rho_basel_ponderado": rho_g_w,
        },
        "contrib_bucket": res_sim["contribuicoes_bucket"],
        "contrib_alpha": res_sim["contrib_alpha"],
        "es_total_contrib": res_sim["es_total"],
        "modelo_contrib": res_sim["modelo_contrib"],
    }
    if use_mf:
        cb = res_sim["contribuicoes_bucket"]
        sec = cb.groupby("setor").agg(
            exposicao_liquida=("exposicao", lambda x: float(
                (df.set_index("id_contraparte").loc[
                    cb.loc[x.index, "id_contraparte"], "exposicao"]
                 * df.set_index("id_contraparte").loc[
                    cb.loc[x.index, "id_contraparte"], "lgd"]).sum())),
            contrib_ES=("contrib_ES", "sum"),
            n_contrapartes=("n_contratos", "sum"))
        sec["exposicao_pct"] = sec["exposicao_liquida"] / sec["exposicao_liquida"].sum()
        sec["contrib_ES_pct"] = sec["contrib_ES"] / sec["contrib_ES"].sum()
        sec["razao_risco_exposicao"] = sec["contrib_ES_pct"] / sec["exposicao_pct"]
        w = sec["exposicao_liquida"] / sec["exposicao_liquida"].sum()
        results["multifator"] = {
            "setores": sec.index.tolist(),
            "calibracao": {"rho_default_intra_alvo": rho_intra,
                           "rho_default_inter_alvo": rho_inter,
                           "rho_ativos_intra": rho_s,
                           "rho_ativos_inter": rho_g, "pbar": pbar},
            "contrib_por_setor": sec.sort_values(
                "contrib_ES", ascending=False).reset_index(),
            "hhi_setorial": float((w ** 2).sum()),
            "contrib_alpha": contrib_alpha,
            "es_medio": res_sim["es_total"],
            "var_medio": np.nan,
            "distribuicoes": {},
        }
    return results
