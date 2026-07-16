"""
Motor de simulação para carteiras agregadas (pools + grandes exposições).

Cada linha da carteira agregada carrega n_g contratos, severidade total
C_g e PD p_g. Condicional ao(s) fator(es) sistemático(s), os defaults
dentro do bucket são independentes com probabilidade condicional p_g(.),
e a perda do bucket é gerada em três regimes:

    n_g = 1                : Bernoulli(p_cond) x C_g        (grande exposição)
    1 < n_g <= POOL_MIN    : Binomial(n_g, p_cond) x C_g/n_g
    n_g > POOL_MIN         : C_g x p_cond                   (large pool / LLN)

Probabilidades condicionais (limiar de ativos):

    Gaussiano 1F : p_cond = Phi( (Phi^{-1}(p_g) - sqrt(rho) G) / sqrt(1-rho) )
    Multifatorial: substitui sqrt(rho) G pela parte sistemática de dois
                   níveis sqrt(rho_G) G + sqrt(rho_S-rho_G) F_s, e o
                   denominador por sqrt(1-rho_S)
    t-Student    : com limiar K = t_nu^{-1}(p_g) e choque radial V ~ chi2_nu,
                   p_cond = Phi( (K sqrt(V/nu) - sistematica) / sqrt(1-rho) )

CreditRisk+ agrega nativamente: H_g ~ Poisson(n_g p_g (1-w+wS)),
perda = (C_g/n_g) H_g. Binomial independente: Binomial(n_g, p_g).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, t as myT

import cmUtilities as util

POOL_MIN = 50          # acima disto usa a aproximação de carteira grande


def _measures(loss_sorted, M, alphas):
    el, ul, var, es = util.computeRiskMeasures(M, loss_sorted, alphas)
    return float(el), float(ul), np.atleast_1d(var), np.atleast_1d(es)


def _bucket_losses(p_cond, C, n, cbar, rng):
    """Perda por bucket dado p condicional (M x G), nos três regimes."""
    M = p_cond.shape[0]
    loss = np.empty(p_cond.shape, dtype=np.float32)
    pool = n > POOL_MIN
    small = ~pool & (n > 1)
    single = n == 1
    if pool.any():
        loss[:, pool] = C[pool] * p_cond[:, pool]
    if small.any():
        loss[:, small] = cbar[small] * rng.binomial(
            n[small], p_cond[:, small])   # broadcasting (G,) x (M,G)
    if single.any():
        loss[:, single] = C[single] * (rng.random((M, single.sum()))
                                       < p_cond[:, single])
    return loss


def simulate_pooled(port, M=100_000, alphas=np.array([0.95, 0.99, 0.999, 0.9997]),
                    rho=0.20, rho_g=None, rho_s=None, nu=10.0,
                    contrib_alpha=0.99, contrib_S=10, seed=42,
                    progress=None):
    """Roda a suíte pooled: binomial independente, CreditRisk+ (via
    parâmetros padrão do chamador), threshold gaussiano/t 1F e, se
    rho_g/rho_s informados e houver setor, multifatorial. Devolve medidas,
    distribuições e contribuições de ES por bucket."""
    rng = np.random.default_rng(seed)
    df = port.df
    n = df["n_contratos"].to_numpy(int) if "n_contratos" in df.columns \
        else np.ones(port.N, int)
    C = port.c
    cbar = C / n
    p = port.p
    G_ = port.N
    pbar = float(np.dot(C / C.sum(), p))

    use_mf = rho_g is not None and rho_s is not None and "setor" in df.columns
    if use_mf:
        _, sec_idx = np.unique(df["setor"].astype(str), return_inverse=True)
        n_sec = sec_idx.max() + 1

    K_g = norm.ppf(p)
    K_t = myT.ppf(p, nu)
    rows, dists = [], {}
    contrib_es = None

    def tick(m):
        if progress:
            progress(m)

    # ---------- binomial independente (pooled) ----------
    tick("Binomial independente (pooled)...")
    loss = np.empty(M)
    _B = 20_000
    for i0 in range(0, M, _B):
        m = min(_B, M - i0)
        loss[i0:i0 + m] = (rng.binomial(n, np.broadcast_to(
            p.astype(np.float32), (m, G_))) * cbar).sum(axis=1)
    el, ul, var, es = _measures(np.sort(loss), M, alphas)
    rows.append(("Binomial independente (pooled)", el, ul, var, es))
    dists["Binomial independente"] = np.sort(loss)

    # ---------- threshold gaussiano / t, 1F e multifatorial ----------
    BLOCO = 20_000   # simulação em blocos p/ controle de memória

    def run_threshold(nome, is_t, mf, keep_by_bucket=False):
        loss = np.empty(M, dtype=np.float64)
        loss_b_full = np.zeros(G_) if not keep_by_bucket else \
            np.empty((M, G_), dtype=np.float32)
        for i0 in range(0, M, BLOCO):
            m = min(BLOCO, M - i0)
            Gf = rng.normal(0, 1, (m, 1))
            if mf:
                F = rng.normal(0, 1, (m, n_sec))
                syst = (np.sqrt(rho_g) * Gf
                        + np.sqrt(max(rho_s - rho_g, 0)) * F[:, sec_idx])
                denom = np.sqrt(1 - rho_s)
            else:
                syst = np.sqrt(rho) * Gf
                denom = np.sqrt(1 - rho)
            if is_t:
                V = rng.chisquare(nu, (m, 1))
                K = K_t[None, :] * np.sqrt(V / nu)
            else:
                K = K_g[None, :]
            p_cond = norm.cdf((K - syst) / denom).astype(np.float32)
            lb = _bucket_losses(p_cond, C, n, cbar, rng)
            loss[i0:i0 + m] = lb.sum(axis=1)
            if keep_by_bucket:
                loss_b_full[i0:i0 + m] = lb
            del p_cond, lb
        return loss, loss_b_full if keep_by_bucket else None

    especs = [("Threshold Gaussiano 1F (pooled)", 0, False),
              (f"Threshold t-Student 1F (pooled, nu={nu:g})", 1, False)]
    if use_mf:
        especs += [("Threshold Gaussiano multifatorial (pooled)", 0, True),
                   (f"Threshold t-Student multifatorial (pooled, nu={nu:g})", 1, True)]

    loss_b_contrib = None
    for nome, is_t, mf in especs:
        tick(f"{nome}...")
        keep = (mf and use_mf and not is_t) or (not use_mf and not is_t and not mf)
        loss, loss_b = run_threshold(nome, is_t, mf, keep_by_bucket=keep)
        el, ul, var, es = _measures(np.sort(loss), M, alphas)
        rows.append((nome, el, ul, var, es))
        dists[nome.replace(" (pooled)", "")] = np.sort(loss)
        if keep:
            loss_b_contrib = (nome, loss_b, loss)   # base p/ contribuições

    # ---------- contribuições de ES por bucket ----------
    if loss_b_contrib is not None:
        tick("Contribuições de ES por bucket...")
        nome_c, loss_b, loss = loss_b_contrib
        var_a = np.quantile(loss, contrib_alpha)
        tail = loss >= var_a
        contrib_es = loss_b[tail].mean(axis=0)
        es_total = float(loss[tail].mean())
        contrib = df[["id_contraparte", "setor", "rating", "exposicao",
                      "pd"]].copy()
        contrib["n_contratos"] = n
        contrib["contrib_ES"] = contrib_es
        contrib["contrib_ES_pct"] = contrib_es / es_total
        contrib = contrib.sort_values("contrib_ES", ascending=False)
    else:
        contrib, es_total, nome_c = None, None, None

    medidas = []
    for nome, el, ul, var, es in rows:
        row = {"modelo": nome, "EL": el, "UL": ul}
        for a, v, e in zip(alphas, var, es):
            row[f"VaR {a:.2%}"] = float(v)
            row[f"ES {a:.2%}"] = float(e)
        medidas.append(row)

    return {
        "medidas": pd.DataFrame(medidas),
        "distribuicoes": dists,
        "contribuicoes_bucket": contrib,
        "contrib_alpha": contrib_alpha,
        "es_total": es_total,
        "modelo_contrib": nome_c,
        "pbar": pbar,
    }
