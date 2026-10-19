"""
CreditRisk+ clássico (Credit Suisse First Boston, 1997).

Versão analítica, como no documento técnico original:

  * cada devedor tem intensidade de default estocástica
        λ_n(S) = p_n · (w₀ + w · S),  w₀ = 1 − w,
    com S ~ Gama(α, 1/α) (média 1, variância σ² = 1/α) — um setor
    sistemático e o setor específico (idiossincrático) w₀;
  * dado S, o número de defaults de cada devedor é Poisson (aproximação
    da Bernoulli: um devedor pode "inadimplir" mais de uma vez, com
    probabilidade desprezível para PD pequena);
  * as perdas são discretizadas em faixas: a severidade c_n vira o
    inteiro ν_n = max(1, round(c_n / L₀)) de unidades L₀, e a intensidade
    da faixa é ajustada para preservar a perda esperada,
        ε_j = Σ_{n∈j} p_n c_n / (ν_j L₀);
  * a função geradora de probabilidade (FGP) da perda, em unidades L₀, é
        G(z) = exp{Σ_j ε⁰_j (z^{ν_j} − 1)} · [(1 − δ)/(1 − δ P(z))]^α,
    com ε⁰_j = w₀ ε_j, ε¹_j = w ε_j, μ₁ = Σ_j ε¹_j, δ = σ²μ₁/(1 + σ²μ₁)
    e P(z) = Σ_j (ε¹_j / μ₁) z^{ν_j}: o setor específico é Poisson
    composta e o sistemático, binomial negativa composta;
  * a distribuição sai da recursão de Panjer aplicada a cada fator da FGP
    e da convolução das duas. Quando P(L = 0) fica abaixo da precisão de
    ponto flutuante (carteiras com milhares de defaults esperados), a
    recursão não pode partir de P(L = 0) — a instabilidade conhecida do
    método —, e a FGP é invertida exatamente por transformada rápida de
    Fourier (mesma distribuição, outro algoritmo).
  * como verificação independente, o VaR é recalculado pela aproximação
    de ponto de sela (Lugannani–Rice; Gordy, 2002) sobre a função
    geradora de cumulantes.

Nos grupos homogêneos da carteira agregada, cada linha entra com
intensidade n·p (n = número efetivo de contratos) e perda por default
c/n, o que é exato para soma de Poissons.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

UNIDADES_MAX = 1000          # maior severidade ≈ 1.000 unidades de L₀
LOG_MIN = -700.0             # abaixo disso, exp() sai da faixa do float


def _faixas(lam, u, L0):
    """Agrupa as linhas em faixas inteiras de L₀ preservando a perda
    esperada. Devolve (ν_j, ε_j)."""
    nu = np.maximum(1, np.rint(u / L0)).astype(np.int64)
    el = lam * u
    nus, inv = np.unique(nu, return_inverse=True)
    eps = np.bincount(inv, weights=el) / (nus * L0)
    return nus, eps


def _panjer_poisson(nus, mu, n_max):
    """Poisson composta: f(0) = e^{-Σμ}; f(n) = (1/n) Σ_j μ_j ν_j f(n−ν_j)."""
    f = np.zeros(n_max + 1)
    f[0] = np.exp(-mu.sum())
    a = mu * nus
    for n in range(1, n_max + 1):
        m = nus <= n
        if m.any():
            f[n] = (a[m] * f[n - nus[m]]).sum() / n
    return f


def _panjer_negbin(nus, q, alpha, delta, n_max):
    """Binomial negativa composta (classe (a, b, 0): a = δ, b = (α−1)δ):
    f(0) = (1−δ)^α; f(n) = Σ_j (a + b ν_j / n) q_j f(n−ν_j)."""
    f = np.zeros(n_max + 1)
    f[0] = (1.0 - delta) ** alpha
    a, b = delta, (alpha - 1.0) * delta
    for n in range(1, n_max + 1):
        m = nus <= n
        if m.any():
            f[n] = ((a + b * nus[m] / n) * q[m] * f[n - nus[m]]).sum()
    return f


def _fft(nus, eps0, eps1, alpha, delta, mu1, n_max):
    """Inversão exata da FGP no círculo unitário (sem aliasing se o
    suporte relevante couber em N pontos)."""
    N = 1 << int(np.ceil(np.log2(max(n_max + 1, 2) * 2)))
    k = np.arange(N)
    logG = np.zeros(N, dtype=complex)
    P = np.zeros(N, dtype=complex)
    for nu_j, e0, e1 in zip(nus, eps0, eps1):      # memória O(N)
        zj = np.exp(-2j * np.pi * ((k * int(nu_j)) % N) / N)
        logG += e0 * (zj - 1.0)
        if mu1 > 0:
            P += (e1 / mu1) * zj
    if mu1 > 0:
        logG += alpha * (np.log(1.0 - delta) - np.log(1.0 - delta * P))
    f = np.real(np.fft.ifft(np.exp(logG)))[: n_max + 1]
    return np.clip(f, 0.0, None)


def _medidas(f, L0, alphas):
    x = np.arange(len(f)) * L0
    F = np.cumsum(f)
    el = float((x * f).sum())
    ul = float(np.sqrt(max((x * x * f).sum() - el ** 2, 0.0)))
    var, es = [], []
    for a in alphas:
        k = int(np.searchsorted(F, a, side="left"))
        k = min(k, len(f) - 1)
        v = x[k]
        cauda = float((x[k + 1:] * f[k + 1:]).sum())
        e = (cauda + v * (F[k] - a)) / (1.0 - a)
        var.append(float(v)); es.append(float(e))
    return el, ul, np.array(var), np.array(es)


def ponto_de_sela(nus, eps0, eps1, alpha, delta, mu1, alphas):
    """Aproximação de ponto de sela de Lugannani–Rice (Gordy, 2002) sobre
    a função geradora de cumulantes K(t) = ln G(e^t). As "faixas" podem
    ser reais (severidades exatas, sem discretização). Devolve
    (UL, VaR, ES) na unidade de `nus`; o ES integra a cauda aproximada."""
    nu = np.asarray(nus, float)
    q = eps1 / mu1 if mu1 > 0 else np.zeros_like(eps1)

    def K(t):
        ex = np.exp(t * nu)
        k = (eps0 * (ex - 1.0)).sum()
        k1 = (eps0 * nu * ex).sum()
        k2 = (eps0 * nu * nu * ex).sum()
        if mu1 > 0:
            P, P1, P2 = (q * ex).sum(), (q * nu * ex).sum(), \
                (q * nu * nu * ex).sum()
            r = 1.0 - delta * P
            k += alpha * (np.log(1.0 - delta) - np.log(r))
            k1 += alpha * delta * P1 / r
            k2 += alpha * (delta * P2 / r + (delta * P1 / r) ** 2)
        return k, k1, k2

    if mu1 > 0:      # domínio: δ P(e^t) < 1
        with np.errstate(over="ignore"):
            tmax = brentq(lambda t: delta * (q * np.exp(t * nu)).sum()
                          - 1.0, 0.0, 50.0) * (1.0 - 1e-9)
    else:
        tmax = 50.0 / nu.max()
    _, media, var0 = K(0.0)
    lo = media + 0.5 * np.sqrt(var0)       # longe da média: L–R estável
    xmax = K(tmax * (1.0 - 1e-7))[1]

    def cauda(x):
        t = brentq(lambda t: K(t)[1] - x, 1e-14, tmax)
        k, _, k2 = K(t)
        w = np.sqrt(max(2.0 * (t * x - k), 1e-300))
        u = t * np.sqrt(k2)
        return 1.0 - norm.cdf(w) + norm.pdf(w) * (1.0 / u - 1.0 / w)

    var, es = [], []
    for a in alphas:
        try:
            v = brentq(lambda x: cauda(x) - (1.0 - a), lo, xmax)
        except ValueError:
            var.append(np.nan); es.append(np.nan)
            continue
        # ES = VaR + ∫_VaR^∞ P(L > x) dx / (1 − α), em grade até a cauda
        # ficar desprezível
        passo = max(np.sqrt(var0), 1e-9)
        fim = v + passo
        while fim < xmax and cauda(fim) > 1e-9 * (1.0 - a):
            fim = min(v + 2.0 * (fim - v), xmax * (1.0 - 1e-9))
            if fim >= xmax * (1.0 - 1e-9):
                break
        xs = np.linspace(v, fim, 241)
        ts = np.array([cauda(x) for x in xs])
        var.append(v)
        es.append(v + np.trapezoid(ts, xs) / (1.0 - a))
    return float(np.sqrt(var0)), np.array(var), np.array(es)


def amostra_pmf(pmf, L0, m=50_000):
    """Amostra estratificada (determinística) da distribuição exata, para
    gráficos: quantis nos pontos (i + 0,5)/m."""
    F = np.cumsum(pmf)
    u = (np.arange(m) + 0.5) / m
    return np.searchsorted(F, u * F[-1], side="left") * L0


def crplus_csfp(p, c, w, v, alphas, n=None, unidades_max=UNIDADES_MAX):
    """CreditRisk+ clássico de um setor sistemático (mais o específico).

    p, c: PD e severidade (EAD × LGD) por linha; n: número (efetivo) de
    contratos por linha na carteira agregada (perda por default = c/n);
    w: carga do setor sistemático; v: parâmetro de forma do fator gama
    (variância 1/v). Devolve (EL, UL, VaR, ES, info)."""
    p = np.asarray(p, float)
    c = np.asarray(c, float)
    n = np.ones_like(p) if n is None else np.asarray(n, float)
    lam, u = n * p, c / n
    L0 = float(u.max()) / unidades_max
    nus, eps = _faixas(lam, u, L0)
    alpha, s2 = float(v), 1.0 / float(v)
    eps0, eps1 = (1.0 - w) * eps, w * eps
    mu0, mu1 = float(eps0.sum()), float(eps1.sum())
    delta = s2 * mu1 / (1.0 + s2 * mu1)
    # suporte: média + 40 desvios (em unidades), mais a maior faixa
    m1 = float((eps * nus).sum())
    var_u = float((eps * nus ** 2).sum()) + s2 * float((eps1 * nus).sum()) ** 2
    n_max = int(m1 + 40.0 * np.sqrt(var_u) + nus.max() + 10)
    log_p0 = -mu0 + alpha * np.log1p(-delta) if mu1 > 0 else -mu0
    if log_p0 > LOG_MIN and n_max <= 400_000:
        f0 = _panjer_poisson(nus, eps0, n_max)
        f1 = (_panjer_negbin(nus, eps1 / mu1, alpha, delta, n_max)
              if mu1 > 0 else np.r_[1.0, np.zeros(n_max)])
        f = np.convolve(f0, f1)[: n_max + 1]
        metodo = "Panjer"
    else:
        f = _fft(nus, eps0, eps1, alpha, delta, mu1, n_max)
        metodo = "FFT (P(L=0) abaixo da precisão do Panjer)"
    massa = float(f.sum())
    el, ul, var, es = _medidas(f / massa, L0, alphas)
    # ponto de sela sobre as severidades exatas (sem faixas)
    try:
        mu1_e = float(w * lam.sum())             # contagem exata
        delta_e = s2 * mu1_e / (1.0 + s2 * mu1_e)
        ul_s, var_s, es_s = ponto_de_sela(u / L0, (1.0 - w) * lam,
                                          w * lam, alpha, delta_e, mu1_e,
                                          alphas)
        ul_s, sela, es_s = ul_s * L0, var_s * L0, es_s * L0
    except (ValueError, FloatingPointError):
        ul_s = np.nan
        sela = es_s = np.full(len(alphas), np.nan)
    info = {"L0": L0, "faixas": int(len(nus)), "metodo": metodo,
            "massa": massa, "n_max": n_max,
            "defaults_esperados": mu0 + mu1, "log_p0": float(log_p0),
            "el_exato": float((lam * u).sum()), "w": w, "v": v,
            "var_sela": {f"{a:.2%}": float(x) for a, x in zip(alphas, sela)},
            "var_exato": {f"{a:.2%}": float(x) for a, x in zip(alphas, var)},
            "pmf": f / massa,
            "sela": {"EL": float((lam * u).sum()), "UL": float(ul_s),
                     "VaR": np.asarray(sela, float),
                     "ES": np.asarray(es_s, float)}}
    return el, ul, var, es, info
