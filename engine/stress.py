"""Programa de testes de estresse de crédito (Res. CMN 4.557/2017, arts. 11
a 18): análise de sensibilidade, análise de cenários plurianual e teste de
estresse reverso, para um modelo de portfólio escolhido pelo usuário.

Convenções
----------
* Índice de crédito z (normal padrão): z < 0 é adverso. A PD do cenário
  desloca o limiar de Vasicek, recentrado para que z = 0 reproduza a PD de
  12 meses, PD(z) = Φ(Φ⁻¹(PD) − z·√(ρ/(1 − ρ))), com ρ a correlação de
  ativos calibrada ao ρ_D alvo da execução principal.
* Balanço constante: a carteira não default é reposta a cada ano com o
  mesmo perfil; os defaults de cada ano formam uma coorte que envelhece na
  curva de workout (ELBE) até a baixa.
* Perda do ano t (PE_t) = Σ PD_t·LGD_t·EAD (novos defaults, reconhecidos
  pela LGD de entrada) + incremento da provisão do estoque em default e das
  coortes de anos anteriores, que envelhecem 12 meses por ano. No ano 1
  sem choque, PE e PI coincidem com a consolidação da execução principal. Capital (PI_t) = VaR_t − EL_t do modelo escolhido
  com os parâmetros estressados + K_D do estoque em default. CEA = PE + PI.
"""
import io

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from scipy.optimize import brentq

from .pooled import _bucket_losses, regimes
from .inadimplidos import elbe as _elbe

MODELOS = ["Threshold Gaussiano 1F", "Threshold t-Student 1F",
           "ASRF (analítico)", "Beta-binomial", "CreditRisk+ (Poisson)"]
ALFAS = np.array([0.95, 0.99, 0.999, 0.9997])
COLS_TEMPLATE = ["cenario", "tipo", "ano", "setor", "z", "mult_pd",
                 "choque_lgd_pp", "mult_k_default", "pib_pct",
                 "desemprego_pct", "selic_pct", "ipca_pct", "descricao"]


# ----------------------------------------------------------- cenários
def cenarios_exemplo():
    """Cenários ilustrativos: base, adverso e severo (3 anos, choque por
    setor) e cinco sensibilidades de um parâmetro (1 ano). Os números são
    hipotéticos, para demonstração; não representam projeção oficial."""
    L = []

    def add(cen, tipo, ano, setor, z=None, mult=1.0, lgd=0.0, k=1.0,
            pib=None, des=None, sel=None, ipca=None, desc=""):
        L.append([cen, tipo, ano, setor, z, mult, lgd, k, pib, des, sel,
                  ipca, desc])

    macro = {"Base": [(2.0, 7.0, 10.50, 4.0), (2.2, 6.8, 9.75, 3.8),
                      (2.3, 6.6, 9.25, 3.5)],
             "Adverso": [(-1.5, 9.5, 14.25, 6.5), (-0.5, 10.5, 13.50, 5.8),
                         (1.0, 9.8, 12.00, 4.8)],
             "Severo": [(-4.0, 12.5, 17.75, 9.0), (-2.0, 13.5, 16.50, 7.5),
                        (0.5, 12.0, 14.00, 5.5)]}
    zs = {"Base": [None, None, None], "Adverso": [-0.5, -0.35, -0.15],
          "Severo": [-1.0, -0.7, -0.35]}
    sens_setor = {"Construção": 1.25, "Varejo": 1.15, "Agro": 1.10}
    for cen in ("Base", "Adverso", "Severo"):
        for a in range(3):
            pib, des, sel, ipca = macro[cen][a]
            z = zs[cen][a]
            add(cen, "cenario", a + 1, "todos", z, 1.0,
                0.0 if cen == "Base" else (5.0 if cen == "Adverso" else 10.0),
                1.0 if cen == "Base" else (1.5 if cen == "Adverso" else 2.0),
                pib, des, sel, ipca,
                {"Base": "Cenário de referência (ilustrativo)",
                 "Adverso": "Recessão moderada com juros altos por mais tempo (ilustrativo)",
                 "Severo": "Recessão profunda com choque de juros e de crédito (ilustrativo)"
                 }[cen])
            if cen != "Base":
                for s, f in sens_setor.items():
                    add(cen, "cenario", a + 1, s, z, f if cen == "Severo"
                        else 1 + (f - 1) / 2, None, None, pib, des, sel, ipca,
                        f"Choque adicional no setor {s}")
    for nome, kw in [("PD × 1,5", {"mult": 1.5}), ("PD × 2,0", {"mult": 2.0}),
                     ("LGD + 10 p.p.", {"lgd": 10.0}),
                     ("Índice de crédito z = −1", {"z": -1.0}),
                     ("Recuperação do default (k × 2)", {"k": 2.0})]:
        add(nome, "sensibilidade", 1, "todos", desc="Análise de sensibilidade "
            "de um parâmetro", **kw)
    return pd.DataFrame(L, columns=COLS_TEMPLATE)


def ler_cenarios(fonte):
    """Lê o template de cenários (CSV com vírgula ou ponto e vírgula)."""
    if isinstance(fonte, (bytes, bytearray)):
        fonte = io.BytesIO(fonte)
    df = pd.read_csv(fonte, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    falta = {"cenario", "ano"} - set(df.columns)
    if falta:
        raise ValueError(f"Template de cenários sem as colunas: {sorted(falta)}")
    for c in COLS_TEMPLATE:
        if c not in df.columns:
            df[c] = np.nan
    df["tipo"] = df["tipo"].fillna("cenario").astype(str).str.lower()
    df["setor"] = df["setor"].fillna("todos").astype(str)
    df["ano"] = df["ano"].astype(int)
    if (df["ano"] < 1).any() or (df["ano"] > 5).any():
        raise ValueError("O ano dos cenários deve ficar entre 1 e 5.")
    return df[COLS_TEMPLATE]


# ----------------------------------------------------------- choques
def pd_condicional(p, rho, z):
    """Vasicek: PD condicional a uma realização z do fator sistemático."""
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return norm.cdf((norm.ppf(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def pd_cenario(p, rho, z):
    """PD do cenário pelo deslocamento do índice de crédito: o limiar é
    recentrado para que z = 0 reproduza a PD de 12 meses da carteira,
    PD(z) = Φ(Φ⁻¹(PD) − z·√(ρ/(1 − ρ))); z < 0 é adverso."""
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return norm.cdf(norm.ppf(p) - np.asarray(z, float) * np.sqrt(rho / (1 - rho)))


def _choques_linha(cen_ano, setores):
    """Para cada linha da carteira, (z, mult_pd, choque_lgd, mult_k): a
    linha do setor tem precedência sobre 'todos'; campos vazios herdam."""
    G = len(setores)
    z = np.full(G, np.nan)
    mult = np.ones(G)
    lgd = np.zeros(G)
    k = np.ones(G)
    geral = cen_ano[cen_ano["setor"].str.lower() == "todos"]
    for df_ in (geral, cen_ano[cen_ano["setor"].str.lower() != "todos"]):
        for r in df_.itertuples(index=False):
            m = np.ones(G, bool) if str(r.setor).lower() == "todos" \
                else (setores == str(r.setor))
            if r.z == r.z and r.z is not None:
                z[m] = float(r.z)
            if r.mult_pd == r.mult_pd and r.mult_pd is not None:
                mult[m] = float(r.mult_pd)
            if r.choque_lgd_pp == r.choque_lgd_pp and r.choque_lgd_pp is not None:
                lgd[m] = float(r.choque_lgd_pp) / 100
            if r.mult_k_default == r.mult_k_default and r.mult_k_default is not None:
                k[m] = float(r.mult_k_default)
    return z, mult, lgd, k


def aplicar(p, lgd, setores, cen_ano, rho):
    """PD e LGD estressadas de um ano de cenário (vetores por linha)."""
    z, mult, dl, k = _choques_linha(cen_ano, setores)
    p_s = np.where(np.isnan(z), p, pd_cenario(p, rho, np.nan_to_num(z)))
    p_s = np.clip(p_s * mult, 1e-8, 0.999)
    lgd_s = np.clip(lgd + dl, 0.0, 1.0)
    return p_s, lgd_s, float(np.nanmean(k)) if len(k) else 1.0, z


# ----------------------------------------------------------- modelos
class Motor:
    """Distribuição de perdas do modelo escolhido para PD e LGD dadas."""

    def __init__(self, modelo, port, results, M=20_000, seed=42):
        if modelo not in MODELOS:
            raise ValueError(f"Modelo de estresse desconhecido: {modelo}")
        self.modelo, self.M, self.seed = modelo, int(M), seed
        df = port.df
        self.ead = port.ead
        n_eff = (df["n_efetivo"].to_numpy(float) if "n_efetivo" in df.columns
                 else np.ones(port.N))
        prm = results.get("parametros", {})
        self.rho_d = float(prm.get("rho_alvo", 0.05))
        self.n, self.lln, _ = regimes(n_eff, self.rho_d)
        self.n_eff = n_eff
        cg = results.get("calibracao_gaussiana", {})
        self.rho = float(cg.get("rho_ativos") or cg.get("rho") or 0.15)
        self.nu = float(prm.get("nu_t", 10.0))
        self.crp = (float(prm.get("crplus_w", 1.0)),
                    float(prm.get("crplus_v", 0.5)))

    def medidas(self, p, lgd):
        """(EL, VaR[alfas], ES[alfas])."""
        C = self.ead * lgd
        if self.modelo == "ASRF (analítico)":
            return self._asrf(p, C)
        rng = np.random.default_rng(self.seed)
        n, lln = self.n, self.lln
        cbar = C / n
        M, B = self.M, 20_000
        loss = np.empty(M)
        G = len(p)
        if self.modelo == "Beta-binomial":
            import mixtureModels as mix
            pbar = float(np.dot(C / C.sum(), p))
            a, b = mix.betaCalibrate(pbar, self.rho_d)
        if self.modelo == "CreditRisk+ (Poisson)":
            # (w, v) recalibrados ao ρ_D alvo na PD média estressada, como
            # na execução principal (w = 1, v resolvido)
            from .crplus_csfp import calibrar_crplus
            pbar = float(np.dot(C / C.sum(), p))
            try:
                v_, _, _ = calibrar_crplus(pbar, self.rho_d, 1.0)
                crp_wv = (1.0, float(v_))
            except Exception:
                crp_wv = self.crp
        Kg = norm.ppf(np.clip(p, 1e-10, 1 - 1e-10))
        Kt = student_t.ppf(np.clip(p, 1e-10, 1 - 1e-10), self.nu)
        for i0 in range(0, M, B):
            m = min(B, M - i0)
            if self.modelo == "Threshold Gaussiano 1F":
                g = rng.normal(0, 1, (m, 1))
                pc = norm.cdf((Kg[None, :] - np.sqrt(self.rho) * g)
                              / np.sqrt(1 - self.rho))
            elif self.modelo == "Threshold t-Student 1F":
                g = rng.normal(0, 1, (m, 1))
                V = rng.chisquare(self.nu, (m, 1))
                pc = norm.cdf((Kt[None, :] * np.sqrt(V / self.nu)
                               - np.sqrt(self.rho) * g) / np.sqrt(1 - self.rho))
            elif self.modelo == "Beta-binomial":
                Z = rng.beta(a, b, (m, 1))
                pc = np.broadcast_to(Z, (m, G))
            else:                                   # CreditRisk+ (Poisson)
                w, v = crp_wv
                S = rng.gamma(v, 1.0 / v, (m, 1))
                lam = (self.n_eff * p) * (1.0 - w + w * S)
                loss[i0:i0 + m] = (rng.poisson(lam) * (C / self.n_eff)).sum(1)
                continue
            loss[i0:i0 + m] = _bucket_losses(pc.astype(np.float32), C, n,
                                             cbar, rng, lln).sum(axis=1)
        loss.sort()
        el = float(loss.mean())
        var = np.quantile(loss, ALFAS)
        es = np.array([loss[loss >= v].mean() for v in var])
        return el, var, es

    def _asrf(self, p, C):
        # mesma forma da execução principal: carteira homogênea com a PD
        # média ponderada pela severidade (Bolder, asrfModel)
        Ct = float(C.sum())
        pbar = float(np.dot(C / Ct, p))
        el = pbar * Ct
        rho = self.rho

        def q(a):
            return Ct * float(pd_condicional(pbar, rho, norm.ppf(1 - a)))
        var = np.array([q(a) for a in ALFAS])
        es = []
        for a in ALFAS:
            u = a + (1 - a) * (np.arange(200) + 0.5) / 200
            es.append(np.mean([q(x) for x in u]))
        return el, var, np.array(es)


# ----------------------------------------------------------- projeção
def _default_existente(ad_tab, k_base, mult_k, choque_lgd, anos,
                       curva=None, prazo_baixa=24.0):
    """ΔL e K_D do estoque em default em cada ano, com o choque na LGD de
    entrada e na margem k (envelhecimento de 12 meses por ano)."""
    if ad_tab is None or not len(ad_tab):
        return np.zeros(anos), np.zeros(anos)
    t0 = ad_tab["meses_em_default"].to_numpy(float)
    lgd0 = np.clip(ad_tab["lgd_nao_default"].to_numpy(float) + choque_lgd,
                   0, 1)
    seg = ad_tab["segmento_irb"].astype(str).to_numpy() \
        if "segmento_irb" in ad_tab.columns else np.array(["todos"] * len(t0))
    ead = ad_tab["exposicao"].to_numpy(float)
    dl, kd = np.zeros(anos), np.zeros(anos)
    for y in range(anos):
        e0 = _elbe(t0 + 12 * y, lgd0, seg, curva, prazo_baixa)
        e1 = _elbe(t0 + 12 * (y + 1), lgd0, seg, curva, prazo_baixa)
        dl[y] = float(np.dot(ead, np.clip(e1 - e0, 0, None)))
        kd[y] = float(np.dot(ead, k_base * mult_k[y] * (1 - np.clip(e1, 0, 1))))
    return dl, kd


def projetar(motor, port, cen, rho, ad=None, curva=None, prazo_baixa=24.0,
             k_base=0.10, base_irb=None):
    """Projeção ano a ano de um cenário: PE, PI, CEA e capital IRB."""
    from .irb_bcb import coeficiente_k, segmentos_da_carteira, \
        receita_da_carteira
    df = port.df
    setores = df["setor"].astype(str).to_numpy() if "setor" in df.columns \
        else np.array(["todos"] * port.N)
    p0, lgd0, ead = port.p, port.lgd, port.ead
    anos = int(cen["ano"].max())
    seg_irb = segmentos_da_carteira(df)
    rec = receita_da_carteira(df)
    tab = (ad or {}).get("tabela")
    choques = [aplicar(p0, lgd0, setores, cen[cen["ano"] == y + 1], rho)
               for y in range(anos)]
    mult_k = [c[2] for c in choques]
    dlg = float(np.mean(choques[0][1] - lgd0)) if anos else 0.0
    dl_ex, kd_ex = _default_existente(tab, k_base, mult_k, dlg, anos, curva,
                                      prazo_baixa)
    coortes = []          # (ano de entrada, ead, lgd0 por linha)
    out = []
    for y in range(anos):
        p_s, lgd_s, _, z = choques[y]
        el, var, es = motor.medidas(p_s, lgd_s)
        novos = p_s * ead
        pe_nd = float(np.dot(novos, lgd_s))
        # coortes de anos anteriores: incremento da provisão e K_D
        dl_c, kd_c = 0.0, 0.0
        for (y0, e_c, l_c) in coortes:
            a0, a1 = 12 * (y - y0 - 1), 12 * (y - y0)
            s_ = np.array(["todos"] * len(l_c))
            e_a0 = _elbe(np.full(len(l_c), a0), l_c, s_, None, prazo_baixa)
            e_a1 = _elbe(np.full(len(l_c), a1), l_c, s_, None, prazo_baixa)
            dl_c += float(np.dot(e_c, np.clip(e_a1 - e_a0, 0, None)))
            kd_c += float(np.dot(e_c, k_base * mult_k[y] * (1 - e_a1)))
        coortes.append((y, novos, lgd_s))
        K, _, _ = coeficiente_k(p_s, port.tenor, seg_irb, rec, 0.999)
        irb = float(np.dot(ead * lgd_s, K))
        pe = pe_nd + dl_ex[y] + dl_c
        pi_nd = float(var[2] - el)
        kd = kd_ex[y] + kd_c
        out.append({
            "ano": y + 1, "pd_media": float(np.dot(ead, p_s) / ead.sum()),
            "lgd_media": float(np.dot(ead, lgd_s) / ead.sum()),
            "z_medio": float(np.nanmean(z)) if np.isfinite(z).any() else np.nan,
            "novos_defaults_ead": float(novos.sum()),
            "PE_ND": pe_nd, "PE_D": dl_ex[y] + dl_c, "PE": pe,
            "EL_modelo": el, "VaR_ND": float(var[2]), "ES_ND": float(es[2]),
            "PI_ND": pi_nd, "PI_ES_ND": float(es[2] - el), "K_D": kd,
            "PI": pi_nd + kd, "CEA": pe + pi_nd + kd,
            "CEA_ES": pe + float(es[2] - el) + kd,
            "IRB_K": irb, "IRB_total": irb + kd})
    return pd.DataFrame(out)


def executar(port, results, cenarios, modelo="Threshold Gaussiano 1F",
             M=20_000, seed=42, ad=None, curva=None, prazo_baixa=24.0,
             k_base=0.10, alvo_reverso=1.5, progress=None):
    """Roda o programa completo: cenários plurianuais, sensibilidades e
    estresse reverso. Devolve um dicionário com tabelas e metadados."""
    motor = Motor(modelo, port, results, M=M, seed=seed)
    rho = motor.rho

    def tick(m):
        if progress:
            progress(m)

    cen = cenarios.copy()
    res = {"modelo": modelo, "M": int(M), "rho_ativos": rho,
           "rho_d": motor.rho_d, "cenarios_def": cen,
           "origem": "cenários informados pelo usuário"}
    set_cart = set(port.df["setor"].astype(str)) if "setor" in port.df.columns \
        else set()
    res["setores_sem_carteira"] = sorted(
        s for s in cen["setor"].astype(str).unique()
        if s.lower() != "todos" and s not in set_cart)
    # reprodução da base (sem choque, 1 ano)
    tick("Estresse: cenário base...")
    base_1 = pd.DataFrame([{"cenario": "_base", "tipo": "cenario", "ano": 1,
                            "setor": "todos", "z": np.nan, "mult_pd": 1.0,
                            "choque_lgd_pp": 0.0, "mult_k_default": 1.0}])
    for c in COLS_TEMPLATE:
        if c not in base_1.columns:
            base_1[c] = np.nan
    b = projetar(motor, port, base_1, rho, ad, curva, prazo_baixa, k_base)
    res["base"] = b.iloc[0].to_dict()
    # cenários
    proj = []
    for nome in cen.loc[cen["tipo"] == "cenario", "cenario"].unique():
        tick(f"Estresse: cenário {nome}...")
        d = projetar(motor, port, cen[cen["cenario"] == nome], rho, ad,
                     curva, prazo_baixa, k_base)
        d.insert(0, "cenario", nome)
        proj.append(d)
    res["cenarios"] = pd.concat(proj, ignore_index=True) if proj else None
    # sensibilidades
    sens = []
    for nome in cen.loc[cen["tipo"] == "sensibilidade", "cenario"].unique():
        tick(f"Estresse: sensibilidade {nome}...")
        d = projetar(motor, port, cen[(cen["cenario"] == nome)
                                      & (cen["ano"] == 1)], rho, ad, curva,
                     prazo_baixa, k_base)
        r = d.iloc[0].to_dict()
        r["cenario"] = nome
        sens.append(r)
    res["sensibilidades"] = pd.DataFrame(sens) if sens else None
    # estresse reverso: z uniforme que leva o CEA a alvo × CEA base
    tick("Estresse reverso...")
    alvo = alvo_reverso * res["base"]["CEA"]

    def cea_z(zv):
        c1 = base_1.copy()
        c1["z"] = zv
        return projetar(motor, port, c1, rho, ad, curva, prazo_baixa,
                        k_base).iloc[0]["CEA"] - alvo
    try:
        lo, hi = -6.0, 0.0
        if cea_z(lo) < 0:
            zr = np.nan
        else:
            zr = brentq(cea_z, lo, hi, xtol=1e-3)
    except Exception:
        zr = np.nan
    pd_r = np.nan
    if zr == zr:
        p_r = pd_cenario(port.p, rho, zr)
        pd_r = float(np.dot(port.ead, p_r) / port.ead.sum())
    res["reverso"] = {"alvo_mult": float(alvo_reverso), "alvo": float(alvo),
                      "z": float(zr) if zr == zr else np.nan,
                      "prob": float(norm.cdf(zr)) if zr == zr else np.nan,
                      "pd_media": pd_r,
                      "pd_media_base": float(np.dot(port.ead, port.p)
                                             / port.ead.sum())}
    res["ead_total"] = float(port.ead.sum()) + float(
        (((ad or {}).get("resumo")) or {}).get("ead", 0.0) or 0.0)
    return res


_PREFIXO = {"Threshold Gaussiano 1F": "Threshold Gaussiano 1F",
            "Threshold t-Student 1F": "Threshold t-Student 1F",
            "ASRF (analítico)": "ASRF",
            "Beta-binomial": "Beta-binomial",
            "CreditRisk+ (Poisson)": "CreditRisk+ simulação Poisson"}


def reproducao(results):
    """Compara o ano 1 do cenário base do estresse com o CEA da execução
    principal para o mesmo modelo (diagnóstico de consistência)."""
    est = results.get("estresse")
    if not est:
        return None
    from .fluxo_cea import tabela_cea
    tv, _ = tabela_cea(results, "VaR")
    if tv is None:
        return None
    pref = _PREFIXO.get(est["modelo"], est["modelo"])
    m = tv[tv["modelo"].str.startswith(pref)]
    if not len(m):
        return None
    cea_p = float(m.iloc[0]["CEA"])
    cea_e = float(est["base"]["CEA"])
    return {"modelo_principal": str(m.iloc[0]["modelo"]), "cea_principal": cea_p,
            "cea_estresse": cea_e, "dif": cea_e / cea_p - 1}


def tabela_cenarios(est):
    """Cenários × ano com a variação do CEA contra o cenário base do mesmo
    ano (ou o ano 1 do base sem choque, se o arquivo não trouxer 'Base')."""
    d = est.get("cenarios")
    if d is None or not len(d):
        return None
    d = d.copy()
    nomes = d["cenario"].astype(str).str.lower()
    base = d[nomes == "base"].set_index("ano")["CEA"]
    ref = d["ano"].map(base) if len(base) else est["base"]["CEA"]
    d["dCEA"] = d["CEA"] - ref
    d["dCEA_pct"] = d["CEA"] / ref - 1
    d["CEA_pct_ead"] = d["CEA"] / est["ead_total"]
    d["PE_acum"] = d.groupby("cenario")["PE"].cumsum()
    return d


def tabela_sensibilidades(est):
    s = est.get("sensibilidades")
    if s is None or not len(s):
        return None
    s = s.copy()
    b = est["base"]["CEA"]
    s["dCEA"] = s["CEA"] - b
    s["dCEA_pct"] = s["CEA"] / b - 1
    return s.sort_values("dCEA", ascending=False).reset_index(drop=True)
