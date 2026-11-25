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
           "ASRF (analítico)", "Beta-binomial", "CreditRisk+ (Poisson)",
           "CreditRisk+ clássico (CSFP)", "CreditRisk+ ponto de sela"]
_CRP = ("CreditRisk+ (Poisson)", "CreditRisk+ clássico (CSFP)",
        "CreditRisk+ ponto de sela")
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


SELETORES = ["setor", "pais", "regiao", "segmento_irb", "rating"]
_CHOQUES = [("z", "z", None), ("mult_pd", "mult", None),
            ("choque_lgd_pp", "lgd", 0.01), ("choque_ead_pct", "ead", 0.01),
            ("mult_k_default", "k", None)]


def _vazio(v):
    return v is None or (isinstance(v, float) and v != v) or \
        str(v).strip().lower() in ("", "todos", "nan", "none")


def _choques_linha(cen_ano, df):
    """Choques por linha da carteira. Cada linha do cenário seleciona a
    fatia da carteira pelas colunas de SELETORES preenchidas (vazio ou
    'todos' = qualquer valor); a linha mais específica, com mais seletores,
    prevalece e herda os campos que deixar vazios. Devolve um dicionário
    de vetores (z, mult, lgd, ead, k)."""
    G = len(df)
    out = {"z": np.full(G, np.nan), "mult": np.ones(G), "lgd": np.zeros(G),
           "ead": np.zeros(G), "k": np.ones(G)}
    sel = [c for c in SELETORES if c in cen_ano.columns]
    linhas = list(cen_ano.itertuples(index=False))
    linhas.sort(key=lambda r: sum(not _vazio(getattr(r, c)) for c in sel))
    for r in linhas:
        m = np.ones(G, bool)
        for c in sel:
            v = getattr(r, c)
            if _vazio(v):
                continue
            if c not in df.columns:
                m[:] = False
                break
            m &= df[c].astype(str).str.strip().str.lower().to_numpy() == \
                str(v).strip().lower()
        if not m.any():
            continue
        for col, chave, esc in _CHOQUES:
            v = getattr(r, col, None)
            if v is None or v != v:
                continue
            out[chave][m] = float(v) * (esc or 1.0)
    return out


def aplicar(port, cen_ano, rho):
    """PD, LGD e EAD estressadas de um ano de cenário (vetores por linha)."""
    ch = _choques_linha(cen_ano, port.df)
    z = ch["z"]
    p = port.p
    p_s = np.where(np.isnan(z), p, pd_cenario(p, rho, np.nan_to_num(z)))
    p_s = np.clip(p_s * ch["mult"], 1e-8, 0.999)
    lgd_s = np.clip(port.lgd + ch["lgd"], 0.0, 1.0)
    ead_s = port.ead * np.clip(1.0 + ch["ead"], 0.0, None)
    return p_s, lgd_s, ead_s, float(np.mean(ch["k"])) if len(z) else 1.0, z


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

    def medidas(self, p, lgd, ead=None):
        """(EL, VaR[alfas], ES[alfas])."""
        C = (self.ead if ead is None else ead) * lgd
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
        if self.modelo in _CRP:
            # (w, v) recalibrados ao ρ_D alvo na PD média estressada, como
            # na execução principal (w = 1, v resolvido)
            from .crplus_csfp import calibrar_crplus
            pbar = float(np.dot(C / C.sum(), p))
            try:
                v_, _, _ = calibrar_crplus(pbar, self.rho_d, 1.0)
                crp_wv = (1.0, float(v_))
            except Exception:
                crp_wv = self.crp
        if self.modelo in _CRP[1:]:
            return self._csfp(p, C, crp_wv)
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

    def _csfp(self, p, C, crp_wv):
        """CreditRisk+ clássico: distribuição exata por faixas (Panjer ou
        FFT) ou aproximação de ponto de sela sobre as severidades exatas."""
        from .crplus_csfp import crplus_csfp
        w, v = crp_wv
        el, _, var, es, info = crplus_csfp(p, C, w, v, ALFAS, n=self.n_eff)
        if self.modelo == "CreditRisk+ ponto de sela":
            s = info.get("sela") or {}
            vs, es_s = np.asarray(s.get("VaR", []), float), \
                np.asarray(s.get("ES", []), float)
            if len(vs) == len(ALFAS) and np.all(np.isfinite(vs)):
                var = vs
                es = np.where(np.isfinite(es_s), es_s, es)
            el = float(s.get("EL", el))
        return float(el), np.asarray(var, float), np.asarray(es, float)

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
    p0, lgd0 = port.p, port.lgd
    anos = int(cen["ano"].max())
    seg_irb = segmentos_da_carteira(df)
    rec = receita_da_carteira(df)
    tab = (ad or {}).get("tabela")
    choques = [aplicar(port, cen[cen["ano"] == y + 1], rho)
               for y in range(anos)]
    mult_k = [c[3] for c in choques]
    dlg = float(np.mean(choques[0][1] - lgd0)) if anos else 0.0
    dl_ex, kd_ex = _default_existente(tab, k_base, mult_k, dlg, anos, curva,
                                      prazo_baixa)
    coortes = []          # (ano de entrada, ead, lgd0 por linha)
    out = []
    for y in range(anos):
        p_s, lgd_s, ead, _, z = choques[y]
        el, var, es = motor.medidas(p_s, lgd_s, ead)
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
            "ead_nd": float(ead.sum()),
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
    res["setores_sem_carteira"] = seletores_ausentes(cen, port.df)
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
    res["tipo"] = "regulatorio"
    res["ead_total"] = float(port.ead.sum()) + float(
        (((ad or {}).get("resumo")) or {}).get("ead", 0.0) or 0.0)
    if alvo_reverso is None:
        res["reverso"] = {}
        return res
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
            zr = brentq(cea_z, lo, hi, xtol=5e-3)
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



# ----------------------------------------------------------- verificação
def seletores_ausentes(cen, df):
    """Valores de seletores do cenário que não existem na carteira (o choque
    não tem efeito) e colunas de seletor ausentes da carteira."""
    out = []
    for c in SELETORES:
        if c not in cen.columns:
            continue
        vals = {str(v).strip() for v in cen[c] if not _vazio(v)}
        if not vals:
            continue
        if c not in df.columns:
            out.append(f"coluna '{c}' ausente da carteira agregada")
            continue
        cart = {str(v).strip().lower() for v in df[c].astype(str)}
        out += [f"{c} = {v}" for v in sorted(vals) if v.lower() not in cart]
    return out


# ----------------------------------------------------------- customizado
COLS_CUSTOM = ["cenario", "modo", "ano", "setor", "pais", "regiao",
               "segmento_irb", "rating", "z", "mult_pd", "choque_lgd_pp",
               "choque_ead_pct", "mult_k_default", "objetivo", "fonte",
               "metodo_depara", "premissas", "descricao"]
_DOC = ["objetivo", "fonte", "metodo_depara", "premissas", "descricao"]


def ler_customizado(fonte):
    """Lê o template de estresse customizado. 'modo' = 'instantaneo' (cada
    horizonte é um choque sobre a carteira atual; 'ano' pode ser o ano do
    horizonte, p.ex. 2030) ou 'projecao' (anos 1 a 5 em sequência, com
    balanço constante, como no programa regulatório)."""
    if isinstance(fonte, (bytes, bytearray)):
        fonte = io.BytesIO(fonte)
    df = pd.read_csv(fonte, sep=None, engine="python") \
        if not isinstance(fonte, pd.DataFrame) else fonte.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    falta = {"cenario", "ano"} - set(df.columns)
    if falta:
        raise ValueError(f"Template customizado sem as colunas: {sorted(falta)}")
    for c in COLS_CUSTOM:
        if c not in df.columns:
            df[c] = np.nan
    df["modo"] = df["modo"].fillna("instantaneo").astype(str).str.lower() \
        .str.replace("â", "a").str.replace("ç", "c").str.replace("ã", "a")
    if not df["modo"].isin(["instantaneo", "projecao"]).all():
        raise ValueError("A coluna 'modo' aceita 'instantaneo' ou 'projecao'.")
    df["ano"] = df["ano"].astype(int)
    proj = df["modo"] == "projecao"
    if ((df.loc[proj, "ano"] < 1) | (df.loc[proj, "ano"] > 5)).any():
        raise ValueError("No modo 'projecao', o ano vai de 1 a 5.")
    for c in SELETORES:
        df[c] = df[c].where(df[c].notna(), "todos").astype(str)
    return df[COLS_CUSTOM]


def _pe_por(port, p_s, lgd_s, ead_s, chave="setor"):
    if chave not in port.df.columns:
        return {}
    g = pd.Series(p_s * lgd_s * ead_s).groupby(
        port.df[chave].astype(str).to_numpy()).sum()
    return g.to_dict()


def executar_customizado(port, results, cenarios, nome="Estresse customizado",
                         modelo="Threshold Gaussiano 1F", M=20_000, seed=42,
                         ad=None, curva=None, prazo_baixa=24.0, k_base=0.10,
                         progress=None):
    """Estresse customizado (p.ex. climático): choques por seletores livres,
    em modo instantâneo (cada horizonte sobre a carteira atual) ou de
    projeção. Devolve tabelas de resultado e a documentação do exercício."""
    motor = Motor(modelo, port, results, M=M, seed=seed)
    rho = motor.rho

    def tick(m):
        if progress:
            progress(m)

    cen = ler_customizado(cenarios)
    res = {"tipo": "customizado", "nome": nome, "modelo": modelo,
           "M": int(M), "rho_ativos": rho, "rho_d": motor.rho_d,
           "cenarios_def": cen, "origem": "cenários informados pelo usuário",
           "seletores_ausentes": seletores_ausentes(cen, port.df)}
    tick("Estresse customizado: base...")
    vazio = pd.DataFrame([{c: np.nan for c in COLS_CUSTOM}])
    vazio["ano"] = 1
    for c in SELETORES:
        vazio[c] = "todos"
    b = projetar(motor, port, vazio, rho, ad, curva, prazo_baixa, k_base)
    res["base"] = b.iloc[0].to_dict()
    pe_base = _pe_por(port, port.p, port.lgd, port.ead)
    linhas, setor = [], []
    doc = {}
    for nome_c in cen["cenario"].unique():
        c = cen[cen["cenario"] == nome_c]
        doc[nome_c] = {k: next((str(v) for v in c[k] if not _vazio(v)), "")
                       for k in _DOC}
        modo = c["modo"].iloc[0]
        tick(f"Estresse customizado: {nome_c}...")
        if modo == "projecao":
            d = projetar(motor, port, c, rho, ad, curva, prazo_baixa, k_base)
            d.insert(0, "horizonte", d["ano"])
        else:
            partes = []
            for h in sorted(c["ano"].unique()):
                ch = c[c["ano"] == h].copy()
                ch["ano"] = 1
                r1 = projetar(motor, port, ch, rho, ad, curva, prazo_baixa,
                              k_base).iloc[0].to_dict()
                r1["horizonte"] = int(h)
                partes.append(r1)
                p_s, lgd_s, ead_s, _, _ = aplicar(port, ch, rho)
                for s_, v in _pe_por(port, p_s, lgd_s, ead_s).items():
                    setor.append({"cenario": nome_c, "horizonte": int(h),
                                  "setor": s_, "PE": v,
                                  "PE_base": pe_base.get(s_, np.nan)})
            d = pd.DataFrame(partes)
        d.insert(0, "modo", modo)
        d.insert(0, "cenario", nome_c)
        linhas.append(d)
    t = pd.concat(linhas, ignore_index=True)
    t["dCEA"] = t["CEA"] - res["base"]["CEA"]
    t["dCEA_pct"] = t["CEA"] / res["base"]["CEA"] - 1
    res["resultados"] = t
    res["por_setor"] = pd.DataFrame(setor)
    res["documentacao"] = doc
    res["ead_total"] = float(port.ead.sum()) + float(
        (((ad or {}).get("resumo")) or {}).get("ead", 0.0) or 0.0)
    return res


def cenarios_climaticos_exemplo():
    """Exemplo ilustrativo de estresse climático no formato NGFS: três
    narrativas (Net Zero 2050, Delayed Transition, Current Policies) em
    choque instantâneo nos horizontes 2030, 2040 e 2050. Transição: PD por
    setor e país; físico: PD e LGD por região e setor. Multiplicadores
    hipotéticos, para demonstração; não são cenários oficiais do NGFS."""
    obj = ("Avaliar a sensibilidade do capital de crédito aos riscos "
           "climáticos de transição e físico (Res. CMN 4.557, arts. 38-C e "
           "38-D, VII).")
    fonte = ("Narrativas inspiradas nos cenários do NGFS; multiplicadores "
             "hipotéticos e ilustrativos, sem relação com valores oficiais.")
    met = ("Transição: multiplicador de PD por setor e país, crescente com a "
           "intensidade de carbono do setor. Físico: multiplicador de PD e "
           "choque de LGD por região e setor. Em produção, derivar de modelos "
           "satélite sobre as variáveis do NGFS (preço de carbono, valor "
           "adicionado setorial, PIB).")
    prem = ("Choque instantâneo sobre a carteira atual em cada horizonte; "
            "balanço estático; correlações e curva de recuperação do default "
            "inalteradas.")
    L = []

    def add(cen, h, setor="todos", pais="todos", regiao="todos", z=None,
            mult=None, lgd=None, desc=""):
        L.append({"cenario": cen, "modo": "instantaneo", "ano": h,
                  "setor": setor, "pais": pais, "regiao": regiao,
                  "segmento_irb": "todos", "rating": "todos", "z": z,
                  "mult_pd": mult, "choque_lgd_pp": lgd,
                  "choque_ead_pct": None, "mult_k_default": None,
                  "objetivo": obj, "fonte": fonte, "metodo_depara": met,
                  "premissas": prem, "descricao": desc})
    trans = {  # setor: multiplicadores por horizonte (2030, 2040, 2050)
        "Net Zero 2050": {"Petroquímica": (1.60, 1.40, 1.20),
                          "Energia": (1.45, 1.25, 1.10),
                          "Indústria": (1.20, 1.10, 1.05),
                          "Logística": (1.15, 1.08, 1.03),
                          "Construção": (1.10, 1.05, 1.00)},
        "Delayed Transition": {"Petroquímica": (1.00, 2.20, 1.60),
                               "Energia": (1.00, 1.90, 1.40),
                               "Indústria": (1.00, 1.35, 1.15),
                               "Logística": (1.00, 1.30, 1.10),
                               "Construção": (1.00, 1.20, 1.05)}}
    desc = {"Net Zero 2050": "Transição ordenada: preço de carbono alto e "
            "imediato; choque concentrado em 2030 e decrescente.",
            "Delayed Transition": "Transição desordenada: políticas só a "
            "partir de 2030, com choque abrupto em 2040 e efeito "
            "macroeconômico.",
            "Current Policies": "Sem transição adicional: o risco físico "
            "cresce ao longo do tempo (secas, cheias, calor extremo)."}
    for cen, tab in trans.items():
        for i, h in enumerate((2030, 2040, 2050)):
            add(cen, h, z=(-0.30 if cen == "Delayed Transition" and h == 2040
                           else None), desc=desc[cen])
            for s, mult in tab.items():
                if mult[i] != 1.0:
                    add(cen, h, setor=s, mult=mult[i], desc=desc[cen])
            if cen == "Delayed Transition" and h == 2040:
                add(cen, h, setor="Energia", pais="Argentina", mult=2.40,
                    desc="Exposição de energia na Argentina, mais dependente "
                    "de fósseis")
                add(cen, h, setor="Petroquímica", pais="México", mult=2.50,
                    desc="Petroquímica no México, maior intensidade de carbono")
        add(cen, 2050, setor="Agro", regiao="Centro-Oeste", mult=1.05,
            desc="Risco físico residual")
    fis = {2030: (1.10, 3.0, 2.0), 2040: (1.30, 6.0, 5.0),
           2050: (1.60, 10.0, 8.0)}
    for h, (m_agro, l_agro, l_sul) in fis.items():
        add("Current Policies", h, z=(-0.20 if h == 2050 else None),
            desc=desc["Current Policies"])
        for r in ("Centro-Oeste", "Nordeste"):
            add("Current Policies", h, setor="Agro", regiao=r, mult=m_agro,
                lgd=l_agro, desc="Seca e calor extremo sobre a produção "
                "agrícola")
        add("Current Policies", h, regiao="Sul", lgd=l_sul,
            desc="Cheias: dano às garantias reais na região")
        add("Current Policies", h, setor="Agro", regiao="Sul",
            mult=1 + (m_agro - 1) / 2, lgd=l_sul,
            desc="Cheias e secas alternadas sobre o agro do Sul")
    return pd.DataFrame(L)[COLS_CUSTOM]

_PREFIXO = {"Threshold Gaussiano 1F": "Threshold Gaussiano 1F",
            "Threshold t-Student 1F": "Threshold t-Student 1F",
            "ASRF (analítico)": "ASRF",
            "Beta-binomial": "Beta-binomial",
            "CreditRisk+ (Poisson)": "CreditRisk+ simulação Poisson",
            "CreditRisk+ clássico (CSFP)": "CreditRisk+ clássico (CSFP)",
            "CreditRisk+ ponto de sela": "CreditRisk+ ponto de sela"}


def runs(results):
    """Execuções de estresse guardadas (etapa 2), da mais antiga à mais
    recente; aceita o formato antigo, com uma execução em 'estresse'."""
    r = list(results.get("estresse_runs") or [])
    if not r and results.get("estresse"):
        r = [results["estresse"]]
    return r


def ultimas(results, tipo):
    """Última execução de cada modelo (regulatório) ou de cada par
    nome × modelo (customizado), na ordem de execução."""
    vistos, out = set(), []
    for r in reversed(runs(results)):
        if r.get("tipo", "regulatorio") != tipo:
            continue
        k = (r.get("nome", ""), r["modelo"])
        if k in vistos:
            continue
        vistos.add(k)
        out.append(r)
    return out[::-1]


def comparacao_modelos(results):
    """CEA do programa regulatório por cenário × ano em cada modelo (última
    execução de cada modelo): a sensibilidade ao risco de modelo."""
    L = []
    for r in ultimas(results, "regulatorio"):
        tc = tabela_cenarios(r)
        if tc is None:
            continue
        for x in tc.itertuples(index=False):
            L.append({"modelo": r["modelo"], "cenario": x.cenario,
                      "ano": int(x.ano), "CEA": x.CEA,
                      "dCEA_pct": x.dCEA_pct})
    if not L:
        return None
    return pd.DataFrame(L)


def reproducao(results, est=None):
    """Compara o ano 1 do cenário base do estresse com o CEA da execução
    principal para o mesmo modelo (diagnóstico de consistência)."""
    if est is None:
        r_ = runs(results)
        est = r_[-1] if r_ else None
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
