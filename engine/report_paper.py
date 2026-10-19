"""
Paper de validação de modelos — relatório PDF da análise executada.

Gera, a partir dos resultados da simulação, um documento no formato de
paper de validação: dados de entrada e tratamento, fundamentação teórica
de cada modelo efetivamente executado (com a formulação e os parâmetros
escolhidos na rodada), resultados comparados e referências bibliográficas
auditáveis. Construído com matplotlib (PdfPages) — sem dependências além
das já usadas pelo aplicativo.
"""
import datetime as dt
import textwrap

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

AZUL = "#12395B"
AZUL2 = "#2E6DA4"
VERM = "#B02418"
CINZA = "#5B6770"
FUNDO = "#F4F6F8"
PALETA = {"Binomial independente": "#9aa0a6",
          "Beta-binomial": "#6a5acd",
          "CreditRisk+": "#0e8a8a",
          "CreditRisk+ simulação Bernoulli": "#7fc8c8",
          "CreditRisk+ simulação Poisson": "#0e8a8a",
          "CreditRisk+ clássico (CSFP)": "#075e5e",
          "CreditRisk+ ponto de sela": "#3fa7a0",
          "Threshold Gaussiano 1F": "#1f4e79",
          "Threshold t-Student 1F": "#b02418",
          "Threshold Gaussiano MF": "#2e8b57",
          "Threshold t-Student MF": "#e07b00"}
A4 = (8.27, 11.69)

REFS = [
    "[1] VASICEK, O. A. (2002). The distribution of loan portfolio value. "
    "Risk, 15(12), 160-162.",
    "[2] GORDY, M. B. (2003). A risk-factor model foundation for "
    "ratings-based bank capital rules. Journal of Financial "
    "Intermediation, 12(3), 199-232.",
    "[3] BASEL COMMITTEE ON BANKING SUPERVISION (2005). An Explanatory "
    "Note on the Basel II IRB Risk Weight Functions. Basileia: BIS.",
    "[4] BASEL COMMITTEE ON BANKING SUPERVISION (2006). International "
    "Convergence of Capital Measurement and Capital Standards: A Revised "
    "Framework. Basileia: BIS.",
    "[5] GOURIEROUX, C.; LAURENT, J.-P.; SCAILLET, O. (2000). Sensitivity "
    "analysis of Values at Risk. Journal of Empirical Finance, 7(3-4), "
    "225-245.",
    "[6] MARTIN, R.; WILDE, T. (2002). Unsystematic credit risk. Risk, "
    "15(11), 123-128.",
    "[7] GORDY, M. B.; LÜTKEBOHMERT, E. (2013). Granularity adjustment "
    "for regulatory capital assessment. International Journal of Central "
    "Banking, 9(3), 33-71.",
    "[8] CREDIT SUISSE FIRST BOSTON (1997). CreditRisk+: A Credit Risk "
    "Management Framework. Londres: CSFB.",
    "[9] TASCHE, D. (2008). Capital allocation to business units and "
    "sub-portfolios: the Euler principle. In: RESTI, A. (org.). Pillar II "
    "in the New Basel Accord. Londres: Risk Books, 423-453.",
    "[10] BELKIN, B.; SUCHOWER, S.; FOREST JR., L. R. (1998). A "
    "one-parameter representation of credit risk and transition matrices. "
    "CreditMetrics Monitor, 3o tri., 46-56.",
    "[11] FREY, R.; McNEIL, A. J. (2003). Dependent defaults in models of "
    "portfolio credit risk. Journal of Risk, 6(1), 59-92.",
    "[12] MERTON, R. C. (1974). On the pricing of corporate debt: the "
    "risk structure of interest rates. Journal of Finance, 29(2), "
    "449-470.",
    "[13] BOLDER, D. J. (2018). Credit-Risk Modelling: Theoretical "
    "Foundations, Diagnostic Tools, Practical Examples, and Numerical "
    "Recipes in Python. Cham: Springer. [Referencial de implementação "
    "deste aplicativo.]",
    "[14] IFRS FOUNDATION (2014). IFRS 9 - Financial Instruments. "
    "Londres. / CONSELHO MONETÁRIO NACIONAL (2021). Resolução no 4.966, "
    "de 25 de novembro de 2021. Brasília: Banco Central do Brasil.",
    "[15] BANCO CENTRAL DO BRASIL (2023). Resolução BCB no 303, de 16 de "
    "março de 2023: cálculo da parcela dos ativos ponderados pelo risco "
    "relativa às exposições ao risco de crédito sob abordagem IRB "
    "(revoga a Circular no 3.648/2013). Brasília: BCB.",
    "[16] BANCO CENTRAL DO BRASIL (2022). Resolução BCB no 229, de 12 de "
    "maio de 2022: abordagem padronizada (RWACPAD), arts. 36 e 46 — "
    "definições de porte SME e de exposição de varejo. Brasília: BCB.",
    "[17] GORDY, M. B. (2002). Saddlepoint approximation of CreditRisk+. "
    "Journal of Banking & Finance, 26(7), 1335-1353.",
]


def mi(v, dec=2):
    """Valor monetário (entrada em reais) em R$ milhões."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return br(float(v) / 1e6, dec)


def br(v, dec=3, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{v*100:,.{dec}f}" if pct else f"{v:,.{dec}f}"
    s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s + "%" if pct else s


class _Doc:
    """Páginas A4 com cabeçalho, rodapé e cursor vertical de texto."""

    def __init__(self, pdf, titulo_curto):
        self.pdf = pdf
        self.titulo_curto = titulo_curto
        self.n = 0
        self.fig = None
        self.y = None

    def nova(self, secao=""):
        self.fecha()
        self.n += 1
        fig = plt.figure(figsize=A4)
        fig.patches.append(plt.Rectangle(
            (0, 0.962), 1, 0.038, transform=fig.transFigure,
            facecolor=AZUL, edgecolor="none"))
        fig.text(0.06, 0.974, "CreditLens — Validação de Modelos",
                 fontsize=8.5, color="white", va="center", fontweight="bold")
        fig.text(0.94, 0.974, secao, fontsize=8.5, color="#B9C7D4",
                 va="center", ha="right")
        fig.text(0.06, 0.028, self.titulo_curto, fontsize=7, color=CINZA)
        fig.text(0.94, 0.028, f"pág. {self.n}", fontsize=7, color=CINZA,
                 ha="right")
        fig.lines.append(plt.Line2D([0.06, 0.94], [0.045, 0.045],
                                    transform=fig.transFigure,
                                    color="#D5DBE1", lw=0.8))
        self.fig = fig
        self.y = 0.935
        return fig

    def fecha(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    # ---------- fluxo de texto ----------
    def need(self, h, secao=""):
        if self.y - h < 0.06:
            self.nova(secao)

    def h1(self, txt):
        self.need(0.06)
        self.fig.text(0.06, self.y, txt, fontsize=14, fontweight="bold",
                      color=AZUL, va="top")
        self.y -= 0.036

    def h2(self, txt, secao=""):
        self.need(0.08, secao)
        self.y -= 0.010                      # respiro antes do título
        self.fig.text(0.06, self.y, txt, fontsize=10.5, fontweight="bold",
                      color=AZUL, va="top")
        self.y -= 0.026

    def par(self, txt, size=8.2, width=122, secao="", color="#222"):
        w = textwrap.fill(txt, width)
        nl = w.count("\n") + 1
        h = 0.0145 * nl + 0.008
        self.need(h, secao)
        self.fig.text(0.06, self.y, w, fontsize=size, va="top",
                      linespacing=1.45, color=color)
        self.y -= h

    def formula(self, tex, h=0.034, secao=""):
        self.need(h + 0.01, secao)
        self.fig.text(0.5, self.y - 0.004, tex, fontsize=9.5, ha="center",
                      va="top", color="#111")
        self.y -= h + 0.004

    def kv(self, pares, secao=""):
        for k, v in pares:
            self.need(0.0225, secao)
            self.fig.text(0.07, self.y, k, fontsize=8.2, color=CINZA,
                          va="top")
            self.fig.text(0.34, self.y, v, fontsize=8.2, color="#222",
                          va="top",
                          family="monospace" if "SHA" in k else "sans-serif")
            self.y -= 0.0215

    def tabela(self, colunas, linhas, larguras, altura, secao="",
               esquerda=(0,)):
        self.need(altura + 0.02, secao)
        ax = self.fig.add_axes([0.06, self.y - altura, 0.88, altura])
        ax.axis("off")
        t = ax.table(cellText=linhas, colLabels=colunas, loc="upper center",
                     cellLoc="right", colWidths=larguras)
        t.auto_set_font_size(False)
        t.set_fontsize(7.2)
        t.scale(1, 1.32)
        for j in range(len(colunas)):
            t[0, j].set_facecolor(AZUL)
            t[0, j].set_text_props(color="white", fontweight="bold")
            t[0, j].set_edgecolor("white")
        for i in range(1, len(linhas) + 1):
            for j in range(len(colunas)):
                cel = t[i, j]
                cel.set_edgecolor("#E3E7EB")
                cel.set_facecolor(FUNDO if i % 2 == 0 else "white")
                if j in esquerda:
                    cel.set_text_props(ha="left")
                    cel._loc = "left"
        self.y -= altura + 0.015

    def eixo(self, altura, secao="", left=0.10):
        self.need(altura + 0.045, secao)
        self.y -= 0.022                      # respiro p/ o título do eixo
        ax = self.fig.add_axes([left, self.y - altura,
                                0.94 - left, altura])
        self.y -= altura + 0.030
        return ax


def _rotulo(n):
    for suf in [" (pooled)", " (setores)", " (setores, nu=10)",
                " (pooled, nu=10)", " (nu=10)", " (Poisson-gama 1F, pooled)",
                " (mixture, pooled)", " (mixture)", " (Poisson-gama 1F)",
                " (MC)", " (analítico, homog.)"]:
        n = n.replace(suf, "")
    return n.replace(" multifatorial", " MF")


# ======================================================================
# Diagnósticos de validação e tabela de parâmetros
# ======================================================================

def _rho_default_gauss(rho, p):
    """Correlação de default implícita pela correlação de ativos rho."""
    from scipy.stats import multivariate_normal, norm
    z = norm.ppf(p)
    p2 = multivariate_normal(mean=[0, 0],
                             cov=[[1, rho], [rho, 1]]).cdf([z, z])
    return float((p2 - p * p) / (p * (1 - p)))


def _ic_var(d, alpha, M_tab):
    """IC 95% do VaR_alpha por estatística de ordem sobre a amostra d;
    meia-amplitude relativa reescalada para o M da tabela."""
    d = np.sort(np.asarray(d))
    n = len(d)
    k = alpha * n
    h = 1.96 * np.sqrt(n * alpha * (1 - alpha))
    lo = d[int(max(np.floor(k - h), 0))]
    hi = d[int(min(np.ceil(k + h), n - 1))]
    q = np.quantile(d, alpha)
    rel = (hi - lo) / 2 / q if q > 0 else np.nan
    return rel * np.sqrt(n / M_tab) if M_tab else rel


def _diagnosticos(results, prm, pooled, mig, rel_agg):
    """Lista de [teste, resultado, critério, status]."""
    med = results["medidas"]
    resumo = results["resumo"]
    M = float(prm.get("M", 0) or 0)
    el0 = resumo["perda_esperada"]
    cal = results.get("calibracao_gaussiana", {})
    rows = []

    # 1. consistência da EL simulada
    for r in med.itertuples(index=False):
        nome, el, ul = r[0], r[1], r[2]
        if ul is None or (isinstance(ul, float) and np.isnan(ul)) or not M:
            continue
        dev = el / el0 - 1
        tol = 3 * ul / np.sqrt(M) / el0
        if "CSFP" in nome or "ponto de sela" in nome:
            continue                     # analíticos: diagnóstico próprio
        if "Bernoulli" in nome:
            st_ = "Esperado"
            crit = "aprox. Poisson: P(H≥1)=1−e^{−p}<p"
        else:
            st_ = "OK" if abs(dev) <= tol else "Atenção"
            crit = f"|desvio| ≤ 3σ_MC = {br(tol, 3, pct=True)}"
        rows.append([f"EL simulada — {_rotulo(nome)}",
                     f"desvio {br(dev, 3, pct=True)}", crit, st_])

    # 1b. CreditRisk+ clássico: discretização e comparação com a simulação
    cs = results.get("crplus_csfp")
    if cs:
        rows.append(["CreditRisk+ clássico: EL preservada nas faixas",
                     f"massa {br(cs['massa'], 6)}; L0 = R$ {br(cs['L0'] / 1e3, 2)} mil",
                     "massa = 1; EL = Σ p·c", "OK" if abs(cs["massa"] - 1)
                     < 1e-6 else "Atenção"])
        _vs = (cs.get("var_sela") or {}).get("99.90%")
        _ve = (cs.get("var_exato") or {}).get("99.90%")
        if _vs is not None and _ve:
            if _vs == _vs:
                _d = _vs / _ve - 1
                rows.append(["CreditRisk+ clássico: exato × ponto de sela",
                             f"VaR 99,9%: diferença {br(_d, 3, pct=True)}",
                             "|dif.| ≤ 1% (Lugannani–Rice)",
                             "OK" if abs(_d) <= 0.01 else "Atenção"])
            else:
                rows.append(["CreditRisk+ clássico: exato × ponto de sela",
                             "ponto de sela não convergiu",
                             "|dif.| ≤ 1% (Lugannani–Rice)", "Atenção"])
        _c9 = [c for c in med.columns if c.startswith("VaR 99.90")]

        def _v9(chave):
            m_ = med[med["modelo"].str.contains(chave, regex=False)]
            return float(m_.iloc[0][_c9[0]]) if len(m_) and _c9 else None
        vp, vc, vb = (_v9("simulação Poisson"), _v9("CSFP"),
                      _v9("simulação Bernoulli"))
        if vp and vc:
            dif = vc / vp - 1
            rows.append(["CreditRisk+: clássico × simulação Poisson "
                         "(VaR 99,9%)", f"diferença {br(dif, 3, pct=True)}",
                         "|dif.| ≤ 3% (mesma contagem Poisson)",
                         "OK" if abs(dif) <= 0.03 else "Atenção"])
        if vp and vb:
            dif = vb / vp - 1
            rows.append(["CreditRisk+: Bernoulli × Poisson (VaR 99,9%)",
                         f"diferença {br(dif, 3, pct=True)}",
                         "Bernoulli ≤ Poisson (no máximo um default "
                         "por contrato)",
                         "Esperado" if dif <= 0.03 else "Atenção"])

    # 2. calibração comum de dependência
    rho_a, pbar = cal.get("rho_ativos"), cal.get("pbar")
    alvo = prm.get("rho_alvo")
    if rho_a and pbar and alvo:
        imp = _rho_default_gauss(rho_a, pbar)
        rows.append(["Calibração Gaussiana: ρ_D implícita",
                     f"{br(imp, 3, pct=True)} (alvo {br(alvo, 3, pct=True)})",
                     "|desvio| ≤ 0,100 p.p.",
                     "OK" if abs(imp - alvo) <= 0.001 else "Atenção"])
    ab = results.get("calibracao_beta") or (
        {"a": prm.get("beta_a"), "b": prm.get("beta_b")}
        if prm.get("beta_a") else None)
    if ab and alvo:
        imp = 1.0 / (ab["a"] + ab["b"] + 1.0)
        rows.append(["Calibração Beta: ρ_D = 1/(a+b+1)",
                     f"{br(imp, 3, pct=True)} (alvo {br(alvo, 3, pct=True)})",
                     "|desvio| ≤ 0,100 p.p.",
                     "OK" if abs(imp - alvo) <= 0.001 else "Atenção"])
    mfc = (results.get("multifator") or {}).get("calibracao", {})
    if mfc.get("cargas_por_setor"):
        from .multifactor import rho_default_par
        ps = mfc["por_setor"]
        for chave, alvo_c, rot in [
                ("rho_ativos_intra", "rho_default_intra_alvo", "intra-setor"),
                ("rho_ativos_inter", "rho_default_inter_alvo", "inter-setor")]:
            desv = []
            for _, r in ps.iterrows():
                pc = r.get("pd_calibracao", r["pd_media"])
                imp = rho_default_par(r[chave], pc, pc)
                desv.append((abs(imp - r[alvo_c]), r["setor"]))
            dmax, smax = max(desv)
            rows.append([f"Calibração MF por setor ({rot}): ρ_D implícita",
                         f"desvio máx. {br(dmax * 100, 3)} p.p. ({smax}; "
                         f"{len(desv)} setores)",
                         "|desvio| ≤ 0,100 p.p. em todos os setores",
                         "OK" if dmax <= 0.001 else "Atenção"])
    for chave, alvo_k, rot in [("rho_ativos_intra", prm.get("rho_intra"),
                                "intra-setor"),
                               ("rho_ativos_inter", prm.get("rho_inter"),
                                "inter-setor")]:
        if mfc.get("cargas_por_setor"):
            break
        if mfc.get(chave) and alvo_k and pbar:
            imp = _rho_default_gauss(mfc[chave], pbar)
            rows.append([f"Calibração MF {rot}: ρ_D implícita",
                         f"{br(imp, 3, pct=True)} "
                         f"(alvo {br(alvo_k, 3, pct=True)})",
                         "|desvio| ≤ 0,100 p.p.",
                         "OK" if abs(imp - alvo_k) <= 0.001 else "Atenção"])

    # 3. ordenação das caudas: t-Student ≥ Gaussiano (mesma estrutura)
    col = next(c for c in med.columns if c.startswith("VaR 99.90"))
    def v(prefixo, mf_):
        sel = med[med["modelo"].str.startswith(prefixo)
                  & (med["modelo"].str.contains("multifatorial") == mf_)]
        return float(sel.iloc[0][col]) if len(sel) else None
    for mf_, rot in [(False, "1 fator"), (True, "multifatorial")]:
        g, t = v("Threshold Gaussiano", mf_), v("Threshold t-Student", mf_)
        if g and t:
            rows.append([f"Ordem de caudas ({rot}): t vs Gauss",
                         f"razão {br(t / g, 3)}",
                         "VaR99,9 t ≥ VaR99,9 Gauss",
                         "OK" if t >= g else "Atenção"])

    # 4. precisão de Monte Carlo do VaR 99,9%
    for nome, d in (results.get("distribuicoes") or {}).items():
        if "independente" in nome.lower() or len(d) < 1000:
            continue
        r_ = _ic_var(d, 0.999, M)
        rows.append([f"IC95% VaR99,9 — {_rotulo(nome)}",
                     f"± {br(r_, 3, pct=True)}",
                     "meia-amplitude ≤ 5,000%",
                     "OK" if r_ <= 0.05 else "Atenção"])

    # 5. agregação
    if pooled and rel_agg:
        rows.append(["Agregação: preservação da EL",
                     ("< 10⁻⁹ (arredondamento)"
                      if abs(rel_agg["el_diferenca_pct"]) <= 1e-9
                      else br(rel_agg["el_diferenca_pct"], 4, pct=True)),
                     "≤ 10⁻⁹ (identidade algébrica)",
                     "OK" if rel_agg["el_diferenca_pct"] <= 1e-9
                     else "Atenção"])
        rows.append(["Agregação: homogeneidade (CV de PD)",
                     f"máx. {br(rel_agg['pd_cv_max_bucket'], 3)}; "
                     f"{rel_agg['buckets_cv_alto']} acima",
                     "CV máx. ≤ 0,500 (Jensen)",
                     "OK" if rel_agg["pd_cv_max_bucket"] <= 0.5
                     else "Atenção"])

    # 5b. regime de granularidade infinita (motor pooled)
    rg = results.get("regimes_pooled")
    if pooled and rg:
        rows.append(["Regime de granularidade infinita: idiossincrático descartado",
                     f"máx. {br(rg['razao_max_lln'], 3, pct=True)} "
                     f"({rg['granularidade_infinita']} grupo(s))",
                     f"≤ {br(rg['tau'], 3, pct=True)} da variância sistemática",
                     "OK" if rg["razao_max_lln"] <= rg["tau"] else "Atenção"])

    # 6. IRB / GA
    irb = results.get("irb") or {}
    if irb:
        rows.append(["IRB: categoria das exposições",
                     "informada (segmento_irb)" if irb.get("segmentado")
                     else "ausente: tudo como atacado",
                     "classe Res. BCB 303 por linha",
                     "OK" if irb.get("segmentado") else "Esperado"])
        if irb.get("sme_sem_receita"):
            rows.append(["IRB: SME sem receita bruta",
                         f"{irb['sme_sem_receita']} exposição(ões)",
                         "receita informada (ajuste de porte)", "Atenção"])
        rows.append(["IRB: PDs elevadas ao piso",
                     f"{irb.get('n_pd_piso', 0)} linha(s)",
                     "piso 0,050% (0,100% rotativo)", "OK"])

    # 6b. padronizado (Res. BCB 229) e piso de saída (Res. BCB 303, art. 34)
    cp = results.get("padronizado") or {}
    if cp:
        if pooled and rel_agg and rel_agg.get("cpad"):
            ra = rel_agg["cpad"]["rwa_total"]
            dif = abs(cp["rwa_total"] / ra - 1) if ra else 0.0
            rows.append(["Padronizado: RWA agregado = contrato a contrato",
                         f"diferença {br(dif, 3, pct=True)}",
                         "identidade (FPR linear na exposição)",
                         "OK" if dif < 1e-9 else "Atenção"])
        av = cp.get("avisos") or {}
        falt = [f"{av[k]} {r}" for k, r in [
            ("residencial_sem_ltv", "residencial sem LTV"),
            ("pj_sem_receita", "PJ sem receita"),
            ("if_categoria_padrao_A", "IF sem categoria")] if av.get(k)]
        rows.append(["Padronizado: dados para os fatores",
                     "; ".join(falt) if falt else "completos",
                     "LTV, receita e categoria de IF informados",
                     "Esperado" if falt else "OK"])
        ps_ = cp["piso"]
        rows.append(["Piso de saída (art. 34, Res. BCB 303)",
                     f"IRB/padronizado {br(ps_['razao_irb_cpad'], 3)}",
                     "RWA = max(IRB; 72,5% × padronizado)",
                     "Esperado" if ps_["vinculante"] else "OK"])
    ad = results.get("inadimplidos")
    if ad:
        r_ = ad["resumo"]
        if r_["provisao_informada"] == r_["provisao_informada"]:
            cob = r_["provisao_informada"] / r_["provisao_elbe"] \
                if r_["provisao_elbe"] else np.nan
            rows.append(["Default: provisão informada × ELBE",
                         f"cobertura {br(cob, 3, pct=True)}",
                         "provisão ≥ ELBE (estágio 3)",
                         "OK" if cob >= 1 - 1e-9 else "Atenção"])
        rows.append(["Default: contratos além do prazo de baixa",
                     f"{r_['ja_baixados']} contrato(s)",
                     "0 (ELBE = 100%: candidatos a write-off)",
                     "OK" if r_["ja_baixados"] == 0 else "Atenção"])
    ga = irb.get("ajuste_granularidade")
    if ga is not None:
        rows.append(["Ajuste de granularidade",
                     f"GA = R$ {mi(ga)} mi", "GA ≥ 0 (risco de nomes)",
                     "OK" if ga >= 0 else "Atenção"])

    # 7. migração
    if mig is not None:
        ecl = mig.get("ecl")
        if ecl is not None:
            e12, eli = float(ecl["ECL_12m"].sum()), float(
                ecl["ECL_lifetime"].sum())
            rows.append(["ECL lifetime vs 12 meses",
                         f"razão {br(eli / e12 if e12 else 0, 3)}",
                         "lifetime ≥ 12 meses",
                         "OK" if eli >= e12 else "Atenção"])
        cum = mig.get("cum")
        if cum is not None:
            mono = bool(np.all(np.diff(np.asarray(cum), axis=1) >= -1e-12))
            rows.append(["PD acumulada (estrutura a termo)",
                         "monótona" if mono else "não monótona",
                         "não decrescente no prazo",
                         "OK" if mono else "Atenção"])
    return rows


LEGENDA_STATUS = [
    ("OK", "o resultado está dentro do critério; nenhuma ação é necessária."),
    ("Atenção", "o resultado ficou fora do critério. Os números da execução "
     "continuam válidos, mas o ponto deve ser examinado pelo validador; em "
     "geral há uma ação corretiva simples, indicada na leitura dos "
     "resultados."),
    ("Esperado", "o resultado se afasta do critério por um efeito conhecido "
     "do próprio modelo, explicado no texto; não requer ação."),
]


def _num(txt):
    """Primeiro número (formato brasileiro) de um texto de resultado."""
    import re
    m = re.search(r"-?[\d.]+,\d+", txt)
    return float(m.group().replace(".", "").replace(",", ".")) if m else None


def _interpretar(drows, M):
    """Síntese em prosa e leitura de cada teste fora do critério.
    Devolve (sintese, itens) com itens = [(status, teste, leitura, ação)]."""
    n = len(drows)
    n_ok = sum(r[3] == "OK" for r in drows)
    n_at = sum(r[3] == "Atenção" for r in drows)
    n_es = sum(r[3] == "Esperado" for r in drows)
    if n_ok == n:
        sint = (f"Todos os {n} testes passaram: a execução é internamente "
                "consistente — a perda esperada simulada adere à analítica, "
                "as calibrações reproduzem os alvos, a ordem das caudas é a "
                "prevista pela teoria e os quantis têm precisão adequada.")
    else:
        partes = [f"{n_ok} de {n} testes passaram"]
        if n_at:
            partes.append(f"{n_at} {'ficou' if n_at == 1 else 'ficaram'} "
                          "em atenção")
        if n_es:
            partes.append(f"{n_es} {'apresenta' if n_es == 1 else 'apresentam'}"
                          " comportamento esperado e documentado")
        sint = (", ".join(partes[:-1]) + " e " + partes[-1]
                if len(partes) > 1 else partes[0]) + "."
        sint += (" Os itens abaixo explicam cada caso e a ação recomendada."
                 if n_at or n_es else "")
    itens = []
    for teste, res, crit, st in drows:
        if st == "OK":
            continue
        v = _num(res)
        if teste.startswith("IC95%"):
            M_sug = None
            if v and M:
                # mira 4,5% (margem de 10% sobre o critério de 5%)
                M_sug = int(np.ceil(M * (v / 4.5) ** 2 / 50_000) * 50_000)
            leitura = (f"a meia-amplitude do intervalo de confiança do VaR "
                       f"99,9% é {res.replace('± ', '±')}, acima de 5%: o VaR "
                       "reportado é correto em média, mas oscila mais que o "
                       "tolerado entre simulações — típico de modelos de "
                       "cauda pesada.")
            acao = ("aumentar o número de simulações; como o erro do "
                    "quantil cai com 1/√M, M ≈ "
                    f"{br(M_sug, 0)} deve reduzir a meia-amplitude para menos de 5%."
                    if M_sug else "aumentar o número de simulações M.")
        elif teste.startswith("EL simulada"):
            if st == "Esperado":
                leitura = ("o CreditRisk+ Bernoulli declara default "
                           "quando há ao menos um evento de Poisson, o que dá "
                           "probabilidade 1−e^(−p) < p: a EL fica levemente "
                           "abaixo da analítica por construção.")
                acao = "nenhuma."
            else:
                leitura = (f"a EL simulada desvia da analítica ({res}) além "
                           "de três erros-padrão de Monte Carlo.")
                acao = ("rodar novamente com outra semente ou M maior; se o "
                        "desvio persistir, revisar a implementação do modelo.")
        elif teste.startswith("CreditRisk+ clássico: exato"):
            leitura = (f"o VaR 99,9% da inversão exata e o da aproximação "
                       f"de ponto de sela divergem ({res}) além de 1%.")
            acao = ("a aproximação perde precisão em carteiras muito "
                    "concentradas; prevalece o valor exato, mas convém "
                    "revisar a discretização (L0) e a carteira.")
        elif teste.startswith("Calibração"):
            leitura = (f"a correlação de default implícita ({res}) não "
                       "reproduz o alvo dentro de 0,1 p.p.")
            acao = ("verificar a convergência do otimizador de calibração e "
                    "se o alvo é atingível para a PD média da carteira.")
        elif teste.startswith("Ordem de caudas"):
            leitura = ("o VaR do t-Student ficou abaixo do gaussiano, contra "
                       "a teoria (a variante t acrescenta dependência de "
                       "cauda).")
            acao = ("aumentar M; se persistir, revisar a implementação.")
        elif teste.startswith("Agregação: preservação"):
            leitura = "a agregação não preservou a perda esperada."
            acao = "revisar a camada de agregação (deveria ser identidade)."
        elif teste.startswith("Agregação: homogeneidade"):
            leitura = ("há grupos homogêneos com PDs muito dispersas, o que "
                       "enviesa as funções de capital (desigualdade de "
                       "Jensen).")
            acao = ("refinar a formação dos grupos (coluna gh) ou subir o "
                    "corte de grandes exposições.")
        elif teste.startswith("IRB: categoria"):
            leitura = ("a carteira não informa a categoria IRB: todas as "
                       "exposições foram tratadas como atacado — adequado a "
                       "carteiras corporativas, conservador para varejo.")
            acao = ("se houver varejo ou SME, informar a coluna "
                    "segmento_irb.")
        elif teste.startswith("IRB: SME"):
            leitura = ("há exposições SME sem receita bruta: aplicou-se a "
                       "correlação sem redução de porte (conservador).")
            acao = "informar a coluna receita_bruta_mi."
        elif teste.startswith("Padronizado: RWA agregado"):
            leitura = ("o RWA padronizado da carteira agregada difere da soma "
                       "contrato a contrato.")
            acao = "revisar a camada de agregação (deveria ser identidade)."
        elif teste.startswith("Padronizado: dados"):
            leitura = (f"faltam atributos para alguns fatores ({res}): "
                       "aplicou-se o fator conservador — faixa de LTV acima "
                       "de 100%, 100% para PJ sem receita, categoria A para "
                       "IF sem classificação.")
            acao = ("informar ltv, receita_bruta_mi e categoria_if para "
                    "obter o fator exato de cada exposição.")
        elif teste.startswith("Piso de saída"):
            leitura = (f"o RWA dos modelos internos ficou abaixo de 72,5% do "
                       f"padronizado ({res}): o piso é vinculante e o capital "
                       "passa a ser determinado pela abordagem padronizada.")
            acao = ("nenhuma no cálculo; para a gestão, o benefício dos "
                    "modelos internos está limitado pelo piso.")
        elif teste.startswith("Ajuste de granularidade"):
            leitura = "o ajuste de granularidade resultou negativo."
            acao = "revisar a concentração da carteira e os parâmetros do GA."
        elif teste.startswith("ECL"):
            leitura = "o ECL lifetime ficou abaixo do de 12 meses."
            acao = "revisar a matriz de transição e os prazos."
        elif teste.startswith("PD acumulada"):
            leitura = "a PD acumulada não é monótona no prazo."
            acao = ("revisar a matriz de transição: linhas devem somar 1 e o "
                    "default deve ser absorvente.")
        else:
            leitura, acao = f"{res} (critério: {crit}).", "examinar."
        itens.append((st, teste, leitura, acao))
    return sint, itens


def _parametros(results, prm, pooled, tem, mig, extra):
    """Lista de [parâmetro, valor, papel, referência]."""
    cal = results.get("calibracao_gaussiana", {})
    mfc = (results.get("multifator") or {}).get("calibracao", {})
    ex = extra or {}
    R = [["Simulações (M)", br(prm.get("M"), 0),
          "precisão dos quantis de Monte Carlo", "[13]"],
         ["Semente", str(prm.get("seed", "—")),
          "reprodutibilidade integral da execução", "—"],
         ["ρ_D alvo (correl. default)", br(prm.get("rho_alvo"), 3, pct=True),
          "dependência comum a todos os modelos", "[11]"],
         ["ρ de ativos calibrada", br(cal.get("rho_ativos"), 3),
          "fator único dos modelos de limiar", "[1, 11, 12]"]]
    if tem("t-Student"):
        R.append(["ν (graus de liberdade)", br(prm.get("nu_t"), 3),
                  "dependência de cauda do modelo t", "[11, 13]"])
    if tem("Beta-binomial"):
        ab = results.get("calibracao_beta") or {"a": prm.get("beta_a"),
                                                "b": prm.get("beta_b")}
        if ab.get("a"):
            R.append(["Beta (a, b)",
                      f"{br(ab['a'], 3)} / {br(ab['b'], 3)}",
                      "mistura calibrada por momentos", "[11, 13]"])
    cs = results.get("crplus_csfp")
    if cs:
        R.append(["CreditRisk+ clássico: L0 / faixas",
                  f"R$ {br(cs['L0'] / 1e3, 2)} mil / {cs['faixas']}",
                  "discretização das severidades (EL preservada)", "[8]"])
        R.append(["CreditRisk+ clássico: algoritmo",
                  "Panjer" if cs["metodo"] == "Panjer" else "FFT da FGP",
                  "distribuição exata da FGP", "[8]"])
    if tem("CreditRisk+"):
        R.append(["CreditRisk+ (w, v)",
                  f"{br(prm.get('crplus_w'), 3)} / "
                  f"{br(prm.get('crplus_v'), 3)}",
                  "peso do fator gama e sua dispersão", "[8]"])
    if tem("multifatorial"):
        R.append(["ρ_D intra / inter-setor",
                  f"{br(prm.get('rho_intra'), 3, pct=True)} / "
                  f"{br(prm.get('rho_inter'), 3, pct=True)}",
                  "dependência setorial (dois níveis)", "[11, 13]"])
        if mfc:
            por_setor = bool(mfc.get("cargas_por_setor"))
            R.append(["ρ de ativos intra / inter"
                      + (" (média ponderada)" if por_setor else ""),
                      f"{br(mfc.get('rho_ativos_intra'), 3)} / "
                      f"{br(mfc.get('rho_ativos_inter'), 3)}",
                      "cargas fatoriais calibradas", "[11]"])
            R.append(["Cargas por setor",
                      "sim" if por_setor else "não (uniformes)",
                      "alvos de ρ_D e calibração setor a setor"
                      if por_setor else "mesmos alvos em todos os setores",
                      "[11, 13]"])
    if ex.get("contrib_alpha") is not None:
        R.append(["α das contribuições", br(ex["contrib_alpha"], 3),
                  "nível da alocação de Euler do ES", "[9]"])
    if not pooled and ex.get("contrib_S"):
        R.append(["Iterações S (decomposição)", str(ex["contrib_S"]),
                  "estabilidade das contribuições", "[9, 13]"])
    if pooled and ex.get("max_grandes") is not None:
        R.append(["Máx. grandes exposições", str(ex["max_grandes"]),
                  "corte de materialidade nominal", "[2, 7]"])
    irb_ = results.get("irb") or {}
    seg_ = irb_.get("por_segmento")
    R.append(["IRB: norma / α", "Res. BCB 303 / 99,900%",
              "função de capital regulatório", "[3, 15]"])
    R.append(["IRB: categorias",
              (str(len(seg_)) if seg_ is not None else "1")
              + (" informadas" if irb_.get("segmentado") else " (padrão)"),
              "R e ajuste de maturidade por classe", "[15, 16]"])
    ad = results.get("inadimplidos")
    if ad:
        r_ = ad["resumo"]
        R.append(["Default: curva de LGD (ELBE)", r_["curva"],
                  "provisão pelo tempo em default", "[15]"])
        R.append(["Default: prazo até a baixa",
                  f"{br(r_['prazo_baixa'], 0)} meses",
                  "ELBE = 100% no write-off", "[15]"])
        R.append(["Default: margem k da LGD_D", br(r_["k"], 3),
                  "LGD_D = ELBE + k(1 − ELBE); K = LGD_D − ELBE", "[3, 15]"])
    if mig is not None:
        R.append(["Horizonte multiperíodo",
                  f"{ex.get('horizonte', '—')} ano(s)",
                  "passos da matriz anual no capital", "[10]"])
        R.append(["Taxa de desconto", br(ex.get("taxa_desc"), 3, pct=True),
                  "valor presente das perdas (ECL)", "[14]"])
    return R


def gerar_paper(destino, port, results, mig=None, nome_base="",
                extra=None, rel_agg=None, descritiva=None):
    """Escreve o paper de validação em `destino` (caminho ou buffer)."""
    resumo = results["resumo"]
    prm = results.get("parametros", {})
    med = results["medidas"]
    total = resumo["exposicao_liquida_total"]
    pooled = bool(results.get("pooled", False)) or \
        any("(pooled)" in m for m in med["modelo"])
    cal_g = results.get("calibracao_gaussiana", {})
    rho_a = cal_g.get("rho_ativos")
    pbar = cal_g.get("pbar", resumo["pd_media_ponderada"])
    beta_ab = results.get("calibracao_beta") or (
        {"a": prm.get("beta_a"), "b": prm.get("beta_b")}
        if prm.get("beta_a") else None)
    modelos = list(med["modelo"])
    tem = lambda s: any(s in m for m in modelos)
    mf = results.get("multifator")
    irb = results.get("irb", {})
    agora = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    col999 = next(c for c in med.columns if c.startswith("VaR 99.90"))

    with PdfPages(destino) as pdf:
        D = _Doc(pdf, f"Base: {nome_base or 'carteira analisada'} • "
                      f"gerado em {agora}")

        # ================= 1. capa / resumo =================
        D.nova("Identificação")
        D.h1("Validação de Modelos")
        D.h2("Modelos de VaR para Risco de Crédito em Bancos Brasileiros")
        D.par("De acordo com as Resoluções CMN nº 4.557/2017 (gerenciamento "
              "de riscos e de capital) e nº 4.966/2021 (perda esperada) e as "
              "Resoluções BCB nº 229/2022 (abordagem padronizada) e nº "
              "303/2023 (abordagem IRB).")
        D.y -= 0.006
        D.par("Documento gerado automaticamente pelo CreditLens a partir da "
              "base de entrada e dos parâmetros escolhidos nesta execução. "
              "Apresenta a fundamentação teórica de cada modelo executado, "
              "os resultados obtidos e as referências bibliográficas que "
              "suportam as técnicas, para verificação independente.",
              color=CINZA)
        D.y -= 0.008
        D.h2("1. Identificação da execução")
        gauss_ref = None
        for alvo in ["Threshold Gaussiano multifatorial",
                     "Threshold Gaussiano 1F"]:
            sel = med[med["modelo"].str.startswith(alvo)]
            if len(sel):
                gauss_ref = (sel.iloc[0]["modelo"],
                             float(sel.iloc[0][col999]))
                break
        D.kv([
            ("Base de entrada", nome_base or "—"),
            ("Data da geração", agora),
            ("Contrapartes / linhas de modelagem",
             br(resumo["n_contrapartes"], 0)),
            ("Representação", "agregada (grandes exposições + grupos "
             "homogêneos × setor)" if pooled else
             "granular (nível contraparte)"),
            ("Simulações (M)", br(prm.get("M", 0), 0)),
            ("Semente", str(prm.get("seed", "—"))),
            ("Correlação de default alvo",
             br(prm.get("rho_alvo"), 3, pct=True)),
            ("Correlação de ativos calibrada", br(rho_a, 3)),
        ])
        D.y -= 0.01
        D.h2("2. A carteira")
        D.kv([
            ("Exposição total (EAD, R$ mi)", mi(resumo["ead_total"])),
            ("Exposição líquida (EAD × LGD, R$ mi)",
             mi(resumo["exposicao_liquida_total"])),
            ("PD média ponderada por severidade", br(pbar, 3, pct=True)),
            ("LGD média", br(resumo["lgd_media"], 3)),
            ("Perda esperada (EL, R$ mi)", mi(resumo["perda_esperada"])),
            ("HHI de nomes / nº efetivo",
             f"{br(resumo['hhi'], 5)}  /  {br(resumo['n_efetivo'], 1)}"),
            ("Maior exposição / top 10",
             f"{br(resumo['maior_exposicao_pct'], 3, pct=True)}  /  "
             f"{br(resumo['top10_exposicao_pct'], 3, pct=True)}"),
        ])
        if gauss_ref:
            D.y -= 0.01
            D.par(f"Síntese: no modelo de referência ({gauss_ref[0]}), o "
                  f"VaR 99,90% é de R$ {mi(gauss_ref[1])} milhões "
                  f"({br(gauss_ref[1]/total, 3, pct=True)} da exposição "
                  "líquida). A seção 5 compara as caudas entre os modelos.")
        if pooled:
            D.y -= 0.006
            D.h2("3. Tratamento dos dados: representação híbrida")
            D.par("A base analítica (contrato a contrato) foi agregada em "
                  "grandes exposições nominais e grupos homogêneos rating × "
                  "setor × categoria IRB, com parâmetros ponderados pela "
                  "severidade c = EAD × LGD — o que preserva a perda "
                  "esperada exatamente. A técnica é legítima porque, nos "
                  "modelos de fator, os defaults são independentes dado o "
                  "fator sistemático: num grupo homogêneo, a contagem de "
                  "defaults é condicionalmente Binomial e resume o grupo "
                  "sem perda de informação [11]. Com severidades desiguais, "
                  "cada grupo é representado pelo seu número efetivo de "
                  "contratos, n_eff = C²/Σc², que preserva a média e a "
                  "variância idiossincrática condicional. O limite de "
                  "granularidade infinita de Vasicek [1, 2], que descarta essa "
                  "variância, só é usado quando ela é inferior a 1% da "
                  "sistemática, isto é, (1 − ρ_D)/(n_eff·ρ_D) < 1%; os "
                  "demais grupos são simulados pela Binomial condicional. "
                  "O termo descartado é o que o ajuste de granularidade "
                  "devolve ao capital [6, 7].")

        # ================= parâmetros =================
        D.nova("Parâmetros")
        D.h1("Parâmetros da execução e suporte bibliográfico")
        D.par("Cada parâmetro escolhido nesta rodada, seu papel na "
              "modelagem e a referência que fundamenta a técnica. Valores "
              "derivados (correlações de ativos, parâmetros da Beta) são "
              "produzidos pela calibração e verificados na seção de "
              "diagnósticos.", color=CINZA)
        prows = _parametros(results, prm, pooled, tem, mig, extra)
        D.tabela(["Parâmetro", "Valor", "Papel na modelagem", "Ref."],
                 prows, [0.27, 0.17, 0.44, 0.12],
                 0.030 + 0.0215 * len(prows), "Parâmetros",
                 esquerda=(0, 2))

        # ================= metodologia =================
        D.nova("Metodologia")
        D.h1("Fundamentação teórica dos modelos executados")
        D.par("Todos os modelos com dependência são calibrados a um alvo "
              "comum de correlação de default "
              f"(ρ_D = {br(prm.get('rho_alvo'), 3, pct=True)}), tornando as "
              "caudas diretamente comparáveis [11]. As seções seguem a "
              "ordem da tabela de resultados; a numeração entre colchetes "
              "remete às referências da última seção.", color=CINZA)
        sec = 0

        def secao(titulo):
            nonlocal sec
            sec += 1
            D.h2(f"M{sec}. {titulo}", "Metodologia")

        if tem("Binomial independente"):
            secao("Binomial independente (contrafactual sem dependência)")
            D.par("Defaults independentes com as PDs individuais da "
                  "carteira. Serve de contrafactual: pela lei dos grandes "
                  "números, a perda concentra-se na EL e o risco de cauda "
                  "desaparece — a distância até os demais modelos mede o "
                  "efeito da dependência [13].")
            D.formula(r"$L=\sum_i c_i\,\mathbf{1}\{D_i\},\quad "
                      r"D_i\sim \mathrm{Bernoulli}(p_i)\ \ "
                      r"\mathrm{independentes}$", secao="Metodologia")
        if tem("Beta-binomial") and beta_ab:
            secao("Beta-binomial (mixture model)")
            D.par("A probabilidade comum de default Z é sorteada de uma "
                  "Beta(a, b); dado Z, os defaults são independentes. A "
                  "calibração por momentos reproduz a PD média e a "
                  "correlação de default alvo [11, 13]: "
                  f"a = {br(beta_ab['a'])}, b = {br(beta_ab['b'])}.")
            D.formula(r"$Z\sim\mathrm{Beta}(a,b),\quad "
                      r"D_i\,|\,Z\sim\mathrm{Bernoulli}(Z),\quad "
                      r"\rho_D=\frac{1}{a+b+1}$", secao="Metodologia")
        if tem("CreditRisk+"):
            secao("CreditRisk+ (mistura Poisson-gama, um fator)")
            D.par("Arcabouço atuarial do Credit Suisse [8]: o fator "
                  "sistêmico S ~ Gamma com média 1 multiplica as "
                  "intensidades de default; os defaults condicionais são "
                  "Poisson. Parâmetros desta execução: peso do fator "
                  f"w = {br(prm.get('crplus_w'), 3)} e dispersão "
                  f"v = {br(prm.get('crplus_v'), 3)}."
                  + (" Na representação agregada, a contagem do pool "
                     "agrega exatamente: a soma de Poissons é Poisson."
                     if pooled else ""))
            D.formula(r"$S\sim\Gamma(v,1/v),\quad H_i\,|\,S\sim"
                      r"\mathrm{Poisson}\left(p_i\,(1-w+wS)\right)$",
                      secao="Metodologia")
            cs = results.get("crplus_csfp")
            if cs:
                D.par("Quatro versões são reportadas como modelos "
                      "distintos, todas com os mesmos (w, v). Por simulação: "
                      "Bernoulli (cada contrato inadimple no máximo uma vez, "
                      "com probabilidade 1 − e^(−λ)) e Poisson (contagem "
                      "Poisson, como no original). Analíticas: a clássica "
                      "(CSFP, 1997) e o ponto de sela (Gordy, 2002), que "
                      "aproxima a cauda pela fórmula de Lugannani–Rice com "
                      "as severidades exatas. A clássica "
                      "trata a contagem como Poisson, discretiza as "
                      "severidades em faixas de L0 (aqui R$ "
                      f"{br(cs['L0'] / 1e3, 2)} mil, {cs['faixas']} faixas, EL "
                      "preservada) e obtém a distribuição exata da função "
                      "geradora de probabilidade por "
                      + ("recursão de Panjer" if cs["metodo"] == "Panjer"
                         else "FFT (o Panjer falha quando P(L=0) é ínfima)")
                      + ": sem ruído de simulação, mas com erro de "
                      "discretização e superestimação em PDs altas. As "
                      "versões por simulação usam as severidades exatas; são "
                      "flexíveis, mas têm erro amostral.")
                D.formula(r"$G_L(z)=e^{\sum_j w_0\varepsilon_j(z^{\nu_j}-1)}"
                          r"\left[\frac{1-\delta}{1-\delta P(z)}\right]^{\alpha}$",
                          secao="Metodologia")
        if tem("Threshold"):
            secao("Modelos de limiar (Gaussiano e t-Student, 1 fator)")
            D.par("Na tradição estrutural de Merton [12], o default ocorre "
                  "quando a variável latente de ativos cai abaixo do limiar "
                  "implícito pela PD. A correlação de ativos "
                  f"ρ = {br(rho_a, 3)} foi calibrada para reproduzir a "
                  "correlação de default alvo via normal bivariada [11]. A "
                  f"variante t-Student (ν = {prm.get('nu_t', 10):g}) "
                  "adiciona um choque radial comum, produzindo dependência "
                  "de cauda estritamente positiva — o contraste entre as "
                  "duas variantes é um teste de sensibilidade de premissa "
                  "de cauda [11, 13].")
            D.formula(r"$Y_i=\sqrt{\rho}\,G+\sqrt{1-\rho}\;\epsilon_i,"
                      r"\quad D_i=\mathbf{1}\{Y_i<\Phi^{-1}(p_i)\}$",
                      secao="Metodologia")
            D.formula(r"$\rho_D=\frac{\Phi_2(z,z;\rho)-p^2}{p(1-p)},"
                      r"\quad z=\Phi^{-1}(p)$", secao="Metodologia")
        if tem("multifatorial"):
            secao("Extensão multifatorial setorial")
            mfc = (mf or {}).get("calibracao", {})
            D.par("Dois níveis de fatores: um global e um por setor, com "
                  "alvos de correlação de default intra-setor "
                  f"({br(prm.get('rho_intra'), 3, pct=True)}) e "
                  "entre setores "
                  f"({br(prm.get('rho_inter'), 3, pct=True)}) — correlações "
                  "de ativos calibradas de "
                  f"{br(mfc.get('rho_ativos_intra'), 3)} e "
                  f"{br(mfc.get('rho_ativos_inter'), 3)}. A decomposição "
                  "de risco por setor usa alocação de Euler: contribuição "
                  "ao ES como perda esperada condicional à cauda [9]."
                  + (" As cargas foram calibradas por setor, cada setor "
                     "com os seus alvos e a sua PD média; os valores acima "
                     "são médias ponderadas pela exposição. Entre os "
                     "setores s e t, a correlação de ativos é "
                     "√(ρ_G,s·ρ_G,t)." if mfc.get("cargas_por_setor") else ""))
            if mfc.get("cargas_por_setor"):
                D.formula(r"$Y_i=\sqrt{\rho_{G,s}}\,G+\sqrt{\rho_{S,s}"
                          r"-\rho_{G,s}}\,F_{s}+\sqrt{1-\rho_{S,s}}\;"
                          r"\epsilon_i,\quad s=s(i)$", secao="Metodologia")
            else:
                D.formula(r"$Y_i=\sqrt{\rho_G}\,G+\sqrt{\rho_S-\rho_G}\,"
                          r"F_{s(i)}+\sqrt{1-\rho_S}\;\epsilon_i$",
                          secao="Metodologia")
        if tem("ASRF"):
            secao("ASRF (carteira assintoticamente granular)")
            D.par("Limite analítico de Vasicek [1] para uma carteira de "
                  "granularidade infinita — número de exposições tendendo ao "
                  "infinito, cada uma com peso desprezível — e um único "
                  "fator sistemático, que fundamenta o IRB [2]. Serve de referência "
                  "superior de diversificação idiossincrática perfeita.")
            D.formula(r"$\mathrm{VaR}_\alpha = C\cdot\Phi\!\left("
                      r"\frac{\Phi^{-1}(\bar p)+\sqrt{\rho}\,"
                      r"\Phi^{-1}(\alpha)}{\sqrt{1-\rho}}\right)$",
                      secao="Metodologia")
        if irb:
            secao("Capital regulatório: IRB (modelos internos, Res. BCB 303/2023), GA e padronizado")
            seg = irb.get("por_segmento")
            cats = (", ".join(seg["descricao"]) if seg is not None
                    and len(seg) else "atacado — demais")
            D.par("Função de capital da abordagem IRB do Banco Central do "
                  "Brasil [15], derivada do modelo assintótico de fator único "
                  "[1, 2, 3]. A correlação R depende da categoria da "
                  "exposição (arts. 39 e 46): no atacado, 0,12·f + 0,24·(1−f) "
                  "com f = (1−e^(−50·PD))/(1−e^(−50)), reduzida em até 4 p.p. "
                  "pelo porte nas SME (receita bruta S entre R$ 15 e 300 "
                  "milhões [16]), elevada a 0,30 no limite superior nas HVCRE "
                  "e multiplicada por 1,25 nas instituições financeiras S1/S2; "
                  "no varejo, 0,15 (residencial), 0,04 (rotativo qualificado) "
                  "e 0,03·g + 0,16·(1−g) com fator 35 (demais). O ajuste de "
                  "maturidade, com coeficiente oficial 0,11852, aplica-se só "
                  "ao atacado; pisos de PD de 0,05% (0,10% no rotativo) "
                  f"(art. 61). Categorias presentes nesta carteira: {cats}."
                  + ("" if irb.get("segmentado") else
                     " A carteira não informa a coluna segmento_irb: todas "
                     "as exposições foram tratadas como atacado — demais."))
            D.formula(r"$K=\left[\Phi\!\left(\frac{\Phi^{-1}(PD)+"
                      r"\sqrt{R}\,\Phi^{-1}(0{,}999)}{\sqrt{1-R}}"
                      r"\right)-PD\right]\cdot MA$",
                      secao="Metodologia")
            D.formula(r"$MA=\frac{1+(M-2{,}5)\,b}{1-1{,}5\,b}\ "
                      r"\mathrm{(atacado)},\quad MA=1\ \mathrm{(varejo)},"
                      r"\quad b=(0{,}11852-0{,}05478\,\ln PD)^2$",
                      secao="Metodologia")
            D.par("O ajuste de granularidade segue Gordy e Lütkebohmert "
                  "[7], na formulação baseada no CreditRisk+ implementada "
                  "por Bolder (cap. 6) [8, 13]: corrige o risco de nomes "
                  "que a função IRB assume diversificado"
                  + (", com o termo c² substituído por C²/n_eff por pool"
                     if pooled else "") + ".")
            if results.get("padronizado"):
                D.par("Abordagem padronizada (Res. BCB 229/2022): RWA = soma "
                      "de EAD × FPR, com o fator da categoria de cada "
                      "exposição — varejo 75% (art. 46), rotativo sem uso em "
                      "360 dias 45% (art. 47), pessoa natural fora do varejo "
                      "100% (art. 48), PME 85% (art. 36), demais PJ 100% "
                      "(art. 41), residencial por faixa de LTV de 20% a 70% "
                      "(art. 50) e ativo problemático conforme a provisão "
                      "(art. 66). O piso de saída da Res. BCB 303 (art. 34) "
                      "limita o RWA dos modelos internos a, no mínimo, 72,5% "
                      "do padronizado.")
                D.formula(r"$\mathrm{RWA}=\max\{\mathrm{RWA}_{IRB};\;"
                          r"0{,}725\cdot\sum_n \mathrm{EAD}_n\,"
                          r"\mathrm{FPR}_n\}$", secao="Metodologia")

        if mig is not None:
            secao("Migração de ratings, ECL e capital multiperíodo")
            D.par("A matriz de transição anual fornece a estrutura a termo "
                  "de PD por potências; o ECL desconta as perdas marginais "
                  "em 12 meses e no prazo de cada contrato (lifetime), com "
                  "estágios por deterioração de rating, conforme IFRS 9 / "
                  "Res. CMN 4.966 [14]. O capital multiperíodo evolui a "
                  "carteira ano a ano pela matriz condicional ao fator "
                  "sistemático — generalização do deslocamento de crédito "
                  "de um parâmetro [10] — acumulando perdas descontadas."
                  + (" Na carteira agregada, cada grupo evolui como vetor de "
                     "frações por rating; nos grupos fora do limite de "
                     "granularidade infinita, os defaults de cada ano são sorteados "
                     "por Binomial(A, q) sobre o número inteiro de contratos "
                     "equivalentes vivos A, com a taxa de default condicional "
                     "entre os vivos q — o risco idiossincrático intragrupo "
                     "entra no capital multiperíodo." if pooled else ""))

        # ================= resultados =================
        D.nova("Resultados")
        D.h1("Resultados por modelo")
        cols = [c for c in ["modelo", "EL", "UL", "VaR 99.00%", "ES 99.00%",
                            "VaR 99.90%", "ES 99.90%"] if c in med.columns]
        linhas = []
        for r in med[cols].itertuples(index=False):
            linhas.append([_rotulo(r[0])] + [mi(v) for v in r[1:]])
        D.tabela(["Modelo (R$ mi)", "EL", "UL", "VaR 99%", "ES 99%", "VaR 99,9%",
                  "ES 99,9%"][:len(cols)], linhas,
                 [0.30] + [0.1167] * (len(cols) - 1),
                 0.030 + 0.0215 * len(linhas), "Resultados")
        D.par("Valores em R$ milhões, duas casas decimais (base de entrada "
              "em reais). A hierarquia das caudas reflete a estrutura "
              "de dependência de cada modelo sob a calibração comum.",
              color=CINZA)
        # indicadores relativos
        base_g = med[med["modelo"].str.startswith("Threshold Gaussiano 1F")]
        vg = float(base_g.iloc[0][col999]) if len(base_g) else None
        col_es = col999.replace("VaR", "ES")
        lin2 = []
        for _, r in med.iterrows():
            v9 = r.get(col999)
            if v9 is None or v9 != v9:
                continue
            e9 = r.get(col_es)
            lin2.append([_rotulo(r["modelo"]),
                         br(v9 / total, 3, pct=True),
                         br(v9 / r["EL"], 3) if r["EL"] else "—",
                         br(e9 / v9, 3) if e9 == e9 and v9 else "—",
                         br(v9 / vg, 3) if vg else "—"])
        D.y -= 0.004
        D.h2("Indicadores relativos (quantil 99,90%)", "Resultados")
        D.tabela(["Modelo", "VaR/expos.", "VaR/EL", "ES/VaR",
                  "× Gauss 1F"], lin2, [0.36, 0.16, 0.16, 0.16, 0.16],
                 0.030 + 0.0215 * len(lin2), "Resultados")
        sub = med.dropna(subset=[col999])
        ax = D.eixo(0.26, "Resultados", left=0.30)
        nomes = [_rotulo(m) for m in sub["modelo"]]
        vals = sub[col999].to_numpy() / total * 100
        orden = np.argsort(vals)
        ax.barh(np.array(nomes)[orden], vals[orden],
                color=[PALETA.get(nomes[i], "#555") for i in orden],
                height=0.62)
        for i, v in enumerate(vals[orden]):
            ax.text(v + 0.3, i, br(v / 100, 3, pct=True), va="center",
                    fontsize=7)
        ax.set_xlabel("VaR 99,90% (% da exposição líquida)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title("Hierarquia de caudas", fontsize=9.5, color=AZUL,
                     loc="left")

        dists = {k: np.asarray(v) / 1e6
                 for k, v in results.get("distribuicoes", {}).items()}
        if dists:
            xmax = max(np.quantile(d, 0.999) for d in dists.values())
            bins = np.linspace(0, xmax, 90)
            picos = {k: float(np.histogram(d, bins=bins, density=True)[0]
                              .max()) for k, d in dists.items()}
            dep = [v for k, v in picos.items()
                   if "independente" not in k.lower()]
            ylim = (max(dep) if dep else max(picos.values())) * 1.25
            D.need(0.54, "Resultados")
            ax1 = D.eixo(0.21, "Resultados")
            ax2 = D.eixo(0.21, "Resultados")
            for nome, d in dists.items():
                rot = _rotulo(nome)
                indep = "independente" in nome.lower()
                cor = PALETA.get(rot, "#555")
                ax1.hist(d, bins=bins, density=True, histtype="step",
                         lw=1.0 if indep else 1.4,
                         ls="--" if indep else "-", label=rot, color=cor)
                srt = np.sort(d)
                tail = 1 - np.arange(1, len(srt) + 1) / len(srt)
                ax2.semilogy(srt * 1e6 / total * 100, np.maximum(tail, 1e-6),
                             lw=1.0 if indep else 1.4,
                             ls="--" if indep else "-", label=rot,
                             color=cor)
            ax1.set_xlim(0, xmax)
            ax1.set_ylim(0, ylim)
            ax1.set_xlabel("Perda (R$ milhões)", fontsize=8)
            ax1.set_title("Distribuições de perda — corpo (escala dos "
                          "modelos com dependência)", fontsize=9.5,
                          color=AZUL, loc="left")
            ax1.legend(fontsize=6)
            ax1.tick_params(labelsize=7)
            ax1.spines[["top", "right"]].set_visible(False)
            ax2.set_title("Cauda P(L > x)", fontsize=9.5, color=AZUL,
                          loc="left")
            ax2.set_xlabel("Perda (% da exposição líquida)", fontsize=8)
            ax2.set_ylim(1e-5, 1)
            ax2.axhline(1e-3, color="k", ls=":", lw=0.7)
            ax2.grid(alpha=0.25, which="both")
            ax2.legend(fontsize=6)
            ax2.tick_params(labelsize=7)
            ax2.spines[["top", "right"]].set_visible(False)

        # ================= diagnósticos =================
        D.nova("Diagnósticos")
        D.h1("Diagnósticos de validação")
        D.par("Testes automáticos de consistência interna da execução: "
              "aderência da EL simulada à analítica dentro do erro de Monte "
              "Carlo, verificação das calibrações (a correlação de default "
              "implícita é recalculada a partir dos parâmetros calibrados), "
              "ordenação teórica das caudas, precisão do quantil 99,9% por "
              "estatística de ordem e coerência dos módulos de agregação, "
              "capital e migração.", color=CINZA)
        drows = _diagnosticos(results, prm, pooled, mig, rel_agg)
        D.tabela(["Teste", "Resultado", "Critério", "Status"], drows,
                 [0.33, 0.22, 0.33, 0.12],
                 0.030 + 0.0215 * len(drows), "Diagnósticos",
                 esquerda=(0, 2))
        ax_d = D.fig.axes[-1]
        tb = ax_d.tables[0]
        for i, r in enumerate(drows, start=1):
            cor = {"OK": "#1E6B3C", "Atenção": VERM}.get(r[3], AZUL2)
            tb[i, 3].set_text_props(color=cor, fontweight="bold")
        sint, itens = _interpretar(drows, prm.get("M"))
        D.h2("Significado dos status", "Diagnósticos")
        for st_, sig in LEGENDA_STATUS:
            D.par(f"• {st_}: {sig}")
        D.h2("Leitura dos resultados", "Diagnósticos")
        D.par(sint)
        for st_, teste, leitura, acao in itens:
            D.par(f"• {st_} — {teste}: {leitura} Ação: {acao}",
                  color=VERM if st_ == "Atenção" else AZUL2)

        # ----- capital / setores -----
        if irb:
            D.nova("Capital e concentração")
            D.h1("Capital regulatório e concentração")
            D.kv([
                ("Capital IRB (99,9%)",
                 f"R$ {mi(irb.get('capital_total_999'))} mi   "
                 f"({br(irb.get('capital_pct_exposicao'), 3, pct=True)} da "
                 "exposição líquida)"),
                ("Ajuste de granularidade (GA)",
                 mi(irb.get("ajuste_granularidade"))),
                ("Capital com GA (R$ mi)", mi(irb.get("capital_com_ga"))),
                ("RWA equivalente (12,5×, R$ mi)", mi(irb.get("rwa_equivalente"))),
                ("R regulatório ponderado",
                 br(irb.get("rho_basel_ponderado"), 3, pct=True)),
            ], "Capital e concentração")
            cp = results.get("padronizado")
            if cp:
                ps_ = cp["piso"]
                D.y -= 0.010
                D.kv([
                    ("RWA IRB (modelos internos, R$ mi)", mi(ps_["rwa_irb"])),
                    ("RWA padronizado (Res. BCB 229, R$ mi)", mi(ps_["rwa_cpad"])),
                    ("IRB / padronizado", br(ps_["razao_irb_cpad"], 3)),
                    ("Piso: 72,5% × padronizado (R$ mi)", mi(ps_["piso"])),
                    ("RWA final (art. 34, Res. BCB 303)",
                     mi(ps_["rwa_final"]) + ("  — piso vinculante"
                                             if ps_["vinculante"] else "")),
                ], "Capital e concentração")
            ad = results.get("inadimplidos")
            if ad:
                r_ = ad["resumo"]
                D.y -= 0.010
                D.h2("Exposições em default (sem write-off)",
                     "Capital e concentração")
                D.par(f"{r_['contratos']:,} exposições em default ".replace(",", ".")
                      + f"(EAD R$ {mi(r_['ead'])} mi) foram separadas dos "
                      "modelos de simulação. Provisão pela ELBE(t), curva de "
                      f"LGD por tempo em default ({r_['curva']}, baixa em "
                      f"{br(r_['prazo_baixa'], 0)} meses); capital "
                      "K = LGD_D − ELBE, com LGD_D = ELBE + k(1 − ELBE), "
                      f"k = {br(r_['k'], 3)}; perda em 12 meses = EAD × "
                      "[ELBE(t+12) − ELBE(t)].")
                D.kv([
                    ("Provisão requerida — ELBE (R$ mi)",
                     mi(r_["provisao_elbe"])),
                    ("Capital IRB do default (R$ mi)", mi(r_["capital_irb"])),
                    ("RWA IRB / RWACPAD do default (R$ mi)",
                     f"{mi(r_['rwa_irb'])} / {mi(r_['rwa_cpad'])}"),
                    ("Perda incremental em 12 meses (R$ mi)",
                     mi(r_["perda_incremental_12m"])),
                ], "Capital e concentração")
            seg = irb.get("por_segmento")
            if seg is not None and len(seg):
                D.y -= 0.010
                D.h2("Capital IRB por categoria (Res. BCB 303/2023)",
                     "Capital e concentração")
                lin_s = [[r.descricao, str(r.linhas),
                          mi(r.exposicao_liquida),
                          br(r.R_ponderado, 3, pct=True),
                          mi(r.capital), br(r.capital_pct, 3, pct=True)]
                         for r in seg.itertuples(index=False)]
                D.tabela(["Categoria", "Linhas", "Expos. líquida", "R",
                          "Capital", "Capital/expos."], lin_s,
                         [0.31, 0.09, 0.16, 0.12, 0.16, 0.16],
                         0.030 + 0.0215 * len(lin_s),
                         "Capital e concentração")
            if mf and mf.get("contrib_por_setor") is not None:
                D.y -= 0.012
                D.h2("Decomposição setorial (contribuições de Euler ao ES)",
                     "Capital e concentração")
                cps = mf["contrib_por_setor"]
                linhas = [[str(r.setor), br(r.exposicao_pct, 3, pct=True),
                           br(r.contrib_ES_pct, 3, pct=True),
                           br(r.razao_risco_exposicao, 3)]
                          for r in cps.itertuples(index=False)]
                D.tabela(["Setor", "Exposição", "Contrib. ES",
                          "Razão risco/exposição"], linhas,
                         [0.34, 0.22, 0.22, 0.22],
                         0.030 + 0.0215 * len(linhas),
                         "Capital e concentração")
                D.par(f"HHI setorial: {br(mf.get('hhi_setorial'), 5)}. "
                      "Razão acima de 1 indica setor que contribui ao risco "
                      "de cauda mais do que à exposição.", color=CINZA)

        # ----- migração / ECL -----
        if mig is not None:
            ecl_df, sim_m = mig.get("ecl"), mig.get("sim")
            D.nova("Migração e ECL")
            D.h1("Migração, ECL e horizonte multiperíodo")
            e12 = float(ecl_df["ECL_12m"].sum())
            eli = float(ecl_df["ECL_lifetime"].sum())
            pares = [("ECL 12 meses",
                      f"R$ {mi(e12)} mi   ({br(e12/total, 3, pct=True)} da "
                      "exposição líquida)"),
                     ("ECL lifetime",
                      f"R$ {mi(eli)} mi   ({br(eli/e12 if e12 else 0, 3)}× o "
                      "12 meses)")]
            if "provisao" in ecl_df.columns:
                pares.append(("Provisão por estágios",
                              f"R$ {mi(float(ecl_df['provisao'].sum()))} mi"))
                if "estagio" in ecl_df.columns:
                    for e_, g_ in ecl_df.groupby("estagio"):
                        pares.append((f"   Estágio {int(e_)} ({len(g_):,} "
                                      "contratos)".replace(",", "."),
                                      f"R$ {mi(float(g_['provisao'].sum()))} mi"))
            if sim_m and sim_m.get("T"):
                pares += [(f"VaR 99,90% em {int(sim_m['T'])} anos",
                           mi(sim_m["VaR"].get("99.90%"))),
                          (f"ES 99,90% em {int(sim_m['T'])} anos",
                           mi(sim_m["ES"].get("99.90%"))),
                          ("EL multiperíodo descontada (R$ mi)", mi(sim_m["EL"]))]
            D.kv(pares, "Migração e ECL")
            if sim_m and sim_m.get("T") and len(np.atleast_1d(
                    sim_m.get("perdas", []))) > 10:
                ax = D.eixo(0.24, "Migração e ECL")
                from .report_tex import _fig_perdas_horizonte
                v9 = sim_m["VaR"].get("99.90%")
                _fig_perdas_horizonte(
                    ax, np.asarray(sim_m["perdas"]) / 1e6,
                    v9 / 1e6 if v9 == v9 else v9, sim_m["EL"] / 1e6,
                    cor=AZUL, verm=VERM)
                ax.set_title(f"Perdas descontadas acumuladas em "
                             f"{int(sim_m['T'])} anos", fontsize=9.5,
                             color=AZUL, loc="left")
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)

        # ================= limitações + referências =================
        D.nova("Limitações")
        D.h1("Limitações e condições de uso")
        for t in [
            "LGD determinística: a severidade é tratada como parâmetro "
            "fixo por contrato; extensões estocásticas (Beta) são "
            "documentadas na literatura [13].",
            "Correlações calibradas a alvos prudenciais informados pelo "
            "usuário, não estimadas da série histórica da carteira — a "
            "sensibilidade a esses alvos deve ser testada.",
            "PD do grau atribuída uniformemente dentro de cada grupo "
            "homogêneo; o coeficiente de variação de PD intra-bucket é o "
            "diagnóstico de adequação (desigualdade de Jensen).",
            "Matriz de transição homogênea no tempo; horizontes acima de "
            "5 anos devem ser lidos como cenário exploratório.",
            "Exposições em default: tratamento determinístico — ELBE pela "
            "curva de LGD em default e LGD_D por proxy (margem k); a "
            "incerteza da recuperação não é simulada.",
            "Monte Carlo simples: erro de simulação no quantil 99,9% da "
            "ordem de 1-2% com M = 100.000; use a mesma semente para "
            "reprodutibilidade.",
            "Este documento é gerado automaticamente a partir dos "
            "parâmetros escolhidos pelo usuário e não substitui a "
            "validação independente exigida pela governança de modelos "
            "da instituição.",
        ]:
            D.par("• " + t)
        D.y -= 0.01
        D.h1("Referências bibliográficas")
        D.par("Fontes públicas e auditáveis; a execução é reprodutível "
              "pelos parâmetros e semente reportados na seção 1.",
              color=CINZA)
        for r in REFS:
            D.par(r, size=7.8, width=118, secao="Referências")

        # ================= apêndice: base de entrada =================
        if descritiva:
            ds, g, cc = descritiva, descritiva["geral"], descritiva["concentracao"]
            sec_ap = "Apêndice A"
            D.nova(sec_ap)
            D.h1("Apêndice A — Análise descritiva da base de entrada")
            D.par(f"Base original ({nome_base or 'informada pelo usuário'}), "
                  "antes de qualquer agregação.", color=CINZA, secao=sec_ap)
            D.kv([("Linhas", br(g["linhas"], 0)),
                  ("Contrapartes distintas", br(g["contrapartes"], 0)),
                  ("Exposição total (R$ mi)", mi(g["ead_total"])),
                  ("Perda esperada", f"R$ {mi(g['el_total'])} mi "
                   f"({br(g['el_pct'], 3, pct=True)} da severidade)"),
                  ("PD média ponderada", br(g["pd_media_ponderada"], 3, pct=True)),
                  ("LGD média ponderada", br(g["lgd_media_ponderada"], 3, pct=True)),
                  ("HHI / nº efetivo", f"{br(cc['hhi'], 6)} / {br(cc['n_efetivo'])}"),
                  ("10 maiores / 1% maior", f"{br(cc['top10'], 3, pct=True)} / "
                   f"{br(cc['top1pct'], 3, pct=True)}"),
                  ("Gini das exposições", br(cc["gini"]))], sec_ap)
            q = ds["qualidade"]
            D.y -= 0.01
            D.tabela(["Coluna", "Tipo", "Preench.", "Não preench.", "Observação"],
                     [[r.coluna, r.tipo, br(r.preenchimento, 3, pct=True),
                       br(r.nao_preenchidos, 0), r.observacao]
                      for r in q.itertuples(index=False)],
                     [0.22, 0.14, 0.14, 0.14, 0.36],
                     0.022 * (len(q) + 1), sec_ap, esquerda=(0, 1, 4))
            nu = ds["numericas"]
            if len(nu):
                pc = lambda var: var in ("PD", "LGD", "LTV")

                def fv(v, var):
                    if var.startswith("Exposição"):
                        return br(v / 1e3, 2)          # por contrato: R$ mil
                    return br(v, 3, pct=pc(var))
                D.tabela(["Variável", "n", "Média", "Mín.", "Mediana", "p99", "Máx."],
                         [[r.variavel.replace("Exposição (EAD)",
                                              "Exposição (R$ mil)"), br(r.n, 0),
                           fv(r.media, r.variavel), fv(r.min, r.variavel),
                           fv(r.mediana, r.variavel), fv(r.p99, r.variavel),
                           fv(r.max, r.variavel)]
                          for r in nu.itertuples(index=False)],
                         [0.22, 0.11, 0.13, 0.13, 0.13, 0.14, 0.14],
                         0.022 * (len(nu) + 1), sec_ap)
            for c, (rot, t) in ds["categoricas"].items():
                D.h2(f"Composição por {rot.lower() if rot != 'Categoria IRB' else rot}",
                     secao=sec_ap)
                D.tabela([rot, "Linhas", "EAD (R$ mi)", "% EAD", "PD média", "% EL"],
                         [[str(getattr(r, c)), br(r.linhas, 0), mi(r.ead),
                           br(r.ead_pct, 3, pct=True), br(r.pd_media, 3, pct=True),
                           br(r.el_pct, 3, pct=True)]
                          for r in t.itertuples(index=False)],
                         [0.26, 0.12, 0.17, 0.14, 0.15, 0.14],
                         0.022 * (len(t) + 1), sec_ap)
        D.fecha()
