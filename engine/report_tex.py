"""
Paper de validação em LaTeX — mesma diagramação da documentação técnica.

Monta o código-fonte LaTeX a partir dos resultados da execução (dados,
parâmetros, fundamentação teórica dos modelos efetivamente executados,
resultados, diagnósticos de validação, capital, migração, limitações e
referências), gera as figuras em PDF vetorial e compila com pdflatex.
Sem pdflatex no ambiente, `latex_disponivel()` devolve False e o chamador
recorre ao gerador em matplotlib (report_paper.gerar_paper).
"""
import datetime as dt
import io
import os
import shutil
import subprocess
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fontes TrueType embutidas (tipo 42): leitores de PDF renderizam o texto
# das figuras de forma consistente (o tipo 3 padrão falha em alguns)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

from .report_paper import _FORMULA
from .report_paper import (LEGENDA_STATUS, PALETA, _diagnosticos,
                           _interpretar, _parametros, _rotulo, br)


def latex_disponivel() -> bool:
    return shutil.which("pdflatex") is not None


# ----------------------------------------------------------------- texto
_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_UNI = {"10⁻⁹": r"$10^{-9}$", "≤": r"$\le$", "≥": r"$\ge$", "±": r"$\pm$", "×": r"$\times$",
        "√": r"$\surd$", "σ": r"$\sigma$", "ρ": r"$\rho$", "ν": r"$\nu$",
        "−": r"$-$", "→": r"$\to$", "∝": r"$\propto$", "·": r"$\cdot$",
        "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$",
        "δ": r"$\delta$", "λ": r"$\lambda$", "μ": r"$\mu$",
        "ξ": r"$\xi$", "Φ": r"$\Phi$", "Σ": r"$\Sigma$", "Δ": r"$\Delta$",
        "χ": r"$\chi$", "²": r"$^2$", "≈": r"$\approx$"}


def tx(s) -> str:
    """Escapa texto livre para LaTeX (inclui símbolos unicode usuais).
    Trechos marcados como fórmula, ⟦tex¦texto⟧ (ver report_paper.fm), saem
    em modo matemático; o restante é escapado."""
    s = str(s)
    if "⟦" in s:
        partes, pos = [], 0
        for m in _FORMULA.finditer(s):
            partes.append(_tx_texto(s[pos:m.start()]))
            partes.append("$" + m.group(1) + "$")
            pos = m.end()
        partes.append(_tx_texto(s[pos:]))
        return "".join(partes)
    return _tx_texto(s)


def _tx_texto(s) -> str:
    s = "".join(_ESC.get(ch, ch) for ch in str(s))
    for k, v in _UNI.items():
        s = s.replace(k, v)
    return (s.replace(r"$\rho$\_D", r"$\rho_D$")
             .replace(r"$\sigma$\_MC", r"$\sigma_{MC}$"))


MI = 1e6                       # saída monetária: R$ milhões
UNID = r"R\$ milhões"


def mi(v, dec=2):
    """Valor monetário (entrada em reais) apresentado em R$ milhões."""
    if v is None:
        return "---"
    return nb(float(v) / MI, dec)


def nb(v, dec=3, pct=False):
    """Número no padrão brasileiro, pronto para texto LaTeX."""
    return tx(br(v, dec, pct=pct))


def nm(v, dec=3, pct=False):
    """Número no padrão brasileiro para modo matemático ({,} decimal)."""
    return br(v, dec, pct=pct).replace(",", "{,}").replace("%", r"\%")


def _tabela_longa(cab, linhas, spec, legenda, rotulo, small=True):
    """Tabela que quebra entre páginas (longtable), repetindo o cabeçalho."""
    corpo = "\n".join(" & ".join(l) + r" \\" for l in linhas)
    n = len(cab)
    h = " & ".join(cab) + r" \\"
    return rf"""
{{{r'\small' if small else ''}\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{{spec}}}
\caption{{{legenda}}}\label{{{rotulo}}}\\
\toprule
{h}
\midrule
\endfirsthead
\multicolumn{{{n}}}{{l}}{{\emph{{Tabela~\ref{{{rotulo}}} (continuação)}}}}\\
\toprule
{h}
\midrule
\endhead
\midrule
\multicolumn{{{n}}}{{r}}{{\emph{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{corpo}
\end{{longtable}}}}
"""


def _tabela(cab, linhas, spec, legenda, rotulo, small=True, H=False):
    corpo = "\n".join(" & ".join(l) + r" \\" for l in linhas)
    return rf"""
\begin{{table}}[{'H' if H else 'htb'}]
\centering{r'\small' if small else ''}
\caption{{{legenda}}}
\label{{{rotulo}}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{{spec}}}
\toprule
{' & '.join(cab)} \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


PREAMBULO = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}
\renewcommand{\contentsname}{Sumário}
\renewcommand{\abstractname}{Resumo}
\renewcommand{\tablename}{Tabela}
\renewcommand{\figurename}{Figura}
\renewcommand{\refname}{Referências}
\usepackage{amsmath,amssymb,bm}
\usepackage[margin=2.4cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{array}
\usepackage[activate=false]{microtype}
\emergencystretch=1.5em
\usepackage{xcolor}
\definecolor{linkblue}{RGB}{25,60,110}
\definecolor{okgreen}{RGB}{30,107,60}
\definecolor{alertred}{RGB}{176,36,24}
\usepackage[colorlinks=true,linkcolor=linkblue,citecolor=linkblue,urlcolor=linkblue]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\newcommand{\VaR}{\mathrm{VaR}}
\newcommand{\ES}{\mathrm{ES}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Ind}{\mathbb{I}}
"""

REFS_TEX = [
    r"Vasicek, O.~A. (2002). The distribution of loan portfolio value. \emph{Risk}, 15(12), 160--162.",
    r"Gordy, M.~B. (2003). A risk-factor model foundation for ratings-based bank capital rules. \emph{Journal of Financial Intermediation}, 12(3), 199--232.",
    r"Basel Committee on Banking Supervision (2005). \emph{An Explanatory Note on the Basel II IRB Risk Weight Functions}. Basileia: BIS.",
    r"Basel Committee on Banking Supervision (2006). \emph{International Convergence of Capital Measurement and Capital Standards: A Revised Framework}. Basileia: BIS.",
    r"Gourieroux, C.; Laurent, J.-P.; Scaillet, O. (2000). Sensitivity analysis of Values at Risk. \emph{Journal of Empirical Finance}, 7(3--4), 225--245.",
    r"Martin, R.; Wilde, T. (2002). Unsystematic credit risk. \emph{Risk}, 15(11), 123--128.",
    r"Gordy, M.~B.; Lütkebohmert, E. (2013). Granularity adjustment for regulatory capital assessment. \emph{International Journal of Central Banking}, 9(3), 33--71.",
    r"Credit Suisse First Boston (1997). \emph{CreditRisk\textsuperscript{+}: A Credit Risk Management Framework}. Londres: CSFB.",
    r"Tasche, D. (2008). Capital allocation to business units and sub-portfolios: the Euler principle. In: Resti, A. (org.). \emph{Pillar II in the New Basel Accord}. Londres: Risk Books, 423--453.",
    r"Belkin, B.; Suchower, S.; Forest Jr., L.~R. (1998). A one-parameter representation of credit risk and transition matrices. \emph{CreditMetrics Monitor}, 3º tri., 46--56.",
    r"Frey, R.; McNeil, A.~J. (2003). Dependent defaults in models of portfolio credit risk. \emph{Journal of Risk}, 6(1), 59--92.",
    r"Merton, R.~C. (1974). On the pricing of corporate debt: the risk structure of interest rates. \emph{Journal of Finance}, 29(2), 449--470.",
    r"Bolder, D.~J. (2018). \emph{Credit-Risk Modelling: Theoretical Foundations, Diagnostic Tools, Practical Examples, and Numerical Recipes in Python}. Cham: Springer. [Referencial de implementação do aplicativo.]",
    r"IFRS Foundation (2014). \emph{IFRS 9: Financial Instruments}. Londres. Conselho Monetário Nacional (2021). \emph{Resolução nº 4.966, de 25 de novembro de 2021}. Brasília: BCB.",
    r"Conselho Monetário Nacional (2017). \emph{Resolução nº 4.557, de 23 de fevereiro de 2017} (estrutura de gerenciamento de riscos e de capital). Brasília: BCB.",
    r"Banco Central do Brasil (2023). \emph{Resolução BCB nº 303, de 16 de março de 2023} (abordagem IRB para risco de crédito; revoga a Circular nº 3.648/2013). Brasília: BCB.",
    r"Banco Central do Brasil (2022). \emph{Resolução BCB nº 229, de 12 de maio de 2022} (RWA de risco de crédito pela abordagem padronizada, RWA\textsubscript{CPAD}; texto consolidado). Brasília: BCB.",
    r"David, H.~A.; Nagaraja, H.~N. (2003). \emph{Order Statistics}. 3.\,ed. Hoboken: Wiley.",
    r"Glasserman, P. (2004). \emph{Monte Carlo Methods in Financial Engineering}. Nova York: Springer.",
    r"Gordy, M.~B. (2002). Saddlepoint approximation of CreditRisk\textsuperscript{+}. \emph{Journal of Banking \& Finance}, 26(7), 1335--1353.",
    r"Basel Committee on Banking Supervision (2018). \emph{Stress testing principles}. Basileia: BIS.",
    r"Network for Greening the Financial System (2024). \emph{NGFS Climate Scenarios for central banks and supervisors}. Paris: NGFS.",
    r"Conselho Monetário Nacional (2021). \emph{Resolução CMN nº 4.943, de 15 de setembro de 2021} (riscos social, ambiental e climático na estrutura de gerenciamento de riscos). Brasília: BCB.",
    r"Conselho Monetário Nacional (2025). \emph{Resolução CMN nº 5.226, de 26 de junho de 2025} (altera a Resolução nº 4.557 quanto a testes de estresse com cenários definidos pelo Banco Central do Brasil e ao envio estruturado dos resultados). Brasília: BCB.",
]


# ---------------------------------------------------------------- figuras
def _fig_hierarquia(med, total, col999, path):
    sub = med.dropna(subset=[col999])
    nomes = [_rotulo(n) for n in sub["modelo"]]
    vals = sub[col999].to_numpy() / total * 100
    o = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(nomes) + 0.9))
    ax.barh(np.array(nomes)[o], vals[o], height=0.62,
            color=[PALETA.get(nomes[i], "#8A8F98") for i in o])
    for i, v in enumerate(vals[o]):
        ax.text(v + 0.3, i, br(v / 100, 3, pct=True), va="center",
                fontsize=8)
    ax.set_xlabel("VaR 99,90% (% da exposição líquida)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_distribuicoes(dists, total, path, unidade="R$ milhões"):
    xmax = max(np.quantile(d, 0.999) for d in dists.values())
    bins = np.linspace(0, xmax, 90)
    picos = {k: float(np.histogram(d, bins=bins, density=True)[0].max())
             for k, d in dists.items()}
    dep = [v for k, v in picos.items() if "independente" not in k.lower()]
    ylim = (max(dep) if dep else max(picos.values())) * 1.25
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nome, d in dists.items():
        rot = _rotulo(nome)
        ind = "independente" in nome.lower()
        kw = dict(color=PALETA.get(rot, "#555"), lw=1.1 if ind else 1.6,
                  ls="--" if ind else "-", label=rot)
        ax[0].hist(d, bins=bins, density=True, histtype="step", **kw)
        srt = np.sort(d)
        tail = 1 - np.arange(1, len(srt) + 1) / len(srt)
        ax[1].semilogy(srt / total * 100, np.maximum(tail, 1e-6), **kw)
    ax[0].set_xlim(0, xmax)
    ax[0].set_ylim(0, ylim)
    ax[0].set_title("Corpo (até 99,9%)", fontsize=10)
    ax[0].set_xlabel(f"Perda ({unidade})")
    ax[0].legend(fontsize=7)
    ax[1].set_title("Cauda P(L > x)", fontsize=10)
    ax[1].set_ylim(1e-5, 1)
    ax[1].axhline(1e-3, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("Perda (% da exposição líquida)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _crp_cal_txt(results, prm):
    """Parágrafo sobre a calibração (w, v) do CreditRisk+."""
    c = results.get("calibracao_crplus")
    if not c:
        return ""
    alvo, imp = c["rho_default_alvo"], c["rho_default_implicita"]
    base = (r" Para dois devedores com a PD média $\bar p$ e intensidade $\lambda$ que a reproduz, a probabilidade de \emph{default} conjunto é $\Pr(D_1D_2)=1-2(1-\bar p)+e^{-2\lambda(1-w)}(1+2\lambda w/v)^{-v}$, e a correlação de \emph{default} implícita é $\rho_D=\bigl(\Pr(D_1D_2)-\bar p^{\,2}\bigr)/\bigl(\bar p(1-\bar p)\bigr)$.")
    if c["calibrado"]:
        t = (rf"""

\paragraph{{Calibração.}} Os parâmetros são calibrados à mesma correlação de \emph{{default}} alvo dos demais modelos.{base} Com todo o risco no fator ($w=1$), a dispersão $v$ é a raiz de $\rho_D(v)=\rho_D^{{\mathrm{{alvo}}}}$: nesta execução, $v={nm(c['v'])}$, com $\rho_D={nm(imp, 3, pct=True)}$ (alvo ${nm(alvo, 3, pct=True)}$).""")
        if not c.get("atingivel", True):
            t += r" O alvo não é atingível com $w=1$; usou-se o valor mais próximo."
    else:
        t = (rf"""

\paragraph{{Calibração.}} Os parâmetros $w$ e $v$ foram informados pelo usuário, sem calibração ao alvo comum.{base} Nesta execução, $\rho_D={nm(imp, 3, pct=True)}$, contra o alvo de ${nm(alvo, 3, pct=True)}$ dos demais modelos, e a comparação entre famílias deve levar essa diferença em conta.""")
    return t + r" Quando $v$ é pequeno, o fator gama tem cauda pesada: nos cenários extremos, $\lambda_n S$ fica grande e a contagem Poisson soma vários \emph{defaults} do mesmo devedor, o que a versão Bernoulli não permite. Por isso as versões Poisson (simulação, clássica e ponto de sela) ficam acima da Bernoulli na cauda, e a distância entre elas mede o peso dessa aproximação."


def _sela_txt(cs):
    """Frase da verificação por ponto de sela (Gordy, 2002)."""
    vs = (cs.get("var_sela") or {}).get("99.90%")
    ve = (cs.get("var_exato") or {}).get("99.90%")
    if vs is None or not ve:
        return ""
    base = (r" Como verificação independente, o VaR é recalculado pela "
            r"aproximação de ponto de sela de Lugannani--Rice sobre a função "
            r"geradora de cumulantes $K(t)=\ln G_L(e^t)$ (Gordy, 2002): com "
            r"$\hat t$ tal que $K'(\hat t)=x$, "
            r"$\Pr(L>x)\approx 1-\Phi(\hat w)+\phi(\hat w)\bigl(1/\hat u-1/\hat w\bigr)$, "
            r"em que $\hat w=\mathrm{sgn}(\hat t)\sqrt{2[\hat t x-K(\hat t)]}$ e "
            r"$\hat u=\hat t\sqrt{K''(\hat t)}$.")
    if vs != vs:
        return base + " Nesta execução, a aproximação não convergiu."
    return base + (rf" Nesta execução, o VaR 99{{,}}9\% pelo ponto de sela "
                   rf"é R\$~{mi(vs)}~milhões, ante R\$~{mi(ve)}~milhões pela "
                   rf"inversão exata (diferença de {nb(vs / ve - 1, 3, pct=True)}).")


def _fig_default(ad, path):
    cv = ad["curva_ilustrativa"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(cv["meses"], cv["elbe"] * 100, color="#1f4e79", lw=1.6,
            label="ELBE (provisão)")
    ax.plot(cv["meses"], cv["lgd_d"] * 100, color="#c0504d", lw=1.4, ls="--",
            label="LGD$_D$")
    ax.fill_between(cv["meses"], cv["elbe"] * 100, cv["lgd_d"] * 100,
                    color="#c0504d", alpha=0.15, label="capital ($K$)")
    ax.set_xlabel("Meses em default", fontsize=8)
    ax.set_ylabel("% da EAD", fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_perdas_horizonte(ax, perdas, var, el=None, unidade="R$ milhões",
                          cor="#12395B", verm="#B02418", fs=8):
    """Histograma da perda acumulada com eixos explícitos: faixa de x até
    além do VaR (sem cauda invisível), y a partir de zero e rótulos dentro
    da área do gráfico."""
    p = np.sort(np.asarray(perdas, float))
    lo = float(p[0])
    hi = float(max(np.quantile(p, 0.9999), var if var == var else p[-1]))
    span = hi - lo if hi > lo else 1.0
    x0, x1 = max(0.0, lo - 0.03 * span), hi + 0.08 * span
    ax.hist(p[p <= x1], bins=np.linspace(x0, x1, 81), density=True,
            color=cor, alpha=0.9, linewidth=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.08)
    ax.set_ylim(bottom=0)
    trans = ax.get_xaxis_transform()
    if el is not None and el == el:
        ax.axvline(el, color="#555555", ls=":", lw=1.0)
        ax.text(el, 0.97, " EL", transform=trans, fontsize=fs - 1,
                color="#555555", va="top", ha="left")
    if var == var:
        ax.axvline(var, color=verm, ls="--", lw=1.2)
        direita = (var - x0) / (x1 - x0) > 0.75
        vtx = f"{var:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        ax.text(var, 0.97, (f"VaR 99,9% = {vtx}  " if direita
                            else f"  VaR 99,9% = {vtx}"),
                transform=trans, fontsize=fs - 1, color=verm, va="top",
                ha="right" if direita else "left")
    ax.set_xlabel(f"Perda descontada acumulada ({unidade})", fontsize=fs)
    ax.set_ylabel("Densidade", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.spines[["top", "right"]].set_visible(False)


def _fig_multiperiodo(sim, path, unidade="R$ milhões"):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    _fig_perdas_horizonte(ax, sim["perdas"], sim["VaR"].get("99.90%"),
                          sim.get("EL"), unidade)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------- documento

# ======================================================================
# Apêndice: análise descritiva da base de entrada
# ======================================================================

def _fig_desc_exposicao(desc, path):
    e = np.asarray(desc["ead_amostra"], float)
    e = e[e > 0]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax[0].hist(np.log10(e), bins=60, color="#1F4E79", alpha=0.85)
    ax[0].set_xlabel("log$_{10}$ da exposição")
    ax[0].set_ylabel("contratos" + (" (amostra)" if len(e) >= 100_000 else ""))
    ax[0].set_title("Distribuição das exposições", fontsize=10)
    x, y = desc["lorenz"]
    ax[1].plot(x, y, color="#B02418", lw=1.8, label="carteira")
    ax[1].plot([0, 1], [0, 1], color="#8A8F98", lw=0.9, ls="--",
               label="igualdade")
    ax[1].set_xlabel("fração acumulada dos contratos (menores primeiro)")
    ax[1].set_ylabel("fração acumulada da exposição")
    ax[1].set_title(f"Curva de Lorenz (Gini {br(desc['concentracao']['gini'], 3)})",
                    fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_desc_pd(desc, path):
    dados = [(k, v[v > 0]) for k, v in desc["pd_por_rating"] if len(v)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.boxplot([v for _, v in dados], showfliers=False, widths=0.55,
               medianprops=dict(color="#B02418"),
               boxprops=dict(color="#1F4E79"), whiskerprops=dict(color="#1F4E79"),
               capprops=dict(color="#1F4E79"))
    ax.set_xticks(range(1, len(dados) + 1))
    ax.set_xticklabels([k for k, _ in dados], rotation=0 if len(dados) <= 12
                       else 45, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("PD (escala logarítmica)")
    ax.set_xlabel("rating")
    ax.set_title("Dispersão da PD por rating (caixa: p25–p75; hastes: 1,5 × IQR)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _apendice_descritiva(desc, figdir, nome_base=""):
    """Código LaTeX do apêndice e lista de figuras geradas."""
    g, cc = desc["geral"], desc["concentracao"]
    figs = []
    T = [r"""
\clearpage
\appendix
\section{Análise descritiva da base de entrada}
\label{ap:descritiva}
""" + rf"""Este apêndice descreve a base de entrada \emph{{original}}{(' (' + tx(nome_base) + ')') if nome_base else ''}, {'contrato a contrato, antes da agregação' if g['linhas'] > 5_000 else 'tal como foi enviada'}, para que o validador avalie os dados que sustentam a execução. A base tem {nb(g['linhas'], 0)} linhas e {nb(g['contrapartes'], 0)} contrapartes distintas, exposição total de R\$~{mi(g['ead_total'])}~milhões e perda esperada de R\$~{mi(g['el_total'])}~milhões, isto é, {nb(g['el_pct'], 3, pct=True)} da severidade (PD média ponderada pela severidade de {nb(g['pd_media_ponderada'], 3, pct=True)} e LGD média ponderada pela exposição de {nb(g['lgd_media_ponderada'], 3, pct=True)}).
"""]
    # qualidade
    q = desc["qualidade"]
    lq = [[r"\texttt{" + tx(r.coluna) + "}", tx(r.tipo),
           nb(r.preenchimento, 3, pct=True), nb(r.nao_preenchidos, 0), tx(r.observacao)]
          for r in q.itertuples(index=False)]
    aus = desc.get("ausentes") or []
    T.append(r"""
\subsection{Completude e domínio das colunas}
A Tabela~\ref{tab:ap-qual} mostra o preenchimento de cada coluna presente e as verificações de domínio. """
             + (r"Colunas opcionais ausentes, tratadas pelos valores padrão: "
                + ", ".join(r"\texttt{" + tx(a) + "}" for a in aus) + "." if aus else
                "Todas as colunas opcionais estão presentes.")
             + _tabela_longa(["Coluna", "Tipo", "Preenchimento", "Não preenchidos",
                              "Observação"], lq,
                             r"llrr>{\raggedright\arraybackslash}p{4.6cm}",
                             "Completude e domínio das colunas da base de entrada.",
                             "tab:ap-qual"))
    # numéricas
    nu = desc["numericas"]
    if len(nu):
        _rot = {"Exposição (EAD)": "EAD (R\\$ mil)", "PD": "PD (\\%)",
                "LGD": "LGD (\\%)", "Prazo (anos)": "Prazo (anos)",
                "LTV": "LTV (\\%)", "Receita bruta (R$ mi)": "Receita (R\\$ mi)"}

        def f(v, var):
            if var.startswith("Exposição"):
                return nb(v / 1e3, 2)          # por contrato: R$ mil
            if var in ("PD", "LGD", "LTV"):
                return nb(v * 100, 3)          # em %, sem o símbolo
            return nb(v, 3)
        ln = [[_rot.get(r.variavel, tx(r.variavel)), nb(r.n, 0),
               f(r.media, r.variavel), f(r.desvio, r.variavel),
               f(r.min, r.variavel), f(r.p01, r.variavel),
               f(r.mediana, r.variavel), f(r.p99, r.variavel),
               f(r.max, r.variavel)]
              for r in nu.itertuples(index=False)]
        T.append(r"""
\subsection{Variáveis de risco}
""" + r"{\footnotesize\setlength{\tabcolsep}{3.5pt}" + _tabela(
            ["Variável", "$n$", "Média", "Desvio", "Mín.", "p1", "Mediana",
             "p99", "Máx."], ln, "lrrrrrrrr",
            "Estatísticas das variáveis numéricas por contrato. PD, LGD e LTV em percentual; exposição em R\\$ mil, para que os contratos de varejo não se reduzam a zero na escala de milhões.",
            "tab:ap-num", small=False, H=True).replace(
                r"\setlength{\tabcolsep}{4pt}", r"\setlength{\tabcolsep}{3pt}")
            + "}\n")
    # concentração
    lc = [["Índice de Herfindahl--Hirschman (HHI)", nb(cc["hhi"], 6)],
          ["Número efetivo de exposições ($1/\\mathrm{HHI}$)", nb(cc["n_efetivo"])],
          ["Participação das 10 maiores exposições", nb(cc["top10"], 3, pct=True)],
          ["Participação das 100 maiores exposições", nb(cc["top100"], 3, pct=True)],
          ["Participação do 1\\% maior das exposições", nb(cc["top1pct"], 3, pct=True)],
          ["Coeficiente de Gini das exposições", nb(cc["gini"])]]
    _fig_desc_exposicao(desc, os.path.join(figdir, "ap_exposicao.pdf"))
    figs.append("ap_exposicao.pdf")
    T.append(r"""
\subsection{Concentração das exposições}
""" + _tabela(["Indicador", "Valor"], lc, "lr",
              "Concentração das exposições da base de entrada.", "tab:ap-conc", H=True)
             + r"""
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{ap_exposicao.pdf}
\caption{Distribuição das exposições (escala logarítmica) e curva de Lorenz.}
\label{fig:ap-exp}
\end{figure}
""")
    # categóricas
    from .irb_bcb import SEGMENTOS as _SEG
    _min = {"Rating": "rating", "Setor": "setor", "Categoria IRB": "categoria IRB"}
    for c, (rot, t) in desc["categoricas"].items():
        if c == "segmento_irb":
            t = t.copy()
            t[c] = t[c].map(lambda k: _SEG.get(k, (k,))[0])
        lt = [[tx(getattr(r, c)), nb(r.linhas, 0), mi(r.ead),
               nb(r.ead_pct, 3, pct=True), nb(r.pd_media, 3, pct=True),
               nb(r.lgd_media, 3, pct=True), nb(r.el_pct, 3, pct=True)]
              for r in t.itertuples(index=False)]
        maior = t.sort_values("el", ascending=False).iloc[0]
        T.append(rf"""
\subsection{{Composição por {tx(_min.get(rot, rot))}}}
O maior contribuinte para a perda esperada é {tx(str(maior[c]))}, com {nb(maior['el_pct'], 3, pct=True)} da EL e {nb(maior['ead_pct'], 3, pct=True)} da exposição.
""" + _tabela_longa([tx(rot), "Linhas", "EAD (R\\$ mi)", "\\% EAD", "PD média", "LGD média",
                     "\\% EL"], lt, "lrrrrrr",
                    f"Composição da base por {tx(_min.get(rot, rot))} (PD ponderada pela severidade; LGD pela exposição).",
                    f"tab:ap-{c}"))
    if desc.get("pd_por_rating"):
        _fig_desc_pd(desc, os.path.join(figdir, "ap_pd_rating.pdf"))
        figs.append("ap_pd_rating.pdf")
        T.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{ap_pd_rating.pdf}
\caption{Dispersão da PD dentro de cada rating. Sobreposição entre classes vizinhas indica discriminação limitada da escala; a dispersão dentro da classe é a fonte do viés de Jensen nos grupos homogêneos.}
\label{fig:ap-pd}
\end{figure}
""")
    cz = desc.get("setor_rating")
    if cz is not None and cz.shape[1] <= 12:
        lz = [[tx(s)] + [nb(v, 3, pct=True) for v in row] + [nb(row.sum(), 3, pct=True)]
              for s, row in zip(cz.index, cz.to_numpy())]
        lz.append([r"\textbf{Total}"] + [nb(v, 3, pct=True) for v in cz.sum(axis=0)]
                  + [nb(cz.to_numpy().sum(), 3, pct=True)])
        T.append(r"""
\subsection{Setor $\times$ rating}
""" + r"{\scriptsize" + _tabela(["Setor"] + [tx(k) for k in cz.columns] + ["Total"], lz,
                                "l" + "r" * (cz.shape[1] + 1),
                                "Distribuição da exposição por setor e rating (\\% da exposição total).",
                                "tab:ap-cruz", small=False, H=True) + "}\n")
    return "".join(T), figs


def _consolidacao_tex(results):
    """Tabela da carteira total (não default + default) no apêndice."""
    from .alocacao import consolidacao
    df, info = consolidacao(results)
    if df is None or not len(df):
        return ""
    ins = info["insuficiencia"]
    ins_txt = (rf"a insuficiência de provisão do \emph{{default}}, R\$~{mi(ins)}~milhões (provisão constituída de R\$~{mi(info['provisao_constituida'])}~milhões contra R\$~{mi(info['provisao_requerida'])}~milhões requeridos pela ELBE)"
               if ins == ins else
               r"a insuficiência de provisão, que não pôde ser medida porque a provisão constituída não foi informada (tomada como nula)")
    lin = [[tx(r.modelo), mi(r.EL_ND), mi(r.VaR_ND), mi(r.ES_ND),
            mi(r.EL_total), mi(r.VaR_total), mi(r.ES_total),
            mi(r.capital_VaR), mi(r.capital_ES), mi(r.capital_alocado)]
           for r in df.itertuples(index=False)]
    corpo = "\n".join(" & ".join(l) + r" \\" for l in lin)
    return rf"""
\subsection*{{Consolidação: carteira não \emph{{default}} e em \emph{{default}}}}
A Tabela~\ref{{tab:consolidacao}} reúne as duas partes da carteira a 99{{,}}9\%, pelas expressões da Seção~\ref{{sec:teoria-default}}. A parte em \emph{{default}} (EAD de R\$~{mi(info['ead_default'])}~milhões) entra com o incremento da provisão em 12 meses, $\Delta L=$~R\$~{mi(info['delta_L'])}~milhões, na EL e no quantil, e com $K_D=$~R\$~{mi(info['K_D'])}~milhões apenas no quantil; assim, $\mathrm{{Capital}}=(\mathrm{{VaR}}_{{ND}}-\mathrm{{EL}}_{{ND}})+K_D$. O capital alocado soma ao capital pelo ES {ins_txt}.
\begin{{table}}[H]
\centering\footnotesize
\caption{{Carteira total a 99{{,}}9\% (R\$ milhões): não \emph{{default}} (ND), total e capital.}}
\label{{tab:consolidacao}}
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{3.6cm}}rrrrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{Não \emph{{default}}}} & \multicolumn{{3}}{{c}}{{Carteira total}} & \multicolumn{{3}}{{c}}{{Capital}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
Modelo & EL & VaR & ES & EL & VaR & ES & VaR$-$EL & ES$-$EL & Alocado \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _fig_estresse(tc, ts, path_c, path_s):
    """Figuras monocromáticas do capítulo de estresse: CEA por cenário e
    ano; variação do CEA nas sensibilidades."""
    estilos = ["-", "--", ":", "-."]
    marcas = ["o", "s", "^", "D", "v"]
    if tc is not None and len(tc):
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        for i, (nome, g) in enumerate(tc.groupby("cenario", sort=False)):
            ax.plot(g["ano"], g["CEA"] / MI, color="black",
                    ls=estilos[i % 4], marker=marcas[i % 5], ms=4, lw=1.1,
                    label=nome)
        ax.set_xlabel("Ano da projeção", fontsize=8)
        ax.set_ylabel("CEA (R\\$ milhões)", fontsize=8)
        ax.set_xticks(sorted(tc["ano"].unique()))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, color="0.6")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(path_c)
        plt.close(fig)
    if ts is not None and len(ts):
        fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(ts) + 0.9))
        y = np.arange(len(ts))[::-1]
        ax.barh(y, ts["dCEA"] / MI, color="0.55", edgecolor="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(ts["cenario"], fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Variação do CEA contra o cenário base, ano 1 "
                      "(R\\$ milhões)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.25, color="0.6")
        fig.tight_layout()
        fig.savefig(path_s)
        plt.close(fig)


def _capitulo_estresse(results, figdir):
    """Capítulo de testes de estresse (Res. CMN 4.557, arts. 11 a 18)."""
    from .stress import (tabela_cenarios, tabela_sensibilidades, reproducao,
                         ultimas, runs, comparacao_modelos)
    reg = [r for r in runs(results) if r.get("tipo", "regulatorio")
           == "regulatorio"]
    if not reg:
        return "", []
    est = reg[-1]
    r_cmn = REFS_TEX.index(next(r for r in REFS_TEX if "4.557" in r)) + 1
    r_966 = REFS_TEX.index(next(r for r in REFS_TEX if "4.966" in r)) + 1
    r_bcbs = REFS_TEX.index(next(r for r in REFS_TEX if "Stress testing" in r)) + 1
    r_5226 = REFS_TEX.index(next(r for r in REFS_TEX if "5.226" in r)) + 1
    tc, ts = tabela_cenarios(est), tabela_sensibilidades(est)
    figs = []
    _fig_estresse(tc, ts, os.path.join(figdir, "estresse_cea.pdf"),
                  os.path.join(figdir, "estresse_sens.pdf"))
    if tc is not None and len(tc):
        figs.append("estresse_cea.pdf")
    if ts is not None and len(ts):
        figs.append("estresse_sens.pdf")

    req = [
        ["Programa com metodologias, documentação e governança próprias", "art.~11, I", "capítulo e template de cenários documentam finalidade, choques e metodologia"],
        ["Análise de sensibilidade", "arts.~11, III, e 12, III", "choques de um parâmetro (PD, LGD, índice de crédito, recuperação do \\emph{default})"],
        ["Análise de cenários", "arts.~11, IV, e 12, III", "cenários plurianuais com variações simultâneas e coerentes de PD, LGD e recuperação"],
        ["Teste de estresse reverso", "arts.~11, V, e 12, III; S2 dispensado (art.~58)", "busca do índice de crédito que leva o CEA ao nível predefinido"],
        ["Premissas adequadamente severas", "art.~12, IV", "severidade medida pelo índice de crédito $z$ e pela probabilidade associada"],
        ["Elementos históricos e hipotéticos, não linearidades", "art.~15", "a PD responde de forma não linear a $z$; cenários podem trazer narrativa macroeconômica"],
        ["Integração com a avaliação de capital", "arts.~17, V e VI, e 40, III", "PE, PI e CEA por ano, comparados ao capital IRB, para o ICAAP e o plano de capital"],
        ["Aprovação dos cenários e do programa pelo conselho", "arts.~18 e 48, II, ``c''", "fora do escopo da ferramenta; os cenários entram como premissas assumidas"],
        ["Cenários definidos pelo Banco Central e envio estruturado", f"Res. CMN 5.226/2025 [{r_5226}]", "o template aceita os cenários fornecidos pelo regulador"],
    ]
    req_tab = _tabela(["Requisito", "Dispositivo (Res. CMN 4.557)", "Como o CreditLens atende"],
                      [[a, b, c] for a, b, c in req],
                      r">{\raggedright\arraybackslash}p{4.4cm}>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{6.6cm}",
                      "Requisitos regulatórios do programa de testes de estresse e como esta execução os atende.",
                      "tab:est-req", H=True)

    cd = est["cenarios_def"]
    cd_t = cd[cd["setor"].str.lower() == "todos"]

    def fz(v, d=2):
        return nb(v, d) if v == v and v is not None else "---"
    lcd = [[tx(r.cenario), tx(r.tipo), str(int(r.ano)), fz(r.z),
            fz(r.mult_pd, 2), fz(r.choque_lgd_pp, 1), fz(r.mult_k_default, 2),
            fz(r.pib_pct, 1), fz(r.desemprego_pct, 1), fz(r.selic_pct, 2),
            fz(r.ipca_pct, 1)] for r in cd_t.itertuples(index=False)]
    n_set = int((cd["setor"].str.lower() != "todos").sum())
    cd_tab = _tabela(["Cenário", "Tipo", "Ano", "$z$", "PD$\\times$", "LGD (p.p.)",
                      "$k\\times$", "PIB", "Desemp.", "Selic", "IPCA"], lcd,
                     r">{\raggedright\arraybackslash}p{3.0cm}lrrrrrrrrr",
                     "Definição dos cenários e das sensibilidades (choques para a carteira inteira; variáveis macroeconômicas em \\%).",
                     "tab:est-def", small=True, H=True)

    T = [rf"""
\section{{Testes de estresse}}
\label{{sec:estresse}}
Este capítulo apresenta o programa de testes de estresse de crédito da carteira, nos termos da Resolução CMN 4.557 [{r_cmn}], arts.~11 a 18, que define o teste de estresse como exercício de avaliação prospectiva dos potenciais impactos de eventos adversos e exige três metodologias: a análise de sensibilidade, que varia um parâmetro relevante; a análise de cenários, que varia de forma simultânea e coerente um conjunto de parâmetros ao longo de um período; e o teste de estresse reverso, que parte de um nível predefinido de resultado ou de capital e identifica os eventos que o produzem. Os resultados devem alimentar a avaliação da adequação de capital (art.~40, III), que no segmento S1 é o ICAAP, e os planos de contingência (art.~17); os cenários e o programa são aprovados pelo conselho de administração, com revisão ao menos anual (arts.~18 e 48). A Resolução CMN 5.226/2025 [{r_5226}] tornou explícita a possibilidade de o Banco Central exigir testes periódicos com cenários por ele definidos e o envio estruturado dos resultados. Os princípios do Comitê de Basileia [{r_bcbs}] orientam o desenho, e a Resolução CMN 4.966 [{r_966}] requer informação prospectiva também na perda esperada contábil. A Tabela~\ref{{tab:est-req}} relaciona os requisitos ao que esta execução entrega.
""" + req_tab + rf"""
\subsection{{Metodologia}}
O estresse é a segunda etapa da execução, independente do cálculo do capital: parte da carteira e dos resultados da execução de {tx(est.get('capital_de', '')) or 'referência'} e foi executado em {tx(est.get('executado_em', '')) or 'seguida'}, sem recalcular o capital. O programa usa um único modelo de portfólio, escolhido pelo usuário: \textbf{{{tx(est['modelo'])}}}, com {nb(est['M'], 0)} simulações por cenário e ano e a mesma semente em todos, de modo que as diferenças entre cenários venham dos choques, não do erro de simulação. Os parâmetros de dependência são os da execução principal, na métrica natural de cada família: a correlação de ativos $\rho={nb(est['rho_ativos'], 3)}$ nos modelos \emph{{threshold}} e no ASRF; a correlação de \emph{{default}} alvo $\rho_D={nb(est['rho_d'], 3)}$ na beta-binomial e no CreditRisk\textsuperscript{{+}}, recalibrados à PD média estressada.

\paragraph{{Ponte entre o cenário e a carteira.}} Cada ano de cenário traz um índice de crédito $z$, normal padrão, em que valores negativos são adversos. A PD do cenário desloca o limiar do modelo de Vasicek, recentrado para que $z=0$ reproduza a PD de 12 meses da carteira:
\begin{{equation}}
\mathrm{{PD}}_t=\Phi\Bigl(\Phi^{{-1}}(\mathrm{{PD}})-z_t\sqrt{{\rho/(1-\rho)}}\Bigr)\cdot m_t,
\end{{equation}}
com $m_t$ um multiplicador opcional por setor. A LGD recebe um choque aditivo em pontos percentuais, aplicado também à LGD de entrada do estoque em \emph{{default}}, e a margem $k$ da LGD em \emph{{default}} pode ser multiplicada, o que representa recuperações mais incertas. As variáveis macroeconômicas do template documentam a narrativa do cenário (art.~12, V) e não entram no cálculo: a tradução da macroeconomia em $z_t$ cabe a modelos satélite ou ao julgamento da área e é premissa assumida desta análise.

\paragraph{{Projeção e capital.}} Adota-se balanço constante: a carteira não \emph{{default}} é reposta a cada ano com o mesmo perfil, e os \emph{{defaults}} de cada ano formam uma coorte que envelhece na curva de \emph{{workout}}. Em cada ano $t$,
\begin{{equation}}
\mathrm{{PE}}_t=\sum_n \mathrm{{PD}}_{{t,n}}\,\mathrm{{LGD}}_{{t,n}}\,\mathrm{{EAD}}_n+\Delta L_t,\qquad
\mathrm{{PI}}_t=\bigl(\mathrm{{VaR}}^{{99{{,}}9\%}}_t-\mathrm{{EL}}_t\bigr)+K_{{D,t}},\qquad
\mathrm{{CEA}}_t=\mathrm{{PE}}_t+\mathrm{{PI}}_t,
\end{{equation}}
em que $\Delta L_t$ é o incremento da provisão do estoque em \emph{{default}} e das coortes anteriores, $K_{{D,t}}$ a margem de recuperação adversa desse estoque, e o VaR e a EL vêm do modelo escolhido com os parâmetros estressados. O capital IRB é recalculado com a PD estressada, para comparação. No ano~1 sem choque, a PE e a PI coincidem com a consolidação da Seção~\ref{{ap:cea}}.
"""]
    rp = reproducao(results, est)
    if rp:
        st_ = "dentro" if abs(rp["dif"]) <= 0.05 else "fora"
        T.append(rf"""
\paragraph{{Reprodução da base.}} O ano~1 do cenário sem choque dá CEA de R\$~{mi(rp['cea_estresse'])}~milhões, contra R\$~{mi(rp['cea_principal'])}~milhões na execução principal ({tx(rp['modelo_principal'])}): diferença de {nb(rp['dif'], 3, pct=True)}, {st_} da tolerância de 5\%, que acomoda o erro de simulação com menos cenários.
""")
    T.append(rf"""
\paragraph{{Definição dos cenários.}} Os cenários desta execução vêm de {tx(est.get('origem', ''))}.{(' Os seletores ' + tx('; '.join(est['setores_sem_carteira'])) + ' constam dos cenários mas não têm correspondência na carteira, e seus choques não têm efeito.') if est.get('setores_sem_carteira') else ''} A Tabela~\ref{{tab:est-def}} mostra os choques aplicados à carteira inteira{(f'; {n_set} linhas adicionais trazem choques específicos por setor, listados na exportação em Excel' if n_set else '')}.
""" + cd_tab)
    T.append(r"""
\subsection{Resultados}
""")
    if tc is not None and len(tc):
        lc = [[tx(r.cenario), str(int(r.ano)), nb(r.pd_media, 3, pct=True),
               mi(r.PE), mi(r.PI), mi(r.CEA), nb(r.CEA_pct_ead, 3, pct=True),
               nb(r.dCEA_pct, 3, pct=True), mi(r.PE_acum), mi(r.IRB_total)]
              for r in tc.itertuples(index=False)]
        pior = tc.loc[tc["CEA"].idxmax()]
        T.append(rf"""
A Tabela~\ref{{tab:est-cen}} traz a projeção de cada cenário. O maior CEA ocorre no cenário {tx(pior['cenario'])}, ano~{int(pior['ano'])}: R\$~{mi(pior['CEA'])}~milhões, {nb(pior['CEA_pct_ead'], 3, pct=True)} da EAD total e {nb(pior['dCEA_pct'], 3, pct=True)} acima do base no mesmo ano. A PE acumulada mede a perda que transitaria pelo resultado ao longo do cenário; a PI, o capital requerido ao fim de cada ano.
""" + _tabela(["Cenário", "Ano", "PD média", "PE", "PI", "CEA", "CEA/EAD", "$\\Delta$ vs base", "PE acum.", "IRB + D"],
              lc, r">{\raggedright\arraybackslash}p{2.0cm}rrrrrrrrr",
              "Análise de cenários: PE, PI e CEA por ano (R\\$ milhões).", "tab:est-cen", small=True, H=True)
            + r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{estresse_cea.pdf}
\caption{CEA por cenário ao longo da projeção.}
\end{figure}
""")
    if ts is not None and len(ts):
        ls = [[tx(r.cenario), nb(r.pd_media, 3, pct=True), mi(r.PE), mi(r.PI),
               mi(r.CEA), mi(r.dCEA), nb(r.dCEA_pct, 3, pct=True)]
              for r in ts.itertuples(index=False)]
        T.append(rf"""
\paragraph{{Análise de sensibilidade.}} A Tabela~\ref{{tab:est-sens}} varia um parâmetro de cada vez no ano~1, a partir do cenário base. A maior variação vem de {tx(ts.iloc[0]['cenario'])}, com {nb(ts.iloc[0]['dCEA_pct'], 3, pct=True)} no CEA.
""" + _tabela(["Choque", "PD média", "PE", "PI", "CEA", "$\\Delta$ CEA", "$\\Delta$ \\%"],
              ls, r">{\raggedright\arraybackslash}p{4.4cm}rrrrrr",
              "Análise de sensibilidade no ano 1 (R\\$ milhões).", "tab:est-sens", small=True, H=True)
            + r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{estresse_sens.pdf}
\caption{Variação do CEA por choque, contra o cenário base.}
\end{figure}
""")
    rv = est.get("reverso") or {}
    if rv.get("z") == rv.get("z") and rv.get("z") is not None:
        T.append(rf"""
\paragraph{{Teste de estresse reverso.}} O nível predefinido é um CEA de {nb(rv['alvo_mult'], 1)} vez o do cenário base no ano~1, R\$~{mi(rv['alvo'])}~milhões. Esse nível é atingido com índice de crédito $z^\ast={nb(rv['z'], 3)}$, que leva a PD média de {nb(rv['pd_media_base'], 3, pct=True)} para {nb(rv['pd_media'], 3, pct=True)}. A probabilidade de um ano tão ou mais adverso, $\Phi(z^\ast)$, é de {nb(rv['prob'], 3, pct=True)}. Quanto mais provável o evento que consome o capital, menor a folga da instituição; o resultado deve ser confrontado com o apetite a risco. A Resolução CMN 4.557 dispensa o estresse reverso no segmento S2 (art.~58).
""")
    cm = comparacao_modelos(results)
    if cm is not None and cm["modelo"].nunique() > 1:
        mods = list(dict.fromkeys(cm["modelo"]))
        pv = cm.pivot_table(index=["cenario", "ano"], columns="modelo",
                            values="CEA", sort=False)[mods]
        lm = [[tx(c), str(int(a))] + [mi(v) for v in row]
              for (c, a), row in zip(pv.index, pv.to_numpy())]
        T.append(rf"""
\paragraph{{Risco de modelo do estresse.}} O programa foi executado com {len(mods)} modelos sobre o mesmo capital e os mesmos cenários. A Tabela~\ref{{tab:est-modelos}} compara o CEA de cada um; a dispersão entre as colunas mede quanto o resultado depende da família de modelo escolhida.
""" + _tabela(["Cenário", "Ano"] + [tx(m) for m in mods], lm,
              r">{\raggedright\arraybackslash}p{2.2cm}r" + "r" * len(mods),
              "CEA por cenário e ano em cada modelo (R\\$ milhões).",
              "tab:est-modelos", small=True, H=True))
    T.append(r"""
\paragraph{Uso e limitações.} Os resultados servem à avaliação da adequação de capital, ao plano de capital, que projeta ao menos três anos (art.~41), e à calibragem do apetite a risco. São premissas assumidas, a validar caso a caso: o índice de crédito de cada cenário, que dispensa aqui um modelo satélite estimado; o balanço constante; os choques uniformes de LGD; a estabilidade das correlações sob estresse, que tende a subestimar a cauda em crises; e a escolha de um único modelo, cujo risco de modelo pode ser medido repetindo o exercício com outra família.
""")
    return "".join(T), figs


def _capitulo_customizado(results, figdir):
    """Estresse customizado (p.ex. climático): um bloco por exercício."""
    from .stress import ultimas
    E = ultimas(results, "customizado")
    if not E:
        return "", []
    r_cmn = REFS_TEX.index(next(r for r in REFS_TEX if "4.557" in r)) + 1
    r_ngfs = REFS_TEX.index(next(r for r in REFS_TEX if "NGFS" in r)) + 1
    r_4943 = REFS_TEX.index(next(r for r in REFS_TEX if "4.943" in r)) + 1
    T = [rf"""
\section{{Estresse customizado}}
\label{{sec:estresse-custom}}
Além do programa regulatório, a instituição pode propor exercícios próprios, com choques definidos livremente por setor, país, região, categoria IRB ou rating. São exercícios internos, de Pilar 2, cujos resultados informam a avaliação da adequação de capital e o apetite a risco, sem substituir o programa regulatório. Um uso típico é o risco climático: a Resolução CMN 4.557 [{r_cmn}] define os riscos climáticos de transição e físico (art.~38-C) e exige que o programa de testes de estresse inclua análise de cenários que os considere (art.~38-D, VII), dispositivos incluídos pela Resolução CMN 4.943 [{r_4943}] e observados a partir de 1º de dezembro de 2022 no S1 e no S2. Os cenários do NGFS [{r_ngfs}] são a referência usual para as narrativas.

Cada linha do arquivo de cenários seleciona uma fatia da carteira pelas colunas preenchidas, e a linha mais específica prevalece, herdando o que deixar vazio. Os choques possíveis são o índice de crédito $z$ e o multiplicador de PD, o choque aditivo de LGD, o choque percentual de EAD e o multiplicador da margem $k$ da LGD em \emph{{default}}. No modo \emph{{instantâneo}}, cada horizonte é um choque sobre a carteira atual, o que convém a horizontes longos como 2030 ou 2050; no modo de \emph{{projeção}}, os anos se sucedem com balanço constante, como no programa regulatório. O capital de cada horizonte é o CEA do modelo escolhido, como na Seção~\ref{{sec:estresse}}.
"""]
    figs = []
    for k, est in enumerate(E):
        t = est["resultados"]
        doc = est.get("documentacao") or {}
        b = est["base"]
        d0 = next(iter(doc.values()), {})
        T.append(rf"""
\subsection{{{tx(est.get('nome', 'Exercício'))}}}
Modelo: \textbf{{{tx(est['modelo'])}}}, com {nb(est['M'], 0)} simulações por horizonte; cenários de {tx(est.get('origem', ''))}; executado em {tx(est.get('executado_em', ''))} sobre o capital de {tx(est.get('capital_de', ''))}.
""")
        itens = [(lab, d0.get(c, "")) for c, lab in (
            ("objetivo", "Objetivo"), ("fonte", "Fonte dos cenários"),
            ("metodo_depara", "Método do de-para"),
            ("premissas", "Premissas assumidas"))]
        itens = [(a, b_) for a, b_ in itens if b_]
        if itens:
            T.append("\\begin{description}\n" + "\n".join(
                f"\\item[{tx(a)}] {tx(b_)}" for a, b_ in itens)
                + "\n\\end{description}\n")
        narr = [(n, d.get("descricao", "")) for n, d in doc.items()
                if d.get("descricao")]
        if narr:
            T.append("Narrativas: " + "; ".join(
                f"\\emph{{{tx(n)}}}, {tx((d[:1].lower() + d[1:]).rstrip('. '))}"
                for n, d in narr) + ".\n")
        if est.get("seletores_ausentes"):
            T.append("Os seletores " + tx("; ".join(est["seletores_ausentes"]))
                     + " não têm correspondência na carteira, e seus choques "
                     "não têm efeito.\n")
        lin = [[tx(r.cenario), str(r.horizonte), nb(r.pd_media, 3, pct=True),
                nb(r.lgd_media, 3, pct=True), mi(r.PE), mi(r.PI), mi(r.CEA),
                nb(r.CEA / est["ead_total"], 3, pct=True),
                nb(r.dCEA_pct, 3, pct=True)] for r in t.itertuples(index=False)]
        pior = t.loc[t["CEA"].idxmax()]
        rot = f"tab:cus-{k}"
        T.append(rf"""
Sem choque, o CEA é de R\$~{mi(b['CEA'])}~milhões. O maior valor ocorre em {tx(pior['cenario'])}, horizonte {pior['horizonte']}: R\$~{mi(pior['CEA'])}~milhões, {nb(pior['dCEA_pct'], 3, pct=True)} acima do base (Tabela~\ref{{{rot}}}).
""" + _tabela(["Cenário", "Horiz.", "PD média", "LGD média", "PE", "PI",
               "CEA", "CEA/EAD", "$\\Delta$ vs base"], lin,
              r">{\raggedright\arraybackslash}p{2.6cm}rrrrrrrr",
              f"Estresse customizado {tx(est.get('nome', ''))}: PE, PI e CEA por cenário e horizonte (R\\$ milhões).",
              rot, small=True, H=True))
        nf = f"estresse_cus_{k}.pdf"
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        for i, (nome, g) in enumerate(t.groupby("cenario", sort=False)):
            ax.plot(g["horizonte"].astype(str), g["CEA"] / MI, color="black",
                    ls=["-", "--", ":", "-."][i % 4], marker="os^Dv"[i % 5],
                    ms=4, lw=1.1, label=nome)
        ax.axhline(b["CEA"] / MI, color="0.5", lw=0.8, ls=(0, (1, 2)),
                   label="sem choque")
        ax.set_xlabel("Horizonte", fontsize=8)
        ax.set_ylabel("CEA (R\\$ milhões)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, color="0.6")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, nf))
        plt.close(fig)
        figs.append(nf)
        T.append(rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{{nf}}}
\caption{{CEA por cenário e horizonte: {tx(est.get('nome', ''))}.}}
\end{{figure}}
""")
        ps = est.get("por_setor")
        if ps is not None and len(ps):
            w = ps[(ps["cenario"] == pior["cenario"])
                   & (ps["horizonte"] == pior["horizonte"])].copy()
            w["var"] = w["PE"] / w["PE_base"] - 1
            w = w.sort_values("var", ascending=False)
            lw = [[tx(r.setor), mi(r.PE_base), mi(r.PE), nb(r.var, 3, pct=True)]
                  for r in w.itertuples(index=False)]
            T.append(_tabela(["Setor", "PE base", "PE estressada", "Variação"],
                             lw, r">{\raggedright\arraybackslash}p{4.0cm}rrr",
                             f"PE por setor no cenário mais severo ({tx(pior['cenario'])}, {pior['horizonte']}), em R\\$ milhões.",
                             f"tab:cus-setor-{k}", small=True, H=True))
    T.append(r"""
\paragraph{Leitura.} Os multiplicadores e choques do arquivo são premissas assumidas, e a tradução das variáveis dos cenários em parâmetros de risco é o ponto que mais pesa no resultado; deve ser documentada e validada à parte. Os choques instantâneos não capturam a adaptação da carteira ao longo do tempo, e a estabilidade das correlações sob estresse tende a subestimar a cauda.
""")
    return "".join(T), figs


def _apendice_cea(results, figdir, rel_agg=None, descritiva=None):
    """Apêndice: da base de entrada ao CEA = PE + PI, para todos os modelos."""
    from .fluxo_cea import (tabela_cea, conciliacao, fig_fluxo, ETAPAS,
                            ead_total)
    tv, pv = tabela_cea(results, "VaR")
    if tv is None or not len(tv):
        return "", []
    te, pe_ = tabela_cea(results, "ES")
    fig_fluxo(os.path.join(figdir, "fluxo_cea.pdf"), results, rel_agg,
              descritiva)
    conc, dif = conciliacao(results, rel_agg, descritiva)
    ead = ead_total(results)
    tem_def = bool((results.get("inadimplidos") or {}).get("resumo"))

    et = [[tx(a), tx(b), tx(c)] for a, b, c in ETAPAS]
    lc = [[tx(r.etapa), nb(r.linhas, 0), mi(r.ead),
           mi(r.pe) if r.pe == r.pe else "---", tx(r.observacao)]
          for r in conc.itertuples(index=False)]

    def cea_tab(d, p, med, rot):
        q = r"\mathrm{VaR}" if med == "VaR" else r"\mathrm{ES}"
        corpo = "\n".join(
            " & ".join([tx(r.modelo), mi(r.PE_ND), mi(r.PE_D), mi(r.PE),
                        mi(r.PI_ND), mi(r.PI_D), mi(r.PI), mi(r.CEA),
                        nb(rp.CEA, 3, pct=True)]) + r" \\"
            for r, rp in zip(d.itertuples(index=False),
                             p.itertuples(index=False)))
        return rf"""
\begin{{table}}[H]
\centering\footnotesize
\caption{{Formação do CEA por modelo, com a PI medida pelo ${q}_{{99{{,}}9\%}}$ (R\$ milhões).}}
\label{{{rot}}}
\setlength{{\tabcolsep}}{{2.6pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{3.7cm}}rrrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{PE}} & \multicolumn{{3}}{{c}}{{PI}} & \multicolumn{{2}}{{c}}{{CEA $=$ PE $+$ PI}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-9}}
Modelo & ND & D & Total & ND & D & Total & R\$ mi & \% EAD \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    dif_txt = ("A EAD consolidada coincide com a da base de entrada."
               if abs(dif) < 1.0 else
               rf"A EAD consolidada difere da base de entrada em R\$~{mi(dif)}~milhões, que corresponde a linhas descartadas na leitura e deve ser explicada.")
    d_txt = (r" Na parte em \emph{default}, a PE é o incremento da provisão em 12 meses, $\Delta L$, e a PI é a margem de recuperação no cenário adverso, $K_D=\sum\mathrm{EAD}\,(\mathrm{LGD}_D-\mathrm{ELBE})$; ambas são determinísticas e iguais em todos os modelos."
             if tem_def else
             r" Esta carteira não tem exposições em \emph{default}; a PE e a PI vêm apenas da carteira não \emph{default}.")
    a, b = tv.iloc[0], tv.dropna(subset=["CEA"]).iloc[-1]
    txt = rf"""
\clearpage
\section{{Da base de entrada ao capital econômico alocado}}
\label{{ap:cea}}
Este apêndice percorre a execução de ponta a ponta, da base de entrada ao capital econômico alocado (CEA), para todos os modelos do portfólio. Segue-se a convenção usual dos bancos para o VaR de crédito, tratado como a soma da perda esperada (PE), coberta pela provisão, com a perda inesperada (PI), coberta pelo capital:
\begin{{equation}}
\mathrm{{CEA}}=\mathrm{{PE}}+\mathrm{{PI}},\qquad
\mathrm{{PE}}=\mathrm{{EL}}_{{ND}}+\Delta L,\qquad
\mathrm{{PI}}=\bigl(\mathrm{{VaR}}_{{ND}}-\mathrm{{EL}}_{{ND}}\bigr)+K_D,
\end{{equation}}
em que ND indica a carteira não \emph{{default}}, simulada por cada modelo, e D a carteira em \emph{{default}}.{d_txt} Assim, o CEA coincide com o VaR da carteira total, e o capital propriamente dito é a PI. A Figura~\ref{{fig:fluxo-cea}} mostra o fluxo com os números desta execução, a Tabela~\ref{{tab:etapas}} descreve cada etapa e a Tabela~\ref{{tab:conciliacao}} concilia a base ao longo do caminho.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{fluxo_cea.pdf}}
\caption{{Fluxo da base de entrada ao CEA, com os números desta execução.}}
\label{{fig:fluxo-cea}}
\end{{figure}}
""" + _tabela(["Etapa", "O que faz", "Saída"], et,
              r">{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{8.4cm}>{\raggedright\arraybackslash}p{3.6cm}",
              "Etapas da execução, da base de entrada ao CEA.", "tab:etapas",
              H=True) + rf"""
A conciliação acompanha linhas, EAD e PE em cada etapa. {dif_txt}
""" + _tabela(["Etapa", "Linhas", "EAD", "PE", "Observação"], lc,
              r">{\raggedright\arraybackslash}p{4.4cm}rrr>{\raggedright\arraybackslash}p{5.0cm}",
              r"Conciliação da base ao longo da execução (R\$ milhões).",
              "tab:conciliacao", H=True) + rf"""
A Tabela~\ref{{tab:cea-var}} decompõe o CEA de cada modelo com a PI medida pelo VaR a 99{{,}}9\%, que é a convenção usual; a Tabela~\ref{{tab:cea-es}} repete a decomposição com o ES, como sensibilidade. Os modelos estão ordenados do maior para o menor CEA, e o percentual é sobre a EAD total da carteira (R\$~{mi(ead)}~milhões). Como todos os modelos partem da mesma PD, a PE varia pouco entre eles e reflete apenas o erro de simulação; a dispersão do CEA vem da PI, isto é, da forma da cauda de cada modelo. Pelo VaR, o CEA vai de R\$~{mi(b.CEA)}~milhões ({tx(b.modelo)}) a R\$~{mi(a.CEA)}~milhões ({tx(a.modelo)}).
""" + cea_tab(tv, pv, "VaR", "tab:cea-var") + cea_tab(te, pe_, "ES", "tab:cea-es")
    return txt, ["fluxo_cea.pdf"]


def _apendice_alocacao(results, primeiro, letra):
    """Apêndice com o capital por modelo (R$ milhões e % da EAD total)."""
    from .alocacao import tabela_alocacao, ALPHAS
    df, pct, ead, k_def = tabela_alocacao(results)
    if df is None or not len(df):
        return ""
    cols = [f"VaR {a}" for a in ALPHAS] + [f"ES {a}" for a in ALPHAS]
    niveis = [a.replace(".", "{,}").replace("%", r"\%") for a in ALPHAS]
    cab = (r"& \multicolumn{4}{c}{VaR $-$ EL} & \multicolumn{4}{c}{ES $-$ EL} \\"
           "\n" r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
           "Modelo & " + " & ".join(niveis + niveis) + r" \\")

    def corpo(d, fmt):
        out, ref = [], False
        for r in d.itertuples(index=False):
            lin = dict(zip(d.columns, r))
            if lin["tipo"] == "referencia" and not ref:
                out.append(r"\midrule")
                ref = True
            nome = tx(lin["modelo"])
            out.append(nome + " & " + " & ".join(fmt(lin[c]) for c in cols)
                       + r" \\")
        return "\n".join(out)

    def tab(d, fmt, leg, rot):
        return rf"""
\begin{{table}}[H]
\centering\footnotesize
\caption{{{leg}}}
\label{{{rot}}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{4.4cm}}rrrrrrrr}}
\toprule
{cab}
\midrule
{corpo(d, fmt)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    k_txt = (rf" Em todas as linhas de modelo soma-se o capital das exposições em \emph{{default}}, $\mathrm{{EAD}}\cdot(\mathrm{{LGD}}_D-\mathrm{{ELBE}})$, de R\$~{mi(k_def)}~milhões, que não depende do nível de confiança (Seção~\ref{{sec:capital}})."
             if k_def else "")
    return (r"""
\clearpage
""" + (r"\appendix" + "\n" if primeiro else "") + rf"""\section{{Alocação de capital por modelo}}
\label{{ap:alocacao}}
Este apêndice resume o capital que cada modelo atribuiria à carteira em cada nível de confiança. O capital econômico é a perda inesperada: $\mathrm{{VaR}}_\alpha-\mathrm{{EL}}$ ou $\mathrm{{ES}}_\alpha-\mathrm{{EL}}$, com a EL do próprio modelo. Os modelos estão ordenados do maior para o menor capital pelo $\mathrm{{ES}}_{{99{{,}}9\%}}-\mathrm{{EL}}$.{k_txt} As linhas de referência trazem o IRB e o capital regulatório final, após o piso de saída; ambos são definidos apenas a 99{{,}}9\%. A Tabela~\ref{{tab:aloc-rs}} apresenta os valores em R\$ milhões; a Tabela~\ref{{tab:aloc-pct}}, em percentual da EAD total da carteira (R\$~{mi(ead)}~milhões, incluídas as exposições em \emph{{default}}).
""" + tab(df, lambda v: mi(v),
          r"Capital por modelo e nível de confiança (R\$ milhões).",
          "tab:aloc-rs")
            + tab(pct, lambda v: nb(v, 2, pct=True),
                  r"Capital por modelo e nível de confiança, em percentual da EAD total.",
                  "tab:aloc-pct")
            + _consolidacao_tex(results))


def montar_tex(port, results, mig=None, nome_base="", extra=None,
               rel_agg=None, figdir=".", descritiva=None):
    """Devolve (código LaTeX, lista de figuras geradas em `figdir`)."""
    resumo = results["resumo"]
    prm = results.get("parametros", {})
    med = results["medidas"]
    total = resumo["exposicao_liquida_total"]
    unid = "R$ milhões"
    pooled = bool(results.get("pooled", False)) or any(
        "(pooled)" in m for m in med["modelo"])
    cal = results.get("calibracao_gaussiana", {})
    rho_a, pbar = cal.get("rho_ativos"), cal.get("pbar",
                                                 resumo["pd_media_ponderada"])
    modelos = list(med["modelo"])
    tem = lambda s: any(s in m for m in modelos)
    mf = results.get("multifator")
    irb = results.get("irb", {})
    ab = results.get("calibracao_beta") or (
        {"a": prm.get("beta_a"), "b": prm.get("beta_b")}
        if prm.get("beta_a") else None)
    col999 = next(c for c in med.columns if c.startswith("VaR 99.90"))
    agora = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    figs = []

    ref_g = None
    for alvo in ["Threshold Gaussiano multifatorial", "Threshold Gaussiano 1F"]:
        sel = med[med["modelo"].str.startswith(alvo)]
        if len(sel):
            ref_g = (_rotulo(sel.iloc[0]["modelo"]), float(sel.iloc[0][col999]))
            break

    T = [PREAMBULO]
    T.append(rf"""
\title{{\textbf{{Validação de Modelos}}\\[3mm]
\Large Modelos de VaR para Risco de Crédito em Bancos Brasileiros\\[4mm]
\normalsize De acordo com as Resoluções CMN nº~4.557/2017 (gerenciamento de riscos e de capital) e nº~4.966/2021 (perda esperada) e as Resoluções BCB nº~229/2022 (abordagem padronizada) e nº~303/2023 (abordagem IRB)}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
\noindent Este documento reporta uma execução do CreditLens (base de entrada: \emph{{{tx(nome_base) or 'informada pelo usuário'}}}): {nb(resumo['n_contrapartes'], 0)} linhas de modelagem, exposição líquida de R\$~{mi(total)}~milhões e perda esperada de R\$~{mi(resumo['perda_esperada'])}~milhões ({nb(resumo['perda_esperada'] / total, 3, pct=True)} da exposição). Apresenta a fundamentação teórica de cada modelo efetivamente executado com os parâmetros escolhidos, os resultados obtidos, uma bateria de diagnósticos automáticos de consistência e as referências bibliográficas que suportam as técnicas, para verificação independente."""
             + (rf""" No modelo de referência ({tx(ref_g[0])}), o VaR 99{{,}}90\% é de R\$~{mi(ref_g[1])}~milhões ({nb(ref_g[1] / total, 3, pct=True)} da exposição líquida).""" if ref_g else "")
             + rf""" A execução é reprodutível pela semente {tx(prm.get('seed', '---'))} e pelos parâmetros da Seção~\ref{{sec:param}}.
\end{{abstract}}
\tableofcontents
""")

    # ---------------- 1. dados ----------------
    linhas = [
        ["Base de entrada", tx(nome_base) or "---"],
        ["Linhas de modelagem", nb(resumo["n_contrapartes"], 0)],
        ["Representação", "agregada (grandes exposições + grupos homogêneos)" if pooled else "granular (nível contraparte)"],
        ["Exposição total (EAD, R\\$ milhões)", mi(resumo["ead_total"])],
        ["Exposição líquida ($\\sum c_n$, R\\$ milhões)", mi(total)],
        ["PD média ponderada ($\\bar p$)", nb(pbar, 3, pct=True)],
        ["LGD média", nb(resumo["lgd_media"])],
        ["Perda esperada (R\\$ milhões)", mi(resumo["perda_esperada"])],
        ["HHI de nomes / nº efetivo", f"{nb(resumo['hhi'], 5)} / {nb(resumo['n_efetivo'])}"],
        ["Maior exposição / top 10", f"{nb(resumo['maior_exposicao_pct'], 3, pct=True)} / {nb(resumo['top10_exposicao_pct'], 3, pct=True)}"],
    ]
    T.append(r"""
\section{A carteira e os dados de entrada}
Cada linha da carteira de modelagem tem severidade $c_n=\mathrm{EAD}_n\times\mathrm{LGD}_n$, em que $\mathrm{EAD}_n$ é a exposição no default e $\mathrm{LGD}_n$ a perda dado o default; a perda da carteira é $L=\sum_n c_n\,\Ind\{D_n\}$, em que $D_n$ é o evento de default da linha $n$ no horizonte de um ano. A Tabela~\ref{tab:carteira} resume a carteira.
""" + _tabela(["Atributo", "Valor"], linhas, "lr",
              "Perfil da carteira de modelagem.", "tab:carteira", H=True))

    if pooled:
        txt_rel = ""
        if rel_agg:
            dif_el = rel_agg["el_diferenca_pct"]
            dif_txt = (r"diferença relativa abaixo de $10^{-9}$, apenas arredondamento numérico"
                       if abs(dif_el) <= 1e-9 else
                       "diferença relativa de " + nb(dif_el, 4, pct=True))
            rg = results.get("regimes_pooled") or {}
            txt_rel = (rf"""

\paragraph{{Nesta execução.}} {nb(rel_agg['linhas_analitica'], 0)} contratos foram reduzidos a {nb(rel_agg['linhas_agregada'], 0)} linhas: {nb(rel_agg['n_grandes'], 0)} grandes exposições e {nb(rel_agg['n_buckets'], 0)} grupos. A perda esperada foi preservada ({dif_txt}), e o maior coeficiente de variação da PD dentro de um grupo foi {nb(rel_agg['pd_cv_max_bucket'], 2)}. Na mediana, o número efetivo de um grupo é {nb(rel_agg.get('n_efetivo_mediano', 0), 1)}, ou {nb(rel_agg.get('razao_n_efetivo_mediana', 1), 1, pct=True)} do seu número de contratos."""
                       + (rf""" Na simulação, {rg.get('nomes', 0)} linhas seguiram o regime de Bernoulli, {rg.get('binomial', 0)} o binomial e {rg.get('granularidade_infinita', 0)} o de granularidade infinita. Este último vale quando $n^{{\mathrm{{eff}}}}>{nm(rg.get('limiar_n_eff', 0), 0)}$, dado $\rho_D={nm(prm.get('rho_alvo'), 1, pct=True)}$."""
                          if rg else ""))
        T.append(r"""
\subsection{Tratamento dos dados: agregação e motor de grupos}
A base analítica, contrato a contrato, foi convertida numa carteira compacta: as maiores exposições passam individualmente, preservando a informação de concentração de nomes, e o restante é agregado em grupos homogêneos (rating $\times$ setor $\times$ categoria IRB). A técnica apoia-se em quatro resultados.

\paragraph{Independência condicional.} Nos modelos de fator, os defaults são independentes dado o fator sistemático $Z$. Num grupo $g$ de $n_g$ contratos com a mesma PD e a mesma severidade, a perda depende apenas de quantos contratos entram em default, e essa contagem é exatamente
\begin{equation}
N_g\mid Z\;\sim\;\mathrm{Binomial}\bigl(n_g,\,p_g(Z)\bigr),
\end{equation}
em que $p_g(Z)$ é a PD do grupo condicional ao fator (a expressão de cada modelo). A agregação reduz a simulação de $M\times N$ para $M\times G$ sorteios sem aproximação, em que $N$ é o número de contratos e $G$ o de grupos. É isso que torna viável a carteira de milhões de linhas (Frey e McNeil, 2003).

\paragraph{Preservação da perda esperada e número efetivo.} Os parâmetros do grupo são ponderados pela severidade,
\begin{equation}
C_g=\sum_{i\in g}c_i,\qquad p_g=\frac{\sum_{i\in g}p_i\,c_i}{C_g},\qquad n_g^{\mathrm{eff}}=\frac{C_g^2}{\sum_{i\in g}c_i^2},
\end{equation}
em que $C_g$ é a severidade total, $p_g$ a PD ponderada e $n_g^{\mathrm{eff}}$ o número efetivo de contratos, um índice de Herfindahl interno ao grupo. Com essa ponderação, $p_g C_g=\sum_i p_i c_i$, e a EL é preservada exatamente. Com severidades desiguais, a variância idiossincrática condicional do grupo é $\sum_i c_i^2\,p_g(Z)(1-p_g(Z))=C_g^2\,p_g(Z)(1-p_g(Z))/n_g^{\mathrm{eff}}$: simular $\mathrm{Binomial}(n_g^{\mathrm{eff}},p_g(Z))$ com severidade $C_g/n_g^{\mathrm{eff}}$ por default preserva a média e a variância condicionais, enquanto usar o número bruto de contratos superestimaria a granularidade. Para a parte idiossincrática, tratar PDs heterogêneas como homogêneas é conservador: a variância de uma soma de Bernoullis heterogêneas nunca excede a da Binomial com a PD média (desigualdade de Hoeffding).

\paragraph{Limite de granularidade infinita e seu critério.} Chama-se de \emph{granularidade infinita} o limite em que o grupo tem tantos contratos, e tão pequenos, que o risco idiossincrático se dilui por completo; na literatura em inglês, fala-se em \emph{large pool} ou carteira \emph{infinitely fine-grained} (Gordy, 2003). Pela lei dos grandes números condicional, a fração em default converge para $p_g(Z)$ e a perda do grupo para $L_g=C_g\,p_g(Z)$, que é o limite de Vasicek (2002) em que se apoiam o ASRF e o IRB (Gordy, 2003). O limite descarta a variância idiossincrática; a razão entre ela e a sistemática é
\begin{equation}
r_g=\frac{\E\bigl[p_g(Z)(1-p_g(Z))\bigr]/n_g^{\mathrm{eff}}}{\mathrm{Var}\bigl[p_g(Z)\bigr]}=\frac{1-\rho_D}{n_g^{\mathrm{eff}}\,\rho_D},
\end{equation}
em que $\rho_D$ é a correlação de default, usada pela identidade $\mathrm{Var}[p_g(Z)]=\rho_D\,p_g(1-p_g)$. O motor aplica o limite apenas quando $r_g<1\%$; os demais grupos são simulados pela Binomial condicional, e as grandes exposições, por Bernoulli. O termo descartado é justamente o que o ajuste de granularidade devolve ao capital, com $C_g^2/n_g^{\mathrm{eff}}$ no lugar de $c_n^2$ (Martin e Wilde, 2002; Gordy e Lütkebohmert, 2013).

\paragraph{Heterogeneidade e o cuidado de Jensen.} A média ponderada preserva a EL, que é linear; as funções de capital são não lineares em PD, e um grupo com PDs muito dispersas introduz viés. O coeficiente de variação de PD dentro de cada grupo é o diagnóstico: valores acima de $0{,}5$ indicam que a formação dos grupos deve ser refinada.""" + txt_rel + "\n")

    # ---------------- 2. parâmetros ----------------
    prows = _parametros(results, prm, pooled, tem, mig, extra)
    T.append(r"""
\section{Parâmetros da execução e suporte bibliográfico}
\label{sec:param}
A Tabela~\ref{tab:param} associa cada parâmetro escolhido nesta execução ao seu papel na modelagem e à referência que fundamenta a técnica. Valores derivados (correlações de ativos, parâmetros da distribuição beta) são produzidos pela calibração e verificados na Seção~\ref{sec:diag}.
""" + _tabela(["Parâmetro", "Valor", "Papel na modelagem", "Ref."],
              [[tx(a), tx(b), tx(c), tx(d)] for a, b, c, d in prows],
              r">{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{6.0cm}l",
              "Parâmetros da execução. As referências numeradas correspondem à lista ao final do documento.",
              "tab:param", H=True))

    # ---------------- 3. metodologia ----------------
    alvo = prm.get("rho_alvo")
    T.append(rf"""
\section{{Fundamentação teórica dos modelos executados}}
Todos os modelos com dependência são calibrados a um alvo comum de correlação de default, $\rho_D={nm(alvo, 3, pct=True)}$, em que $\rho_D$ é a correlação entre os indicadores de default de dois devedores; as caudas tornam-se, assim, diretamente comparáveis, e as diferenças entre os modelos refletem a forma da dependência, não o seu nível (Frey e McNeil, 2003).
""")
    if tem("Binomial independente"):
        T.append(r"""
\subsection{Binomial independente}
Defaults independentes com as PDs individuais da carteira,
\begin{equation}
L=\sum_n c_n\,\Ind\{D_n\},\qquad D_n\sim\mathrm{Bernoulli}(p_n)\ \text{independentes},
\end{equation}
em que $p_n$ é a PD da linha $n$. É o contrafactual sem dependência: pela lei dos grandes números a perda concentra-se em torno da EL, e a distância até os demais modelos mede o efeito da dependência.
""")
    if tem("Beta-binomial") and ab and ab.get("a"):
        T.append(rf"""
\subsection{{Beta-binomial (\emph{{mixture model}})}}
A probabilidade comum de default é sorteada de uma distribuição beta e, dado o sorteio, os defaults são independentes:
\begin{{equation}}
Z\sim\mathrm{{Beta}}(a,b),\qquad D_n\mid Z\sim\mathrm{{Bernoulli}}(Z),\qquad \E[Z]=\frac{{a}}{{a+b}}=\bar p,\qquad \rho_D=\frac{{1}}{{a+b+1}},
\end{{equation}}
em que $a$ e $b$ são os parâmetros de forma, calibrados por momentos: $a={nm(ab['a'])}$ e $b={nm(ab['b'])}$.""" + (r" Na representação agregada, cada grupo segue os três regimes com $p_g(\cdot)=Z$." if pooled else "") + "\n")
    if tem("CreditRisk+ simulação"):
        T.append(rf"""
\subsection{{CreditRisk\textsuperscript{{+}} por simulação: Bernoulli e Poisson}}
O fator sistemático $S\sim\Gamma(v,1/v)$, com $\E[S]=1$ e $\mathrm{{Var}}(S)=1/v$, escala as intensidades de default:
\begin{{equation}}
H_n\mid S\sim\mathrm{{Poisson}}\bigl(p_n(1-w+wS)\bigr),
\end{{equation}}
em que $H_n$ é a contagem de defaults, $w={nm(prm.get('crplus_w'))}$ a carga do fator e $v={nm(prm.get('crplus_v'))}$ a dispersão (Credit Suisse First Boston, 1997). Duas versões são simuladas, como modelos distintos. Na \emph{{Poisson}}, a perda é $L=\sum_n c_n H_n$, como no original: um devedor pode, em tese, inadimplir mais de uma vez, e $\E[L]=\sum_n p_n c_n$. Na \emph{{Bernoulli}}, o default é o evento $H_n\ge1$ e cada devedor perde no máximo uma vez a sua severidade. Para que a PD seja reproduzida, a intensidade $p_n$ é trocada por $\lambda_n$, solução de
\begin{{equation}}
\Pr(H_n\ge1)=1-\E\bigl[e^{{-\lambda_n(1-w+wS)}}\bigr]=1-e^{{-\lambda_n(1-w)}}\Bigl(1+\frac{{\lambda_n w}}{{v}}\Bigr)^{{-v}}=p_n,
\end{{equation}}
e a EL da versão Bernoulli coincide com a analítica.{_crp_cal_txt(results, prm)}"""
                 + (r" Na representação agregada, a versão Poisson agrega exatamente, pois a soma de Poissons independentes é Poisson: $H_g\mid S\sim\mathrm{Poisson}\bigl(n_g p_g(1-w+wS)\bigr)$, com perda $L_g=(C_g/n_g)\,H_g$; na Bernoulli, $D_g\mid S\sim\mathrm{Binomial}\bigl(n_g,1-e^{-\lambda_g(1-w+wS)}\bigr)$." if pooled
                    else "") + "\n")
    cs = results.get("crplus_csfp")
    if tem("CSFP") and cs:
        _met = "pela recursão de Panjer" if cs["metodo"] == "Panjer" else \
            (rf"pela inversão exata da FGP por transformada rápida de Fourier, porque o suporte tem {nb(cs['n_max'], 0)} unidades de $L_0$ e a recursão de Panjer, de custo quadrático no suporte, seria lenta" if cs["log_p0"] > -700 else
             r"pela inversão exata da FGP por transformada rápida de Fourier, porque $\Pr(L=0)=e^{" + nm(cs["log_p0"], 1) + r"}$ está abaixo da precisão de ponto flutuante e a recursão de Panjer não pode partir dela")
        T.append(rf"""
\subsection{{CreditRisk\textsuperscript{{+}} clássico (Credit Suisse First Boston, 1997)}}
A versão original é analítica. Com a mesma intensidade estocástica, $\lambda_n(S)=p_n(w_0+wS)$, $w_0=1-w$ e $S\sim\Gamma(\alpha,1/\alpha)$ com $\alpha=v$ e $\sigma^2=1/\alpha$, a contagem de cada devedor é tratada como Poisson (um devedor pode, em tese, inadimplir mais de uma vez), e as severidades são discretizadas em faixas: com a unidade $L_0$, cada linha recebe $\nu_n=\max\{{1,\mathrm{{round}}(c_n/L_0)\}}$, e a intensidade de cada faixa $j$ é ajustada para preservar a perda esperada,
\begin{{equation}}
\varepsilon_j=\sum_{{n:\,\nu_n=\nu_j}}\frac{{p_n c_n}}{{\nu_j L_0}}.
\end{{equation}}
A função geradora de probabilidade (FGP) da perda, em unidades de $L_0$, tem forma fechada:
\begin{{equation}}
G_L(z)=\exp\Bigl\{{\sum_j w_0\varepsilon_j\bigl(z^{{\nu_j}}-1\bigr)\Bigr\}}\left[\frac{{1-\delta}}{{1-\delta P(z)}}\right]^{{\alpha}},\qquad \delta=\frac{{\sigma^2\mu_1}}{{1+\sigma^2\mu_1}},\quad P(z)=\sum_j\frac{{w\varepsilon_j}}{{\mu_1}}z^{{\nu_j}},
\end{{equation}}
em que $\mu_1=\sum_j w\varepsilon_j$: o setor específico é uma Poisson composta e o sistemático, uma binomial negativa composta. A distribuição sai da recursão de Panjer aplicada a cada fator,
\begin{{equation}}
f(n)=\sum_j\Bigl(a+b\,\frac{{\nu_j}}{{n}}\Bigr)q_j\,f(n-\nu_j),
\end{{equation}}
com $(a,b)=(0,\mu)$ na Poisson composta e $(a,b)=(\delta,(\alpha-1)\delta)$ na binomial negativa, e da convolução das duas. Nesta execução, $L_0=$~R\$~{nb(cs['L0'] / 1e3, 2)}~mil, com {nb(cs['faixas'], 0)} faixas e {nb(cs['defaults_esperados'], 1)} \emph{{defaults}} esperados; a distribuição foi obtida {_met}.""" + _sela_txt(cs)
                 + (r" Na carteira agregada, cada grupo entra com intensidade $n_g p_g$ e perda por \emph{default} $C_g/n_g$, o que é exato para soma de Poissons." if pooled else "") + r"""

\paragraph{As quatro versões.} As quatro versões partem das mesmas hipóteses do CreditRisk\textsuperscript{+}: o evento de \emph{default} não é modelado a partir de uma variável de ativos, a PD é aleatória e depende linearmente do fator, os \emph{defaults} são independentes quando o fator é conhecido e a correlação entre devedores surge apenas do fator comum. O que muda entre elas é a contagem de \emph{defaults}, o tratamento da severidade e o método de cálculo (Tabela~\ref{tab:crplus}). O ponto de sela é reportado como modelo próprio: a aproximação de Lugannani--Rice sobre a FGP com as severidades exatas, sem faixas, com o ES obtido pela integral da cauda aproximada.
""" + _tabela(["Aspecto", "Simulação Bernoulli", "Simulação Poisson", r"Clássico (CSFP)", "Ponto de sela"],
              [["Contagem", r"$\Ind\{H_n\ge1\}$: no máximo um \emph{default} por contrato", "Poisson", "Poisson", "Poisson"],
               ["Severidade", "exata", "exata", r"faixas de $L_0$ (EL preservada)", "exata"],
               ["Cálculo", r"Monte Carlo, $M$ cenários", r"Monte Carlo, $M$ cenários", "FGP; Panjer (ou FFT)", "cumulantes da FGP; Lugannani--Rice"],
               ["Erro", r"amostral ($1/\sqrt{M}$)", r"amostral ($1/\sqrt{M}$)", "discretização; Panjer instável se $\\Pr(L=0)$ é ínfima", "da aproximação, maior perto da média"],
               ["EL", r"analítica ($\lambda_n$ tal que $\Pr(H_n\ge1)=p_n$)", "analítica", "analítica", "analítica"],
               ["Vantagens", "realista para PD alta; flexível", "fiel ao original; flexível", "exato; reprodutível; rápido", "instantâneo; sem faixas; cauda precisa"],
               ["Desvantagens", "ruído; custo; afasta-se do original", "ruído; custo; múltiplos \\emph{defaults} sob fator gama pesado", "faixas distorcem a cauda; correlação só positiva", "aproxima o corpo; ES por integração"]],
              r">{\raggedright\arraybackslash}p{2.25cm}>{\raggedright\arraybackslash}p{2.85cm}>{\raggedright\arraybackslash}p{2.85cm}>{\raggedright\arraybackslash}p{2.85cm}>{\raggedright\arraybackslash}p{2.85cm}",
              r"CreditRisk\textsuperscript{+}: as quatro versões implementadas.", "tab:crplus", H=True))
    if tem("Threshold"):
        T.append(rf"""
\subsection{{Modelos de limiar: Gaussiano e t-Student}}
Na tradição estrutural de Merton (1974), o default ocorre quando a variável latente de ativos cruza o limiar implícito pela PD:
\begin{{equation}}
Y_n=\sqrt{{\rho}}\,G+\sqrt{{1-\rho}}\,\varepsilon_n,\qquad D_n=\Ind\bigl\{{Y_n<\Phi^{{-1}}(p_n)\bigr\}},
\end{{equation}}
em que $G$ é o fator sistemático, $\varepsilon_n$ o choque idiossincrático (ambos normais padrão independentes), $\rho$ a correlação de ativos e $\Phi$ a distribuição normal padrão acumulada. A correlação de ativos $\rho={nm(rho_a)}$ foi calibrada para reproduzir o alvo pela normal bivariada,
\begin{{equation}}
\rho_D=\frac{{\Phi_2(z,z;\rho)-\bar p^{{\,2}}}}{{\bar p(1-\bar p)}},\qquad z=\Phi^{{-1}}(\bar p),
\label{{eq:rhoD}}
\end{{equation}}
em que $\Phi_2(\cdot,\cdot;\rho)$ é a distribuição normal bivariada com correlação $\rho$. A variante t-Student, com $\nu={nm(prm.get('nu_t', 10), 0)}$ graus de liberdade, multiplica as variáveis latentes por um choque radial comum $\sqrt{{\nu/V}}$, $V\sim\chi^2_\nu$, e introduz dependência de cauda. O contraste entre as duas variantes funciona como teste de sensibilidade à premissa de cauda.
""")
    if tem("multifatorial"):
        mfc = (mf or {}).get("calibracao", {})
        T.append(rf"""
\subsection{{Extensão multifatorial setorial}}
Com um fator global e um fator por setor,
\begin{{equation}}
Y_n=\sqrt{{\rho_G}}\,G+\sqrt{{\rho_S-\rho_G}}\,F_{{s(n)}}+\sqrt{{1-\rho_S}}\,\varepsilon_n,
\end{{equation}}
em que $F_{{s(n)}}$ é o fator do setor do devedor $n$, $\rho_S$ a correlação de ativos intra-setor e $\rho_G$ a entre setores. Os alvos de correlação de default, {nb(prm.get('rho_intra'), 3, pct=True)} e {nb(prm.get('rho_inter'), 3, pct=True)}, resultaram em $\rho_S={nm(mfc.get('rho_ativos_intra'))}$ e $\rho_G={nm(mfc.get('rho_ativos_inter'))}$. As contribuições setoriais seguem a alocação de Euler: a contribuição ao ES é a perda esperada da posição condicional à cauda (Tasche, 2008).
""")
        if mfc.get("cargas_por_setor"):
            ps = mfc["por_setor"]
            T.append(r"""
\paragraph{Cargas por setor.} Nesta execução, cada setor $s$ recebeu os próprios alvos de correlação de default, e as correlações de ativos foram calibradas setor a setor, na PD média do setor:
\begin{equation}
Y_n=\sqrt{\rho_{G,s}}\,G+\sqrt{\rho_{S,s}-\rho_{G,s}}\,F_{s}+\sqrt{1-\rho_{S,s}}\,\varepsilon_n,\qquad s=s(n).
\end{equation}
A correlação de ativos dentro do setor $s$ é $\rho_{S,s}$; entre os setores $s$ e $t$, $\sqrt{\rho_{G,s}\,\rho_{G,t}}$. A estrutura entre setores é de posto 1: os pares diferem pela carga de cada setor no fator global, mas a razão entre as correlações de um setor com outros dois é a mesma para todo setor; blocos de setores exigiriam um segundo fator global. Os valores de $\rho_S$ e $\rho_G$ acima são médias ponderadas pela exposição; a Tabela~\ref{tab:mf-setor} traz a calibração de cada setor."""
                     + _tabela(
                         ["Setor", "PD média", r"$\rho_D$ intra", r"$\rho_D$ inter",
                          r"$\rho_{S,s}$", r"$\rho_{G,s}$"],
                         [[tx(r["setor"]), nb(r["pd_media"], 3, pct=True),
                           nb(r["rho_default_intra_alvo"], 3, pct=True),
                           nb(r["rho_default_inter_alvo"], 3, pct=True),
                           nb(r["rho_ativos_intra"]), nb(r["rho_ativos_inter"])]
                          for _, r in ps.iterrows()],
                         "lrrrrr",
                         "Calibração das cargas por setor (alvos de correlação "
                         "de default e correlações de ativos).",
                         "tab:mf-setor", H=True))
            from .multifactor import leitura_setores, calibrate_rho
            _pb = mfc.get("pbar", pbar)
            _ri, _re = prm.get("rho_intra", 0.08), prm.get("rho_inter", 0.03)
            lps, lpares, lres = leitura_setores(
                mfc, _ri, _re, calibrate_rho(_pb, _ri), calibrate_rho(_pb, _re))
            (v1, a1, b1), (vz, az, bz) = lpares[0], lpares[-1]
            T.append(r"""
\paragraph{Leitura das escolhas.} A fração da correlação de ativos intra-setor que vem do fator global, $\rho_{G,s}/\rho_{S,s}$, indica a origem da dependência: perto de 1, o setor é \emph{cíclico}: os defaults andam juntos porque a economia os move juntos, e o setor contagia os demais. Perto de 0, é \emph{concentrado}: um choque próprio derruba muitos nomes de uma vez, mas pouco se espalha. Nesta execução:
\begin{itemize}\setlength\itemsep{1pt}
""" + "".join(rf"\item \textbf{{{tx(r['setor'])}}} (fração global {nb(r['fracao_global'])}, {tx(r['perfil'])}): {tx(r['leitura'])}." + "\n"
              for _, r in lps.iterrows()) + rf"""\end{{itemize}}
O par de setores mais ligado é {tx(a1)}--{tx(b1)}, com correlação de default de {nb(v1, 3, pct=True)}; o menos ligado, {tx(az)}--{tx(bz)}, com {nb(vz, 3, pct=True)}. As correlações de ativos médias, ponderadas pela exposição, ficaram em {nm(lres['rho_s_medio'])} (intra) e {nm(lres['rho_g_medio'])} (inter), ante {nm(lres['rho_s_ref'])} e {nm(lres['rho_g_ref'])} na calibração uniforme, ou seja, {tx({'acima': 'acima do padrão', 'abaixo': 'abaixo do padrão'}.get(lres['direcao'], 'em linha com o padrão'))}. Os alvos por setor são hipóteses do usuário; esta seção verifica a sua calibração, não a sua fundamentação empírica.
""")
    if tem("ASRF"):
        T.append(r"""
\subsection{ASRF: carteira assintoticamente granular}
O limite analítico de Vasicek (2002), para uma carteira de granularidade infinita (número de exposições tendendo ao infinito, cada uma com peso desprezível) e um único fator sistemático,
\begin{equation}
\VaR_\alpha=C\,\Phi\!\left(\frac{\Phi^{-1}(\bar p)+\sqrt{\rho}\,\Phi^{-1}(\alpha)}{\sqrt{1-\rho}}\right),
\end{equation}
em que $C=\sum_n c_n$ é a exposição líquida e $\alpha$ o nível de confiança, fundamenta a função de capital do IRB (Gordy, 2003).
""")
    if irb:
        seg = irb.get("por_segmento")
        cats = ", ".join(seg["descricao"]) if seg is not None and len(seg) else "atacado (demais)"
        T.append(rf"""
\subsection{{Capital regulatório: abordagem IRB (modelos internos, Resolução BCB 303/2023)}}
O requerimento pela abordagem IRB do Banco Central do Brasil (Resolução BCB 303/2023), derivada do modelo ASRF:
\begin{{equation}}
K_n=\left[\Phi\!\left(\frac{{\Phi^{{-1}}(p_n)+\sqrt{{R_n}}\,\Phi^{{-1}}(0{{,}}999)}}{{\sqrt{{1-R_n}}}}\right)-p_n\right]\mathrm{{MA}}_n,\qquad \mathrm{{Capital}}=\sum_n c_n K_n,
\end{{equation}}
em que $K_n$ é o coeficiente de capital (fração da severidade), $R_n$ a correlação regulatória da categoria e $\mathrm{{MA}}_n$ o ajuste de maturidade. No atacado, $R=0{{,}}12f+0{{,}}24(1-f)$ com $f=(1-e^{{-50p}})/(1-e^{{-50}})$, reduzida em até $0{{,}}04$ pelo porte nas SME e elevada a $0{{,}}30$ no limite superior nas HVCRE; nas instituições financeiras S1/S2, multiplicada por $1{{,}}25$; no varejo, $0{{,}}15$ (residencial), $0{{,}}04$ (rotativo qualificado) e $0{{,}}03g+0{{,}}16(1-g)$ com $g=(1-e^{{-35p}})/(1-e^{{-35}})$ (demais). O ajuste de maturidade,
\begin{{equation}}
\mathrm{{MA}}=\frac{{1+(M-2{{,}}5)\,b}}{{1-1{{,}}5\,b}},\qquad b=\bigl(0{{,}}11852-0{{,}}05478\ln p\bigr)^2,
\end{{equation}}
em que $M\in[1,5]$ é o prazo efetivo, aplica-se só ao atacado e às instituições financeiras; no varejo, $\mathrm{{MA}}=1$. As PDs têm piso de $0{{,}}05\%$ ($0{{,}}10\%$ no rotativo). Categorias presentes nesta carteira: {tx(cats)}."""
                 + ("" if irb.get("segmentado") else r" A carteira não informa a coluna \texttt{segmento\_irb}: todas as exposições foram tratadas como atacado.")
                 + r""" O ajuste de granularidade segue Gordy e Lütkebohmert (2013), na formulação baseada no CreditRisk\textsuperscript{+} implementada por Bolder (2018)"""
                 + (r", com o termo $c^2$ substituído por $C_g^2/n_g^{\mathrm{eff}}$ em cada grupo." if pooled else ".") + "\n")
    cp = results.get("padronizado")
    if cp:
        T.append(r"""
\subsection{Capital regulatório: abordagem padronizada (Resolução BCB 229/2022) e piso de saída}
Na abordagem padronizada, o ativo ponderado pelo risco é a soma das exposições ponderadas pelo fator de ponderação de risco (FPR) da sua categoria,
\begin{equation}
\mathrm{RWA}_{\mathrm{CPAD}}=\sum_n \mathrm{EAD}_n\,\mathrm{FPR}_n ,
\end{equation}
sem uso de PD ou LGD. Os fatores aplicados são os da Resolução BCB 229/2022: varejo, 75\% (art.~46: pessoa natural ou empresa de pequeno porte, com exposição agregada por contraparte de até R\$~5~milhões e inferior a 0{,}2\% do total de varejo), 45\% para o rotativo sem atraso, parcelamento ou saque em 360 dias (art.~47) e 100\% para a pessoa natural fora desses critérios (art.~48); pessoa jurídica, 65\% para a grande empresa de baixo risco (art.~35), 85\% para a PME com ativo inferior a R\$~240~milhões e receita inferior a R\$~300~milhões (art.~36) e 100\% nos demais casos (art.~41); imobiliário residencial por faixa de LTV, de 20\% a 70\% sem dependência do fluxo do imóvel (art.~50) e de 30\% a 105\% com dependência (art.~51); imóvel não residencial com dependência do fluxo, de 70\% a 110\% (art.~53), e demais exposições imobiliárias, 150\% (art.~54); instituições financeiras pela categoria A, B ou C e pelo prazo original (art.~33). O ativo problemático recebe 150\%, 100\% ou 50\% conforme a provisão seja inferior a 20\%, entre 20\% e 50\% ou superior a 50\% do saldo (art.~66), prevalecendo o maior entre esse fator e o específico (art.~22, II). """
            + (r"Na base analítica, o fator é atribuído contrato a contrato, antes da agregação; como o RWA é linear na exposição, a soma por grupo é exata. " if pooled else "")
            + r"""A Resolução BCB 303/2023 (art.~34) limita o benefício dos modelos internos pelo piso de saída:
\begin{equation}
\mathrm{RWA}=\max\bigl\{\mathrm{RWA}_{\mathrm{IRB}};\;0{,}725\cdot\mathrm{RWA}_{\mathrm{CPAD}}\bigr\},
\end{equation}
calculados sobre as mesmas exposições. O requerimento de capital é $F=8\%$ do RWA.
""")
    ad = results.get("inadimplidos")
    if ad:
        r_ = ad["resumo"]
        T.append(r"""
\subsection{Exposições em default na carteira ativa (sem \emph{write-off})}
\label{sec:teoria-default}
Contratos com 90 dias ou mais de atraso, marcados como ativo problemático ou com tempo em \emph{default} informado saem dos modelos de simulação: o evento de \emph{default} já ocorreu, e o risco remanescente é o da recuperação. Seja $t$ o tempo em \emph{default}, em meses. A melhor estimativa da perda, $\mathrm{ELBE}(t)$, é a curva de LGD em \emph{default}, não decrescente em $t$ e igual a 100\% no prazo de baixa $T_b$; """
                 + (r"nesta execução, usou-se a proxy linear a partir da LGD não \emph{default} do contrato, $\mathrm{LGD}_0$,"
                    if r_["curva"] != "informada" else r"nesta execução, a curva foi informada por categoria IRB, com interpolação linear entre os pontos; para categorias sem curva, usou-se a proxy linear")
                 + r"""
\begin{equation}
\mathrm{ELBE}(t)=\mathrm{LGD}_0+(1-\mathrm{LGD}_0)\min\{t/T_b,\,1\}.
\end{equation}
A provisão de estágio 3 é $\mathrm{EAD}\cdot\mathrm{ELBE}(t)$. O capital cobre a incerteza da recuperação: pela abordagem IRB para exposições inadimplidas, $K=\max\{0;\,\mathrm{LGD}_D-\mathrm{ELBE}\}$, em que a LGD em \emph{default} $\mathrm{LGD}_D$ incorpora uma margem sobre a ELBE. Adota-se a proxy
\begin{equation}
\mathrm{LGD}_D(t)=\mathrm{ELBE}(t)+k\,\bigl[1-\mathrm{ELBE}(t)\bigr]\;\Rightarrow\;K(t)=k\,\bigl[1-\mathrm{ELBE}(t)\bigr],
\end{equation}
com capital $\mathrm{EAD}\cdot K$ e RWA $=12{,}5\cdot\mathrm{EAD}\cdot K$: máximo logo após o \emph{default} e nulo na baixa. Com o tempo, a absorção migra do capital para a provisão. Em um ano, a perda é o incremento determinístico da provisão,
\begin{equation}
\Delta L=\sum_{n\in\mathcal{D}}\mathrm{EAD}_n\bigl[\mathrm{ELBE}_n(t_n+12)-\mathrm{ELBE}_n(t_n)\bigr],
\end{equation}
em que $\mathcal{D}$ é o conjunto das exposições em \emph{default}; por ser constante, $\Delta L$ soma-se ao EL, ao VaR e ao ES de qualquer modelo sem alterar a perda inesperada. No padronizado, essas exposições recebem o fator do ativo problemático (art.~66 da Resolução BCB 229/2022), e no piso de saída entram dos dois lados. Contratos em atraso abaixo de 90 dias seguem na carteira não \emph{default}, com a LGD não \emph{default}; acima de 30 dias, passam ao estágio 2 do ECL.

\paragraph{Consolidação da carteira total.} As medidas dos modelos de simulação referem-se à carteira não \emph{default} (ND). A carteira total soma a parte em \emph{default} (D), que no horizonte de um ano contribui com o incremento da provisão $\Delta L$ na perda esperada e com $\Delta L+K_D$ no quantil, em que $K_D=\sum_{n\in\mathcal D}\mathrm{EAD}_n(\mathrm{LGD}_{D,n}-\mathrm{ELBE}_n)$ é a perda adicional de recuperação no cenário adverso:
\begin{equation}
\mathrm{EL}_{\mathrm{total}}=\mathrm{EL}_{ND}+\Delta L,\qquad
\mathrm{VaR}^{\alpha}_{\mathrm{total}}=\mathrm{VaR}^{\alpha}_{ND}+\Delta L+K_D,
\end{equation}
e o mesmo para o ES. O capital é a perda inesperada da carteira total, na qual $\Delta L$ se cancela:
\begin{equation}
\mathrm{Capital}^{\alpha}=\mathrm{VaR}^{\alpha}_{\mathrm{total}}-\mathrm{EL}_{\mathrm{total}}=\bigl(\mathrm{VaR}^{\alpha}_{ND}-\mathrm{EL}_{ND}\bigr)+K_D.
\end{equation}
O estoque já provisionado, $\sum_{n\in\mathcal D}\mathrm{EAD}_n\,\mathrm{ELBE}_n$, é perda reconhecida em exercícios anteriores e não entra na perda do ano; contá-lo no VaR e na EL não mudaria o capital, porque se cancelaria na diferença. O que consome capital é a parte da perda esperada \emph{não} provisionada: o capital alocado soma ao capital pelo ES a insuficiência de provisão, $\max\{0;\,\mathrm{EAD}\cdot\mathrm{ELBE}-\text{provisão constituída}\}$, que na regra de Basileia é deduzida do capital principal. A soma simples é exata porque a perda do \emph{default} é determinística: não há diversificação entre as duas partes a capturar.
""")
    if mig is not None:
        T.append(r"""
\subsection{Migração de ratings, ECL e capital multiperíodo}
A matriz de transição anual $P$ fornece a estrutura a termo de PD pelas potências $P^t$; o ECL desconta as perdas marginais em 12 meses e no prazo de cada contrato (\emph{lifetime}), com estágios conforme o IFRS 9 e a Resolução CMN 4.966: estágio 1 (ECL de 12 meses), estágio 2 (ECL \emph{lifetime}, com PD da matriz e LGD não \emph{default}) para atraso acima de 30 dias ou queda de rating desde a origem, e estágio 3 para as exposições em \emph{default} (provisão $\mathrm{EAD}\cdot\mathrm{ELBE}(t)$). Como o ECL é linear na exposição, é calculado contrato a contrato, antes da agregação. O capital multiperíodo evolui a carteira ano a ano pela matriz condicional ao fator sistemático, acumulando perdas descontadas; a construção generaliza o deslocamento de crédito de um parâmetro de Belkin, Suchower e Forest (1998)."""
            + (r" Na carteira agregada, cada grupo evolui como vetor de frações por rating; nos grupos fora do limite de granularidade infinita, os \emph{defaults} de cada ano são sorteados, $D_t\mid Z_t\sim\mathrm{Binomial}(A_{t-1},q_t)$, sobre o número inteiro de contratos equivalentes vivos $A_t$ (com $A_0=n^{\mathrm{eq}}$) e a taxa de \emph{default} condicional entre os vivos $q_t$, de modo que o risco idiossincrático intragrupo entra no capital multiperíodo; as migrações entre ratings não \emph{default} seguem as frações condicionais."
               if pooled else "") + "\n")

    # ---------------- 4. resultados ----------------
    cols = [c for c in ["modelo", "EL", "UL", "VaR 99.00%", "ES 99.00%",
                        "VaR 99.90%", "ES 99.90%"] if c in med.columns]
    lin = [[tx(_rotulo(r[0]))] + [mi(v) if v == v else "---" for v in r[1:]]
           for r in med[cols].itertuples(index=False)]
    cab = ["Modelo (R\\$ milhões)", "EL", "UL", r"VaR$_{99\%}$", r"ES$_{99\%}$",
           r"VaR$_{99,9\%}$", r"ES$_{99,9\%}$"][:len(cols)]
    base_g = med[med["modelo"].str.startswith("Threshold Gaussiano 1F")]
    vg = float(base_g.iloc[0][col999]) if len(base_g) else None
    col_es = col999.replace("VaR", "ES")
    lin2 = []
    for _, r in med.iterrows():
        v9 = r.get(col999)
        if v9 is None or v9 != v9:
            continue
        e9 = r.get(col_es)
        lin2.append([tx(_rotulo(r["modelo"])), nb(v9 / total, 3, pct=True),
                     nb(v9 / r["EL"]) if r["EL"] else "---",
                     nb(e9 / v9) if e9 == e9 and v9 else "---",
                     nb(v9 / vg) if vg else "---"])
    _fig_hierarquia(med, total, col999, os.path.join(figdir, "hierarquia.pdf"))
    figs.append("hierarquia.pdf")
    T.append(r"""
\section{Resultados por modelo}
A Tabela~\ref{tab:medidas} reporta as medidas de risco de cada modelo, em R\$ milhões, com duas casas decimais (a base de entrada está em reais); a Tabela~\ref{tab:relativos} normaliza as caudas, e a Figura~\ref{fig:hier} ordena os modelos pelo VaR 99{,}90\%. A hierarquia reflete a estrutura de dependência de cada modelo sob a calibração comum.
""" + _tabela(cab, lin, "l" + "r" * (len(cols) - 1),
              "Medidas de risco por modelo (R\\$ milhões).", "tab:medidas")
             + _tabela(["Modelo", r"VaR/expos.", r"VaR/EL", r"ES/VaR", r"$\times$ Gauss 1F"],
                       lin2, "lrrrr",
                       "Indicadores relativos no quantil 99{,}90\\%.", "tab:relativos")
             + r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{hierarquia.pdf}
\caption{VaR 99{,}90\% por modelo, em percentual da exposição líquida.}
\label{fig:hier}
\end{figure}
""")
    dists = results.get("distribuicoes") or {}
    if dists:
        _fig_distribuicoes({k: v / MI for k, v in dists.items()}, total / MI, os.path.join(figdir, "distribuicoes.pdf"))
        figs.append("distribuicoes.pdf")
        T.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{distribuicoes.pdf}
\caption{Distribuições de perda. Esquerda: corpo até o quantil 99{,}9\%, na escala dos modelos com dependência; a curva tracejada do modelo independente, quase degenerada em torno da EL, pode ultrapassar o topo do gráfico. Direita: cauda $\Pr(L>x)$ em escala logarítmica, com a linha pontilhada no nível $10^{-3}$.}
\label{fig:dist}
\end{figure}
""")

    # ---------------- 5. diagnósticos ----------------
    drows = _diagnosticos(results, prm, pooled, mig, rel_agg)
    cor = {"OK": r"\textcolor{okgreen}{\textbf{OK}}",
           "Atenção": r"\textcolor{alertred}{\textbf{Atenção}}",
           "Esperado": r"\textcolor{linkblue}{\textbf{Esperado}}"}
    sint, itens = _interpretar(drows, prm.get("M"))
    T.append(rf"""
\section{{Diagnósticos de validação}}
\label{{sec:diag}}
Os testes da Tabela~\ref{{tab:diag}} verificam a consistência interna da execução. A EL simulada deve reproduzir a analítica dentro do erro de Monte Carlo,
\begin{{equation}}
\bigl|\widehat{{\E}}[L]-\E[L]\bigr|\le 3\,\widehat\sigma_L/\sqrt{{M}},
\end{{equation}}
em que $\widehat{{\E}}[L]$ é a média das $M$ perdas simuladas e $\widehat\sigma_L$ o seu desvio-padrão (Glasserman, 2004); as calibrações são verificadas recalculando $\rho_D$ a partir dos parâmetros calibrados por \eqref{{eq:rhoD}}; a variante t-Student deve ter cauda maior ou igual à gaussiana; e a precisão do quantil é medida pelo intervalo de confiança de 95\% por estatísticas de ordem,
\begin{{equation}}
\bigl[L_{{(k_-)}},\,L_{{(k_+)}}\bigr],\qquad k_\pm=\alpha M\pm1{{,}}96\sqrt{{M\alpha(1-\alpha)}},
\end{{equation}}
em que $L_{{(k)}}$ é a $k$-ésima menor perda simulada (David e Nagaraja, 2003). Cada teste recebe um de três status:
\begin{{description}}\setlength\itemsep{{1pt}}
""" + "".join(rf"\item[{cor[k]}] {tx(v)}" + "\n" for k, v in LEGENDA_STATUS) + r"""\end{description}
""" + _tabela_longa(["Teste", "Resultado", "Critério", "Status"],
              [[tx(a), tx(b), tx(c), cor.get(d, tx(d))] for a, b, c, d in drows],
              r">{\raggedright\arraybackslash}p{5.9cm}>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{4.4cm}c",
              "Diagnósticos de validação da execução.", "tab:diag")
             + r"""
\subsection*{Leitura dos resultados}
""" + tx(sint) + ("\n\\begin{itemize}\\setlength\\itemsep{2pt}\n" + "".join(
                rf"\item \textbf{{{tx(teste)}}} ({cor[st_]}): {tx(leitura)} \emph{{Ação:}} {tx(acao)}" + "\n"
                for st_, teste, leitura, acao in itens) + "\\end{itemize}\n" if itens else "\n"))

    # ---------------- 6. capital ----------------
    if irb:
        lin_k = [["Capital IRB ($K^*$, 99{,}9\\%)", f"{mi(irb.get('capital_total_999'))} ({nb(irb.get('capital_pct_exposicao'), 3, pct=True)} da exposição líquida)"],
                 ["Ajuste de granularidade (GA)", mi(irb.get("ajuste_granularidade"))],
                 ["Capital com GA", mi(irb.get("capital_com_ga"))],
                 ["RWA equivalente ($12{,}5\\times$)", mi(irb.get("rwa_equivalente"))],
                 ["$R$ regulatório ponderado", nb(irb.get("rho_basel_ponderado"), 3, pct=True)]]
        cp = results.get("padronizado")
        T.append(r"""
\section{Capital regulatório e concentração}
\label{sec:capital}
""" + _tabela(["Indicador", "Valor"], lin_k, "lr",
              "Capital regulatório pela abordagem IRB (modelos internos, Resolução BCB 303/2023); valores em R\\$ milhões.", "tab:irb", H=True))
        if cp:
            ps_ = cp["piso"]
            lin_p = [["RWA IRB (modelos internos)", mi(ps_["rwa_irb"])],
                     [r"RWA padronizado (RWA\textsubscript{CPAD})", mi(ps_["rwa_cpad"])],
                     ["Razão IRB / padronizado", nb(ps_["razao_irb_cpad"])],
                     [r"Piso: $0{,}725\times$RWA\textsubscript{CPAD}", mi(ps_["piso"])],
                     ["RWA final (art.~34, Res. BCB 303)", mi(ps_["rwa_final"])],
                     [r"Capital final ($8\%$ do RWA)", mi(ps_["capital_final"])],
                     ["Piso vinculante", "sim" if ps_["vinculante"] else "não"]]
            T.append(rf"""
Pela abordagem padronizada, o RWA da carteira é R\$~{mi(ps_['rwa_cpad'])}~milhões, com FPR médio de {nb(cp['fpr_medio'], 3, pct=True)}. O RWA dos modelos internos corresponde a {nb(ps_['razao_irb_cpad'] * 100, 3)}\% do padronizado; """
                     + (r"como a razão é inferior a 72{,}5\%, \textbf{o piso de saída é vinculante} e o capital passa a ser determinado pela abordagem padronizada." if ps_["vinculante"]
                        else r"como a razão supera 72{,}5\%, o piso de saída não é vinculante e prevalece o RWA dos modelos internos.")
                     + "\n" + _tabela(["Indicador", "Valor"], lin_p, "lr",
                                      "Abordagem padronizada e piso de saída (R\\$ milhões).", "tab:piso", H=True))
            pa = cp.get("por_artigo")
            if pa is not None and len(pa):
                la = [[tx(r.descricao), nb(r.linhas, 0), mi(r.ead), mi(r.rwa),
                       nb(r.fpr_medio, 3, pct=True)]
                      for r in pa.itertuples(index=False)]
                T.append(_tabela(["Fator (artigo)", "Linhas", "EAD (R\\$ mi)", "RWA (R\\$ mi)", "FPR médio"],
                                 la, r">{\raggedright\arraybackslash}p{6.4cm}rrrr",
                                 r"RWA\textsubscript{CPAD} por fator de ponderação da Resolução BCB 229/2022 (R\$ milhões).",
                                 "tab:cpad", H=True))
            cmp_ = cp.get("comparacao")
            if cmp_ is not None and len(cmp_) and "descricao" in cmp_.columns:
                lc = [[tx(r.descricao), mi(r.rwa_irb), mi(r.rwa_cpad), nb(r.razao_irb_cpad)]
                      for r in cmp_.itertuples(index=False)]
                T.append(_tabela(["Categoria", "RWA IRB", "RWA padronizado", "IRB / padr."],
                                 lc, "lrrr",
                                 "Comparação entre as abordagens por categoria (RWA em R\\$ milhões).",
                                 "tab:irbcpad", H=True))
        ad = results.get("inadimplidos")
        if ad:
            r_ = ad["resumo"]
            _fig_default(ad, os.path.join(figdir, "default.pdf"))
            figs.append("default.pdf")
            ld = [["Exposições em \\emph{default}", nb(r_["contratos"], 0)],
                  ["EAD (R\\$ milhões)", mi(r_["ead"])],
                  ["Tempo médio em \\emph{default} (meses)", nb(r_["meses_medio"], 1)],
                  ["ELBE média / LGD$_D$ média",
                   f"{nb(r_['elbe_media'], 3, pct=True)} / {nb(r_['lgd_d_media'], 3, pct=True)}"],
                  ["Provisão requerida, EAD$\\cdot$ELBE (R\\$ milhões)", mi(r_["provisao_elbe"])]]
            if r_["provisao_informada"] == r_["provisao_informada"]:
                ld += [["Provisão informada (R\\$ milhões)", mi(r_["provisao_informada"])],
                       ["Insuficiência frente à ELBE (R\\$ milhões)", mi(r_["insuficiencia_provisao"])]]
            ld += [["Capital IRB, EAD$\\cdot$(LGD$_D-$ELBE) (R\\$ milhões)", mi(r_["capital_irb"])],
                   ["RWA IRB / RWA\\textsubscript{CPAD} (R\\$ milhões)",
                    f"{mi(r_['rwa_irb'])} / {mi(r_['rwa_cpad'])}"],
                   ["Perda incremental em 12 meses (R\\$ milhões)", mi(r_["perda_incremental_12m"])],
                   ["Contratos além do prazo de baixa", nb(r_["ja_baixados"], 0)]]
            pf = ad["por_faixa"]
            lf = [[tx(str(r.meses)), nb(r.contratos, 0), mi(r.ead), nb(r.elbe, 3, pct=True),
                   nb(r.K, 3, pct=True), mi(r.provisao_elbe), mi(r.capital_irb)]
                  for r in pf.itertuples(index=False)]
            T.append(rf"""
\paragraph{{Exposições em \emph{{default}}.}} A Tabela~\ref{{tab:default}} resume o tratamento das exposições em \emph{{default}} e a Tabela~\ref{{tab:defaultfaixa}} mostra, por tempo em \emph{{default}}, a passagem do capital para a provisão; a Figura~\ref{{fig:default}} traz a curva usada. A perda incremental de R\$~{mi(r_['perda_incremental_12m'])}~milhões em 12 meses soma-se às medidas da Tabela~\ref{{tab:medidas}}, que se referem à carteira não \emph{{default}}; a consolidação da carteira total, com o capital alocado, está na Tabela~\ref{{tab:consolidacao}}.
""" + _tabela(["Indicador", "Valor"], ld, "lr",
              "Exposições em \\emph{default} (sem \\emph{write-off}).", "tab:default", H=True)
              + _tabela(["Meses em \\emph{default}", "Contratos", "EAD", "ELBE", "$K$",
                         "Provisão", "Capital"], lf, "lrrrrrr",
                        "Exposições em \\emph{default} por tempo em \\emph{default} (valores em R\\$ milhões).",
                        "tab:defaultfaixa", H=True)
              + r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.72\textwidth]{default.pdf}
\caption{Curva de LGD em \emph{default}: ELBE (provisão), LGD$_D$ e, entre as duas, o coeficiente de capital $K$.}
\label{fig:default}
\end{figure}
""")
        seg = irb.get("por_segmento")
        if seg is not None and len(seg):
            ls = [[tx(r.descricao), nb(r.linhas, 0), mi(r.exposicao_liquida),
                   nb(r.R_ponderado, 3, pct=True), mi(r.capital),
                   nb(r.capital_pct, 3, pct=True)]
                  for r in seg.itertuples(index=False)]
            T.append(_tabela(["Categoria", "Linhas", "Expos. líquida (R\\$ mi)", "$R$",
                              "Capital (R\\$ mi)", "Capital/expos."], ls, "lrrrrr",
                             "Capital IRB por categoria da Resolução BCB 303/2023; $R$ é a correlação média ponderada pela severidade.",
                             "tab:irbseg"))
        if mf and mf.get("contrib_por_setor") is not None:
            cps = mf["contrib_por_setor"]
            lst = [[tx(r.setor), nb(r.exposicao_pct, 3, pct=True),
                    nb(r.contrib_ES_pct, 3, pct=True),
                    nb(r.razao_risco_exposicao)]
                   for r in cps.itertuples(index=False)]
            T.append(rf"""
A Tabela~\ref{{tab:setores}} decompõe a cauda por setor pela alocação de Euler; razão acima de 1 indica setor que contribui ao risco de cauda mais do que à exposição. HHI setorial: {nb(mf.get('hhi_setorial'), 5)}.
""" + _tabela(["Setor", "Exposição", "Contrib. ES", "Razão risco/expos."],
              lst, "lrrr", "Decomposição setorial (contribuições de Euler ao ES).",
              "tab:setores"))

    # ---------------- 7. migração ----------------
    if mig is not None:
        ecl_df, sim_m = mig.get("ecl"), mig.get("sim")
        e12 = float(ecl_df["ECL_12m"].sum())
        eli = float(ecl_df["ECL_lifetime"].sum())
        lm = [["ECL 12 meses (R\\$ milhões)", f"{mi(e12)} ({nb(e12 / total, 3, pct=True)} da exposição líquida)"],
              ["ECL \\emph{lifetime} (R\\$ milhões)", f"{mi(eli)} ({nb(eli / e12 if e12 else 0)}$\\times$ o de 12 meses)"]]
        if "provisao" in ecl_df.columns:
            lm.append(["Provisão por estágios (R\\$ milhões)", mi(float(ecl_df["provisao"].sum()))])
            if "estagio" in ecl_df.columns:
                _nm = {1: "12 meses", 2: "\\emph{lifetime}", 3: "EAD$\\cdot$ELBE"}
                for e_, g_ in ecl_df.groupby("estagio"):
                    lm.append([f"\\quad Estágio {int(e_)} ({_nm.get(int(e_), '')}): {nb(len(g_), 0)} contratos",
                               f"{mi(float(g_['provisao'].sum()))} ({nb(float(g_['provisao'].sum()) / float(g_['exposicao'].sum()), 3, pct=True)} da EAD)"])
        if sim_m and sim_m.get("T"):
            Tn = int(sim_m["T"])
            lm += [[f"VaR 99{{,}}90\\% em {Tn} anos", mi(sim_m["VaR"].get("99.90%"))],
                   [f"ES 99{{,}}90\\% em {Tn} anos", mi(sim_m["ES"].get("99.90%"))],
                   ["EL multiperíodo descontada (R\\$ milhões)", mi(sim_m["EL"])]]
        T.append(r"""
\section{Migração, ECL e horizonte multiperíodo}
""" + _tabela(["Indicador", "Valor"], lm, "lr",
              "Perda esperada de crédito e capital multiperíodo.", "tab:mig", H=True))
        if sim_m and len(np.atleast_1d(sim_m.get("perdas", []))) > 10:
            _fig_multiperiodo({**sim_m, "perdas": sim_m["perdas"] / MI, "VaR": {k: v / MI for k, v in sim_m["VaR"].items()}, "ES": {k: v / MI for k, v in sim_m["ES"].items()}, "EL": sim_m["EL"] / MI}, os.path.join(figdir, "multiperiodo.pdf"))
            figs.append("multiperiodo.pdf")
            T.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{multiperiodo.pdf}
\caption{Distribuição das perdas descontadas acumuladas no horizonte multiperíodo, com o VaR 99{,}9\%.}
\end{figure}
""")

    _te, _fe = _capitulo_estresse(results, figdir)
    if _te:
        T.append(_te)
        figs += _fe
    _tc2, _fc2 = _capitulo_customizado(results, figdir)
    if _tc2:
        T.append(_tc2)
        figs += _fc2

    # ---------------- 8. limitações e referências ----------------
    T.append(r"""
\section{Limitações e condições de uso}
\begin{itemize}\setlength\itemsep{1pt}
\item A LGD é determinística: a severidade é tratada como parâmetro fixo por contrato.
\item As correlações são calibradas a alvos informados pelo usuário, não estimadas da série histórica da carteira; a sensibilidade a esses alvos deve ser testada.
\item Nos grupos homogêneos, a PD é atribuída uniformemente; o coeficiente de variação de PD intra-grupo é o diagnóstico de adequação (desigualdade de Jensen).
\item A matriz de transição é homogênea no tempo; horizontes acima de cinco anos devem ser lidos como cenário exploratório.
\item O Monte Carlo é simples: o erro do quantil 99{,}9\% decai com $1/\sqrt{M}$, e o diagnóstico de intervalo de confiança indica quando $M$ é insuficiente.
\item No IRB, não são calculados os pisos de LGD da abordagem avançada. As exposições em \emph{default} recebem tratamento determinístico: ELBE pela curva de LGD em \emph{default} e LGD$_D$ por proxy (margem $k$), sem simular a incerteza da recuperação. No padronizado, a coluna \texttt{exposicao} é tomada como a EAD final (fatores de conversão de exposições fora do balanço já aplicados), e não são tratados o descasamento cambial (art.~55), a mitigação de risco de crédito, os financiamentos especializados (arts.~37 a 40), participações e cotas de fundos; na ausência de LTV, receita ou categoria da instituição financeira, aplica-se o fator conservador indicado no aplicativo.
\item Este documento é gerado automaticamente a partir dos parâmetros escolhidos e não substitui a validação independente exigida pela governança de modelos da instituição.
\end{itemize}

\section*{Referências}
\addcontentsline{toc}{section}{Referências}
\begin{itemize}\setlength\itemsep{1pt}
""" + "\n".join(rf"\item[] [{i}] {r}" for i, r in enumerate(REFS_TEX, 1)) + r"""
\end{itemize}
""")
    if descritiva:
        _ta, _fa = _apendice_descritiva(descritiva, figdir, nome_base)
        T.append(_ta)
        figs += _fa
    T.append(_apendice_alocacao(results, primeiro=not descritiva,
                                letra="B" if descritiva else "A"))
    _tc, _fc = _apendice_cea(results, figdir, rel_agg, descritiva)
    T.append(_tc)
    figs += _fc
    T.append(r"""
\end{document}
""")
    return "".join(T), figs


def gerar_paper_latex(port, results, mig=None, nome_base="", extra=None,
                      rel_agg=None, timeout=180, descritiva=None):
    """Compila o paper. Devolve (pdf_bytes, zip_fonte_bytes).
    Levanta RuntimeError se a compilação falhar."""
    with tempfile.TemporaryDirectory() as tmp:
        tex, figs = montar_tex(port, results, mig, nome_base, extra,
                               rel_agg, figdir=tmp, descritiva=descritiva)
        with open(os.path.join(tmp, "paper.tex"), "w", encoding="utf-8") as f:
            f.write(tex)
        for _ in range(2):           # 2 passagens: sumário e referências
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 "paper.tex"], cwd=tmp, capture_output=True, timeout=timeout)
        pdf_path = os.path.join(tmp, "paper.pdf")
        if r.returncode != 0 or not os.path.exists(pdf_path):
            log = r.stdout.decode("latin-1", "ignore")[-1500:]
            raise RuntimeError(f"Falha na compilação LaTeX:\n{log}")
        pdf = open(pdf_path, "rb").read()
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("paper.tex", tex)
            for fg in figs:
                z.write(os.path.join(tmp, fg), fg)
        return pdf, zbuf.getvalue()
