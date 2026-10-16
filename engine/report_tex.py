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
        "−": "--", "→": r"$\to$", "∝": r"$\propto$", "·": r"$\cdot$",
        "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$",
        "δ": r"$\delta$", "λ": r"$\lambda$", "μ": r"$\mu$",
        "ξ": r"$\xi$", "Φ": r"$\Phi$", "Σ": r"$\Sigma$",
        "χ": r"$\chi$", "²": r"$^2$", "≈": r"$\approx$"}


def tx(s) -> str:
    """Escapa texto livre para LaTeX (inclui símbolos unicode usuais)."""
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
\multicolumn{{{n}}}{{l}}{{\emph{{Tabela~\ref{{{rotulo}}} --- continuação}}}}\\
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
    r"IFRS Foundation (2014). \emph{IFRS 9 --- Financial Instruments}. Londres. Conselho Monetário Nacional (2021). \emph{Resolução nº 4.966, de 25 de novembro de 2021}. Brasília: BCB.",
    r"Conselho Monetário Nacional (2017). \emph{Resolução nº 4.557, de 23 de fevereiro de 2017} (estrutura de gerenciamento de riscos e de capital). Brasília: BCB.",
    r"Banco Central do Brasil (2023). \emph{Resolução BCB nº 303, de 16 de março de 2023} (abordagem IRB para risco de crédito; revoga a Circular nº 3.648/2013). Brasília: BCB.",
    r"Banco Central do Brasil (2022). \emph{Resolução BCB nº 229, de 12 de maio de 2022} (RWA de risco de crédito pela abordagem padronizada --- RWA\textsubscript{CPAD}; texto consolidado). Brasília: BCB.",
    r"David, H.~A.; Nagaraja, H.~N. (2003). \emph{Order Statistics}. 3.\,ed. Hoboken: Wiley.",
    r"Glasserman, P. (2004). \emph{Monte Carlo Methods in Financial Engineering}. Nova York: Springer.",
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


def _fig_multiperiodo(sim, path, unidade="R$ milhões"):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hist(sim["perdas"], bins=80, density=True, color="#12395B",
            alpha=0.85)
    v9 = sim["VaR"].get("99.90%")
    if v9 == v9:
        ax.axvline(v9, color="#B02418", ls="--", lw=1.2)
        ax.text(v9, ax.get_ylim()[1] * 0.9, " VaR 99,9%", fontsize=8,
                color="#B02418")
    ax.set_xlabel(f"Perda descontada acumulada ({unidade})")
    ax.spines[["top", "right"]].set_visible(False)
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
""" + rf"""Este apêndice descreve a base de entrada \emph{{original}}{(' (' + tx(nome_base) + ')') if nome_base else ''} --- {'contrato a contrato, antes da agregação' if g['linhas'] > 5_000 else 'como enviada'} ---, para que o validador avalie os dados que sustentam a execução. A base tem {nb(g['linhas'], 0)} linhas e {nb(g['contrapartes'], 0)} contrapartes distintas, exposição total de R\$~{mi(g['ead_total'])}~milhões e perda esperada de R\$~{mi(g['el_total'])}~milhões, isto é, {nb(g['el_pct'], 3, pct=True)} da severidade (PD média ponderada pela severidade de {nb(g['pd_media_ponderada'], 3, pct=True)} e LGD média ponderada pela exposição de {nb(g['lgd_media_ponderada'], 3, pct=True)}).
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
em que $p_g(Z)$ é a PD do grupo condicional ao fator (a expressão de cada modelo). A agregação reduz a simulação de $M\times N$ para $M\times G$ sorteios sem aproximação --- $N$ contratos, $G$ grupos --- e é o que torna viável a carteira de milhões de linhas (Frey e McNeil, 2003).

\paragraph{Preservação da perda esperada e número efetivo.} Os parâmetros do grupo são ponderados pela severidade,
\begin{equation}
C_g=\sum_{i\in g}c_i,\qquad p_g=\frac{\sum_{i\in g}p_i\,c_i}{C_g},\qquad n_g^{\mathrm{eff}}=\frac{C_g^2}{\sum_{i\in g}c_i^2},
\end{equation}
em que $C_g$ é a severidade total, $p_g$ a PD ponderada --- com a qual $p_g C_g=\sum_i p_i c_i$ e a EL é preservada exatamente --- e $n_g^{\mathrm{eff}}$ o número efetivo de contratos, um índice de Herfindahl interno ao grupo. Com severidades desiguais, a variância idiossincrática condicional do grupo é $\sum_i c_i^2\,p_g(Z)(1-p_g(Z))=C_g^2\,p_g(Z)(1-p_g(Z))/n_g^{\mathrm{eff}}$: simular $\mathrm{Binomial}(n_g^{\mathrm{eff}},p_g(Z))$ com severidade $C_g/n_g^{\mathrm{eff}}$ por default preserva a média e a variância condicionais, enquanto usar o número bruto de contratos superestimaria a granularidade. Para a parte idiossincrática, tratar PDs heterogêneas como homogêneas é conservador: a variância de uma soma de Bernoullis heterogêneas nunca excede a da Binomial com a PD média (desigualdade de Hoeffding).

\paragraph{Limite de granularidade infinita e seu critério.} Chama-se de \emph{granularidade infinita} o limite em que o grupo tem tantos contratos, e tão pequenos, que o risco idiossincrático se dilui por completo --- na literatura em inglês, \emph{large pool} ou carteira \emph{infinitely fine-grained} (Gordy, 2003). Pela lei dos grandes números condicional, a fração em default converge para $p_g(Z)$ e a perda do grupo para $L_g=C_g\,p_g(Z)$ --- o limite de Vasicek (2002) que fundamenta o ASRF e o IRB (Gordy, 2003). O limite descarta a variância idiossincrática; a razão entre ela e a sistemática é
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
    if tem("CreditRisk+"):
        T.append(rf"""
\subsection{{CreditRisk\textsuperscript{{+}} (Poisson-gama, um fator)}}
O fator sistemático $S\sim\Gamma(v,1/v)$, com $\E[S]=1$ e $\mathrm{{Var}}(S)=1/v$, escala as intensidades de default:
\begin{{equation}}
H_n\mid S\sim\mathrm{{Poisson}}\bigl(p_n(1-w+wS)\bigr),
\end{{equation}}
em que $H_n$ é a contagem de defaults, $w={nm(prm.get('crplus_w'))}$ a carga do fator e $v={nm(prm.get('crplus_v'))}$ a dispersão (Credit Suisse First Boston, 1997)."""
                 + (r" Na representação agregada, a contagem do grupo agrega exatamente, pois a soma de Poissons independentes é Poisson: $H_g\mid S\sim\mathrm{Poisson}\bigl(n_g p_g(1-w+wS)\bigr)$, com perda $L_g=(C_g/n_g)\,H_g$." if pooled
                    else r" No fluxo granular, o default é declarado quando $H_n\ge1$, de modo que $\Pr(D_n)=1-e^{-p_n}<p_n$: a EL simulada fica ligeiramente abaixo da analítica.") + "\n")
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
em que $\Phi_2(\cdot,\cdot;\rho)$ é a distribuição normal bivariada com correlação $\rho$. A variante t-Student, com $\nu={nm(prm.get('nu_t', 10), 0)}$ graus de liberdade, multiplica as variáveis latentes por um choque radial comum $\sqrt{{\nu/V}}$, $V\sim\chi^2_\nu$, e introduz dependência de cauda --- o contraste entre as duas variantes é um teste de sensibilidade à premissa de cauda.
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
A correlação de ativos dentro do setor $s$ é $\rho_{S,s}$; entre os setores $s$ e $t$, $\sqrt{\rho_{G,s}\,\rho_{G,t}}$. A estrutura entre setores é de posto 1: os pares diferem pela carga de cada setor no fator global, mas a razão entre as correlações de um setor com outros dois é a mesma para todo setor --- blocos de setores exigiriam um segundo fator global. Os valores de $\rho_S$ e $\rho_G$ acima são médias ponderadas pela exposição; a Tabela~\ref{tab:mf-setor} traz a calibração de cada setor."""
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
\paragraph{Leitura das escolhas.} A fração da correlação de ativos intra-setor que vem do fator global, $\rho_{G,s}/\rho_{S,s}$, indica a origem da dependência: perto de 1, o setor é \emph{cíclico} --- os defaults andam juntos porque a economia os move juntos, e o setor contagia os demais; perto de 0, é \emph{concentrado} --- um choque próprio derruba muitos nomes de uma vez, mas pouco se espalha. Nesta execução:
\begin{itemize}\setlength\itemsep{1pt}
""" + "".join(rf"\item \textbf{{{tx(r['setor'])}}} (fração global {nb(r['fracao_global'])}, {tx(r['perfil'])}): {tx(r['leitura'])}." + "\n"
              for _, r in lps.iterrows()) + rf"""\end{{itemize}}
O par de setores mais ligado é {tx(a1)}--{tx(b1)}, com correlação de default de {nb(v1, 3, pct=True)}; o menos ligado, {tx(az)}--{tx(bz)}, com {nb(vz, 3, pct=True)}. As correlações de ativos médias, ponderadas pela exposição, ficaram em {nm(lres['rho_s_medio'])} (intra) e {nm(lres['rho_g_medio'])} (inter), ante {nm(lres['rho_s_ref'])} e {nm(lres['rho_g_ref'])} na calibração uniforme --- {tx({'acima': 'acima do padrão', 'abaixo': 'abaixo do padrão'}.get(lres['direcao'], 'em linha com o padrão'))}. Os alvos por setor são hipóteses do usuário; esta seção verifica a sua calibração, não a sua fundamentação empírica.
""")
    if tem("ASRF"):
        T.append(r"""
\subsection{ASRF: carteira assintoticamente granular}
O limite analítico de Vasicek (2002), para uma carteira de granularidade infinita --- número de exposições tendendo ao infinito, cada uma com peso desprezível --- e um único fator sistemático,
\begin{equation}
\VaR_\alpha=C\,\Phi\!\left(\frac{\Phi^{-1}(\bar p)+\sqrt{\rho}\,\Phi^{-1}(\alpha)}{\sqrt{1-\rho}}\right),
\end{equation}
em que $C=\sum_n c_n$ é a exposição líquida e $\alpha$ o nível de confiança, fundamenta a função de capital do IRB (Gordy, 2003).
""")
    if irb:
        seg = irb.get("por_segmento")
        cats = ", ".join(seg["descricao"]) if seg is not None and len(seg) else "atacado --- demais"
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
""")
    if mig is not None:
        T.append(r"""
\subsection{Migração de ratings, ECL e capital multiperíodo}
A matriz de transição anual $P$ fornece a estrutura a termo de PD pelas potências $P^t$; o ECL desconta as perdas marginais em 12 meses e no prazo de cada contrato (\emph{lifetime}), com estágios por deterioração de rating, conforme o IFRS 9 e a Resolução CMN 4.966. O capital multiperíodo evolui a carteira ano a ano pela matriz condicional ao fator sistemático --- generalização do deslocamento de crédito de um parâmetro de Belkin, Suchower e Forest (1998) ---, acumulando perdas descontadas."""
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
                rf"\item {cor[st_]} --- \textbf{{{tx(teste)}}}: {tx(leitura)} \emph{{Ação:}} {tx(acao)}" + "\n"
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
\paragraph{{Exposições em \emph{{default}}.}} A Tabela~\ref{{tab:default}} resume o tratamento das exposições em \emph{{default}} e a Tabela~\ref{{tab:defaultfaixa}} mostra, por tempo em \emph{{default}}, a passagem do capital para a provisão; a Figura~\ref{{fig:default}} traz a curva usada. A perda incremental de R\$~{mi(r_['perda_incremental_12m'])}~milhões em 12 meses soma-se às medidas da Tabela~\ref{{tab:medidas}}, que se referem à carteira não \emph{{default}}.
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
