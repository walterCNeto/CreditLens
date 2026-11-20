"""Visão de ponta a ponta: da base de entrada ao capital econômico alocado.

CEA = PE + PI, na convenção usual dos bancos para o VaR de crédito:
  PE = EL_ND + ΔL                       (perda esperada do ano)
  PI = (VaR_ND − EL_ND) + K_D           (perda inesperada; idem com o ES)
  CEA = PE + PI = VaR_ND + ΔL + K_D     (= VaR da carteira total)
em que ND é a carteira não default (simulada), ΔL o incremento da provisão
das exposições em default em 12 meses e K_D = Σ EAD·(LGD_D − ELBE).
"""
import numpy as np
import pandas as pd

from .alocacao import _f

NIVEL = "99.90%"


def _default(results):
    rd = ((results.get("inadimplidos") or {}).get("resumo")) or {}
    dl = _f(rd.get("perda_incremental_12m"))
    kd = _f(rd.get("capital_irb"))
    return (0.0 if dl != dl else dl), (0.0 if kd != kd else kd), rd


def ead_total(results):
    _, _, rd = _default(results)
    ead_d = _f(rd.get("ead")) if rd else 0.0
    return _f(results.get("resumo", {}).get("ead_total")) + \
        (0.0 if ead_d != ead_d else ead_d)


def tabela_cea(results, medida="VaR", nivel=NIVEL):
    """CEA = PE + PI por modelo (todos os modelos do portfólio), com a PI
    medida pelo VaR ou pelo ES no nível dado. Ordenada do maior para o
    menor CEA; modelos sem a medida (p.ex. ES do binomial analítico) ao fim.
    Retorna (df_reais, df_pct_ead) ou (None, None)."""
    med = results.get("medidas")
    if med is None or not len(med):
        return None, None
    dl, kd, _ = _default(results)
    L = []
    for _, r in med.iterrows():
        nome = str(r["modelo"])
        if nome.startswith("IRB"):
            continue
        el, q = _f(r.get("EL")), _f(r.get(f"{medida} {nivel}"))
        pi_nd = q - el if q == q and el == el else np.nan
        L.append({"modelo": nome, "PE_ND": el, "PE_D": dl, "PE": el + dl,
                  "PI_ND": pi_nd, "PI_D": kd,
                  "PI": pi_nd + kd if pi_nd == pi_nd else np.nan,
                  "CEA": el + dl + pi_nd + kd if pi_nd == pi_nd else np.nan})
    df = pd.DataFrame(L).sort_values("CEA", ascending=False,
                                     na_position="last", kind="stable")
    df = df.reset_index(drop=True)
    ead = ead_total(results)
    pct = df.copy()
    num = [c for c in df.columns if c != "modelo"]
    if ead and ead == ead:
        pct[num] = df[num] / ead
    return df, pct


def conciliacao(results, rel_agg=None, descritiva=None):
    """Conciliação da base, etapa a etapa: linhas, EAD e PE em cada passo.
    Retorna (DataFrame, diferença de EAD entre entrada e consolidação)."""
    dl, kd, rd = _default(results)
    res = results.get("resumo", {})
    ead_nd = _f(res.get("ead_total"))
    el_nd = _f(res.get("perda_esperada"))
    n_d = int(rd.get("contratos", 0)) if rd else 0
    ead_d = _f(rd.get("ead")) if rd else 0.0
    ead_d = 0.0 if ead_d != ead_d else ead_d
    n_nd = int(rel_agg["linhas_analitica"]) if rel_agg else \
        int(res.get("n_contrapartes", 0))
    g = (descritiva or {}).get("geral") or {}
    n_in = int(g["linhas"]) if g else n_nd + n_d
    ead_in = _f(g["ead_total"]) if g else ead_nd + ead_d
    E = [("1. Base de entrada", n_in, ead_in, np.nan,
          "leitura e validação do layout")]
    if rd:
        E.append(("2a. Exposições em default (separadas)", n_d, ead_d,
                  dl, "PE = ⟦\\Delta L¦ΔL⟧, incremento da provisão em 12 meses"))
    E.append(("2b. Carteira não default", n_nd, ead_nd, el_nd,
              "PE = ⟦\\sum \\mathrm{PD}\\cdot\\mathrm{LGD}\\cdot\\mathrm{EAD}¦Σ PD·LGD·EAD⟧"))
    if rel_agg:
        E.append(("3. Agregação pooled", int(rel_agg["linhas_agregada"]),
                  ead_nd, _f(rel_agg["el_agregada"]),
                  f"{int(rel_agg['n_grandes'])} grandes + "
                  f"{int(rel_agg['n_buckets'])} grupos; EL preservada"))
    E.append(("4. Consolidação (2a + 2b)", n_nd + n_d, ead_nd + ead_d,
              el_nd + dl, "PE total da carteira"))
    df = pd.DataFrame(E, columns=["etapa", "linhas", "ead", "pe",
                                  "observacao"])
    return df, ead_in - (ead_nd + ead_d)


ETAPAS = [
    ("Base de entrada",
     "Lê a carteira contrato a contrato (EAD, PD, LGD, segmento, atraso, "
     "garantias) e valida o layout.",
     "carteira validada"),
    ("Separação",
     "Classifica em default os contratos com atraso de 90 dias ou mais, "
     "ativo problemático, meses em default informados ou PD = 1; os demais "
     "formam a carteira não default.",
     "carteira em default e não default"),
    ("Agregação (pooled)",
     "Mantém as maiores exposições individualmente e agrupa as demais por "
     "segmento e faixa de PD, preservando a EAD e a perda esperada.",
     "grandes exposições e grupos"),
    ("Calibração",
     "Ajusta cada família ao mesmo par (PD média, correlação de default "
     "alvo): fator de carga dos threshold, (a, b) da Beta, (w, v) do "
     "CreditRisk+.",
     "parâmetros por modelo"),
    ("Simulação e modelos analíticos",
     "Gera a distribuição de perdas de cada modelo por Monte Carlo ou por "
     "fórmula fechada (ASRF, binomial, CreditRisk+ clássico e ponto de sela).",
     "distribuição de perdas por modelo"),
    ("Medidas da carteira não default",
     "Extrai de cada distribuição a perda esperada e os quantis.",
     "⟦\\mathrm{EL}_{ND},\\ \\mathrm{VaR}_{ND},\\ \\mathrm{ES}_{ND}¦EL_ND, VaR_ND, ES_ND⟧"),
    ("Exposições em default",
     "Aplica a curva de workout (ELBE por tempo em default): incremento da "
     "provisão em 12 meses e margem de recuperação adversa.",
     "⟦\\Delta L¦ΔL⟧ e ⟦K_D¦K_D⟧"),
    ("Consolidação",
     "Soma as duas partes: "
     "⟦\\mathrm{PE}=\\mathrm{EL}_{ND}+\\Delta L¦PE = EL_ND + ΔL⟧; "
     "⟦\\mathrm{PI}=(\\mathrm{VaR}_{ND}-\\mathrm{EL}_{ND})+K_D¦"
     "PI = (VaR_ND − EL_ND) + K_D⟧; ⟦\\mathrm{CEA}=\\mathrm{PE}+\\mathrm{PI}¦"
     "CEA = PE + PI⟧.",
     "CEA por modelo"),
]


def fig_fluxo(path, results, rel_agg=None, descritiva=None):
    """Fluxograma monocromático da base de entrada ao CEA, com os números
    da execução."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    conc, _ = conciliacao(results, rel_agg, descritiva)
    dl, kd, rd = _default(results)
    c = {r.etapa.split(".")[0]: r for r in conc.itertuples(index=False)}

    def mi(v):
        return f"{v / 1e6:,.3f}".replace(",", "X").replace(".", ",") \
            .replace("X", ".")

    def nb(v):
        return f"{int(v):,}".replace(",", ".")

    n_mod = int(sum(1 for m in results["medidas"]["modelo"]
                    if not str(m).startswith("IRB")))
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.5, 100)
    ax.axis("off")
    B = {}

    def caixa(k, x, y, w, titulo, corpo, forte=False):
        n = corpo.count("\n") + 1
        h = 5.2 + 3.4 * n
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc="#efefef" if forte else "white",
                                    ec="black", lw=1.1 if forte else 0.8))
        ax.text(x, y + h / 2 - 1.4, titulo, ha="center", va="top",
                fontsize=7.6, weight="bold")
        ax.text(x, y + h / 2 - 5.0, corpo, ha="center", va="top",
                fontsize=6.8, linespacing=1.45)
        B[k] = (x, y - h / 2 - 0.3, y + h / 2 + 0.3)

    def seta(a, b, xa=None, xb=None):
        x0, y0 = (xa if xa is not None else B[a][0]), B[a][1]
        x1, y1 = (xb if xb is not None else B[b][0]), B[b][2]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black",
                                    shrinkA=0, shrinkB=0))

    b1 = c["1"]
    nd = c["2b"]
    xL, xR = 27, 75
    caixa("base", 50, 94, 46, "Base de entrada",
          f"{nb(b1.linhas)} linhas; EAD R$ {mi(b1.ead)} mi")
    caixa("sep", 50, 81, 50, "Separação",
          "atraso ≥ 90 dias, ativo problemático,\n"
          "meses em default informados ou PD = 1")
    caixa("nd", xL, 64, 42, "Carteira não default",
          f"{nb(nd.linhas)} linhas; EAD R$ {mi(nd.ead)} mi\n"
          f"PE analítica R$ {mi(nd.pe)} mi")
    if "3" in c:
        caixa("agr", xL, 49.5, 42, "Agregação pooled",
              f"{nb(c['3'].linhas)} linhas; EAD e EL preservadas")
    else:
        caixa("agr", xL, 49.5, 42, "Contrato a contrato", "sem agregação")
    caixa("mod", xL, 35, 42, "Calibração e modelos",
          f"{n_mod} modelos; os com dependência no mesmo $\\rho_D$ alvo\n"
          "simulação e fórmulas fechadas")
    caixa("med", xL, 19.5, 42, "Medidas não default",
          "$\\mathrm{PE}_{ND}=\\mathrm{EL}_{ND}$\n"
          "$\\mathrm{PI}_{ND}=\\mathrm{VaR}_{ND}-\\mathrm{EL}_{ND}$")
    caixa("cea", 50, 4.5, 62, "CEA = PE + PI  (= VaR da carteira total)",
          "$\\mathrm{PE}=\\mathrm{EL}_{ND}+\\Delta L$;    "
          "$\\mathrm{PI}=(\\mathrm{VaR}_{ND}-\\mathrm{EL}_{ND})+K_D$",
          forte=True)
    seta("base", "sep")
    seta("sep", "nd", xa=38)
    seta("nd", "agr")
    seta("agr", "mod")
    seta("mod", "med")
    seta("med", "cea", xb=36)
    if rd:
        d = c["2a"]
        caixa("def", xR, 64, 40, "Exposições em default",
              f"{nb(d.linhas)} linhas; EAD R$ {mi(d.ead)} mi\n"
              "curva de workout (ELBE)")
        caixa("pdef", xR, 35, 40, "Perda do ano no default",
              f"$\\mathrm{{PE}}_D=\\Delta L$ = R\\$ {mi(dl)} mi\n"
              f"$\\mathrm{{PI}}_D=K_D$ = R\\$ {mi(kd)} mi")
        seta("sep", "def", xa=62)
        seta("def", "pdef")
        seta("pdef", "cea", xb=64)
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)
