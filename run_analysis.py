"""
CLI do CreditLens: análise completa de uma carteira em lote.

Uso:
    python run_analysis.py data/carteira_teste.csv --M 100000 --rho 0.05 --out output/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.portfolio import load_portfolio
from engine.analytics import run_full_analysis, results_to_excel, DEFAULT_ALPHAS


def make_charts(results, port, outdir):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    total = results["resumo"]["exposicao_liquida_total"]

    # 1) Distribuições de perda (corpo + cauda)
    PAL = {"Binomial independente": "#9aa0a6",
           "Threshold Gaussiano 1F": "#1f4e79",
           "Threshold t-Student 1F": "#b02418",
           "Threshold Gaussiano multifatorial (setores)": "#2e8b57",
           "Threshold t-Student multifatorial (setores, nu=10)": "#e07b00"}
    LBL = {"Threshold Gaussiano multifatorial (setores)": "Threshold Gaussiano MF",
           "Threshold t-Student multifatorial (setores, nu=10)": "Threshold t-Student MF"}
    dists = results["distribuicoes"]
    xmax = max(np.quantile(d, 0.999) for d in dists.values())
    bins = np.linspace(0, xmax, 90)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for nome, dist in dists.items():
        cor, rot = PAL.get(nome, "#555"), LBL.get(nome, nome)
        ax[0].hist(dist, bins=bins, density=True, histtype="step",
                   label=rot, color=cor, lw=1.5)
        srt = np.sort(dist)
        tail = 1.0 - np.arange(1, len(srt) + 1) / len(srt)
        ax[1].semilogy(srt / total * 100, np.maximum(tail, 1e-6),
                       label=rot, color=cor, lw=1.5)
    ax[0].set_xlim(0, xmax)
    ax[0].set_xlabel("Perda"); ax[0].set_ylabel("Densidade")
    ax[0].set_title("Corpo da distribuição (até o quantil 99,9%)")
    ax[0].legend(fontsize=7, framealpha=0.9)
    ax[1].set_xlabel("Perda (% da exposição líquida)")
    ax[1].set_ylabel("P(L > x)  (escala log)")
    ax[1].set_title("Cauda da distribuição")
    ax[1].set_ylim(1e-5, 1)
    ax[1].axhline(0.001, color="k", ls=":", lw=0.8)
    ax[1].text(0.3, 0.0013, "99,9%", fontsize=7)
    ax[1].legend(fontsize=7, framealpha=0.9)
    ax[1].grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "distribuicoes_perda.png"))
    plt.close(fig)

    # 2) VaR por modelo (barras)
    med = results["medidas"]
    col_var = [c for c in med.columns if c.startswith("VaR 99.90")][0]
    dfp = med.dropna(subset=[col_var]).sort_values(col_var)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(dfp["modelo"], dfp[col_var] / total * 100, color="#1f4e79")
    ax.set_xlabel("VaR 99,90% (% da exposição líquida)")
    ax.set_title("VaR 99,90% por modelo")
    for i, v in enumerate(dfp[col_var] / total * 100):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "var_por_modelo.png"))
    plt.close(fig)

    # 3) Contribuições de risco (top 15)
    if "contribuicoes" in results:
        cc = results["contribuicoes"]
        df_c = port.df[["id_contraparte", "rating"]].copy() \
            if "rating" in port.df.columns else port.df[["id_contraparte"]].copy()
        df_c["contrib_ES"] = cc["es_contrib"]
        top = df_c.sort_values("contrib_ES", ascending=False).head(15)[::-1]
        labels = (top["id_contraparte"] + " (" + top["rating"] + ")"
                  ) if "rating" in top.columns else top["id_contraparte"]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.barh(labels, top["contrib_ES"] / cc["es_medio"] * 100, color="#b02418")
        ax.set_xlabel(f"Contribuição ao ES {cc['alpha']:.0%} (% do total)")
        ax.set_title("Top 15 contribuições de risco (decomposição MC, threshold gaussiano)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "contribuicoes_risco.png"))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="CreditLens — análise de carteira de crédito")
    ap.add_argument("carteira", help="CSV no padrão CreditLens")
    ap.add_argument("--M", type=int, default=100_000, help="nº de simulações MC")
    ap.add_argument("--rho", type=float, default=0.05, help="correlação de default alvo")
    ap.add_argument("--nu", type=float, default=10.0, help="graus de liberdade (t-Student)")
    ap.add_argument("--S", type=int, default=10, help="iterações da decomposição de risco")
    ap.add_argument("--sem-contribuicoes", action="store_true")
    ap.add_argument("--multifator", action="store_true",
                    help="ativa o modelo multifatorial setorial (exige coluna setor)")
    ap.add_argument("--rho-intra", type=float, default=0.08,
                    help="correlação de default alvo intra-setor")
    ap.add_argument("--rho-inter", type=float, default=0.03,
                    help="correlação de default alvo inter-setor")
    ap.add_argument("--out", default="output")
    ap.add_argument("--migracao", default=None,
                    help="CSV da matriz de transição anual (ativa ECL lifetime e multiperíodo)")
    ap.add_argument("--horizonte", type=int, default=5, help="horizonte multiperíodo (anos)")
    ap.add_argument("--taxa-desconto", type=float, default=0.10, help="taxa de desconto ao ano")
    ap.add_argument("--block", type=int, default=10_000, help="tamanho do bloco de simulação")
    ap.add_argument("--analitica", action="store_true",
                    help="trata o CSV como base analítica: agrega em grandes + GH×setor e roda o motor pooled")
    ap.add_argument("--max-grandes", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.analitica:
        from engine.aggregation import load_analytic_and_aggregate
        port, rel = load_analytic_and_aggregate(args.carteira,
                                                max_grandes=args.max_grandes)
        print(f"[agregação] {rel['linhas_analitica']:,} contratos -> "
              f"{rel['linhas_agregada']} linhas ({rel['n_grandes']} grandes + "
              f"{rel['n_buckets']} buckets) | EL dif. {rel['el_diferenca_pct']:.4%} | "
              f"CV PD máx. {rel['pd_cv_max_bucket']:.2f}")
        port.df.to_csv(os.path.join(args.out, "carteira_agregada.csv"), index=False)
    else:
        port = load_portfolio(args.carteira)
    for w in port.warnings:
        print(f"[aviso] {w}")

    s = port.summary()
    print(f"\nCarteira: {s['n_contrapartes']} contrapartes | "
          f"EAD {s['ead_total']:,.0f} | EL {s['perda_esperada']:,.1f} "
          f"({s['perda_esperada_pct']:.2%}) | HHI {s['hhi']:.4f} "
          f"(N efetivo {s['n_efetivo']:.0f})\n")

    if args.analitica:
        from engine.analytics import run_pooled_analysis
        results = run_pooled_analysis(
            port, M=args.M, rho_target=args.rho, rho_intra=args.rho_intra,
            rho_inter=args.rho_inter, nu=args.nu,
            progress=lambda m: print(f"  -> {m}"))
    else:
        results = run_full_analysis(
        port, M=args.M, rho_target=args.rho, nu=args.nu,
        run_contributions=not args.sem_contribuicoes, contrib_S=args.S,
        run_multifactor=args.multifator, rho_intra=args.rho_intra,
        rho_inter=args.rho_inter,
        progress=lambda m: print(f"  -> {m}"),
    )

    print("\n=== Medidas de risco por modelo ===")
    with np.printoptions(precision=1):
        print(results["medidas"].to_string(index=False,
              float_format=lambda x: f"{x:,.1f}"))

    irb_res = results["irb"]
    print(f"\n=== Capital IRB de Basileia ===")
    print(f"Capital (VaR 99,90% inesperado): {irb_res['capital_total_999']:,.1f} "
          f"({irb_res['capital_pct_exposicao']:.2%} da exposição líquida)")
    print(f"Ajuste de granularidade:        {irb_res['ajuste_granularidade']:,.1f}")
    print(f"Capital + GA:                   {irb_res['capital_com_ga']:,.1f}")
    print(f"RWA equivalente (12,5x):        {irb_res['rwa_equivalente']:,.1f}")

    if "multifator" in results:
        mf = results["multifator"]
        cal = mf["calibracao"]
        print(f"\n=== Multifatorial setorial ({len(mf['setores'])} setores) ===")
        print(f"rho_ativos intra {cal['rho_ativos_intra']:.2%} | inter {cal['rho_ativos_inter']:.2%} "
              f"(alvos default: {cal['rho_default_intra_alvo']:.0%}/{cal['rho_default_inter_alvo']:.0%}) "
              f"| HHI setorial {mf['hhi_setorial']:.4f}")
        print(mf["contrib_por_setor"][["setor","n_contrapartes","exposicao_pct",
              "contrib_ES_pct","razao_risco_exposicao"]].to_string(
              index=False, float_format=lambda x: f"{x:.3f}"))

    mig_out = None
    if args.migracao:
        from engine.migration import (load_transition_matrix, compute_ecl,
                                      simulate_multiperiod, pd_term_structure)
        import numpy as _np
        P, ratings = load_transition_matrix(args.migracao)
        ecl_df, cum, T_ecl = compute_ecl(port, P, ratings,
                                         discount_rate=args.taxa_desconto)
        cal = results.get("multifator", {}).get("calibracao")
        if args.analitica:
            from engine.migration import simulate_multiperiod_pooled
            if cal:
                import numpy as _np
                _, sector_idx = _np.unique(port.df["setor"].astype(str),
                                           return_inverse=True)
                sim = simulate_multiperiod_pooled(port, P, ratings,
                    M=args.M, T=args.horizonte, rho_g=cal["rho_ativos_inter"],
                    rho_s=cal["rho_ativos_intra"], sector_idx=sector_idx,
                    discount_rate=args.taxa_desconto, block_size=args.block,
                    progress=lambda m: print(f"  -> {m}"))
            else:
                sim = simulate_multiperiod_pooled(port, P, ratings,
                    M=args.M, T=args.horizonte,
                    rho_g=results["calibracao_gaussiana"]["rho_ativos"],
                    discount_rate=args.taxa_desconto, block_size=args.block,
                    progress=lambda m: print(f"  -> {m}"))
        elif cal:
            sec = port.df["setor"].astype(str)
            _, sector_idx = _np.unique(sec, return_inverse=True)
            sim = simulate_multiperiod(port, P, ratings, M=args.M, T=args.horizonte,
                rho_g=cal["rho_ativos_inter"], rho_s=cal["rho_ativos_intra"],
                sector_idx=sector_idx, discount_rate=args.taxa_desconto,
                block_size=args.block, progress=lambda m: print(f"  -> {m}"))
        else:
            rho1 = results["calibracao_gaussiana"]["rho_ativos"]
            sim = simulate_multiperiod(port, P, ratings, M=args.M, T=args.horizonte,
                rho_g=rho1, discount_rate=args.taxa_desconto,
                block_size=args.block, progress=lambda m: print(f"  -> {m}"))
        mig_out = {"ecl": ecl_df, "cum": cum, "ratings": ratings, "sim": sim}
        print(f"\n=== ECL (matriz de migração; desconto {args.taxa_desconto:.1%} a.a.) ===")
        print(f"ECL 12m total:      {ecl_df['ECL_12m'].sum():,.2f}")
        print(f"ECL lifetime total: {ecl_df['ECL_lifetime'].sum():,.2f} "
              f"({ecl_df['ECL_lifetime'].sum()/ecl_df['ECL_12m'].sum():.2f}x)")
        if "provisao" in ecl_df:
            n2 = int((ecl_df['estagio']==2).sum())
            print(f"Estágio 2 (deterioração >=2 notches): {n2} nomes | "
                  f"provisão por estágio: {ecl_df['provisao'].sum():,.2f}")
        if sim is not None:
            print(f"\n=== Multiperíodo ({sim['T']} anos, M={sim['M']:,}, "
                  f"blocos de {sim['block_size']:,}) ===")
            print(f"EL {sim['EL']:,.1f} | UL {sim['UL']:,.1f} | "
                  f"VaR99,9 {sim['VaR']['99.90%']:,.1f} | ES99,9 {sim['ES']['99.90%']:,.1f}")
            print("Fração média da carteira em default por ano:",
                  " ".join(f"{d:.2%}" for d in sim['defaults_por_ano']))

    xlsx = os.path.join(args.out, "resultados_analise.xlsx")
    results_to_excel(results, port, xlsx)
    if mig_out is not None:
        import pandas as _pd
        from openpyxl import load_workbook
        with _pd.ExcelWriter(xlsx, engine="openpyxl", mode="a") as xw:
            mig_out["ecl"].to_excel(xw, sheet_name="ECL", index=False)
            _pd.DataFrame(mig_out["cum"], index=mig_out["ratings"][:-1],
                columns=[f"ano {t+1}" for t in range(mig_out["cum"].shape[1])]
                ).to_excel(xw, sheet_name="PD Acumulada")
        # gráficos de migração
        cum, ratings = mig_out["cum"], mig_out["ratings"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        anos = np.arange(1, cum.shape[1] + 1)
        for i, r in enumerate(ratings[:-1]):
            ax[0].semilogy(anos, cum[i] * 100, marker="o", ms=3, label=r)
        ax[0].set_xlabel("Horizonte (anos)"); ax[0].set_ylabel("PD acumulada (%)")
        ax[0].set_title("Estrutura a termo de PD por rating"); ax[0].legend(fontsize=7, ncol=2)
        ax[0].grid(alpha=0.25, which="both")
        e = mig_out["ecl"].groupby("rating")[["ECL_12m","ECL_lifetime"]].sum()
        ordem = [r for r in ratings[:-1] if r in e.index]
        e = e.loc[ordem]
        x = np.arange(len(e)); w = 0.38
        ax[1].bar(x - w/2, e["ECL_12m"], w, label="ECL 12 meses", color="#1f4e79")
        ax[1].bar(x + w/2, e["ECL_lifetime"], w, label="ECL lifetime", color="#b02418")
        ax[1].set_xticks(x); ax[1].set_xticklabels(e.index)
        ax[1].set_title("ECL por rating"); ax[1].legend(fontsize=8)
        ax[1].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "migracao_ecl.png")); plt.close(fig)
    make_charts(results, port, args.out)
    print(f"\nResultados salvos em: {xlsx} + gráficos PNG em {args.out}/")


if __name__ == "__main__":
    main()
