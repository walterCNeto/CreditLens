"""
CreditLens — Análise de risco de carteiras de crédito (Streamlit).

Executar:  streamlit run app.py
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from engine.portfolio import load_portfolio
from engine.analytics import run_full_analysis, results_to_excel

st.set_page_config(page_title="CreditLens", page_icon="📊", layout="wide")

st.title("CreditLens — Análise de Carteira de Crédito")
st.caption(
    "Motor quantitativo baseado em Bolder (2018), *Credit-Risk Modelling* (Springer): "
    "modelos independentes, mixture models (beta-binomial, CreditRisk+), threshold models "
    "(Gaussiano/t, ASRF), capital IRB de Basileia com ajuste de granularidade e "
    "decomposição de contribuições de risco."
)

# ---------------- Sidebar: parâmetros ----------------
with st.sidebar:
    st.header("Parâmetros")
    M = st.select_slider("Simulações Monte Carlo (M)",
                         options=[10_000, 50_000, 100_000, 250_000],
                         value=100_000)
    st.caption("Em hospedagem gratuita (Streamlit Cloud, ~1 GB RAM), "
               "prefira M ≤ 100 mil para carteiras com centenas de nomes.")
    rho_target = st.slider("Correlação de default alvo (ρ)", 0.01, 0.20, 0.05, 0.01)
    nu = st.slider("Graus de liberdade t-Student (ν)", 3, 50, 10)
    st.subheader("CreditRisk+")
    crp_w = st.slider("Carga do fator (w)", 0.05, 0.95, 0.35, 0.05)
    crp_v = st.slider("Parâmetro gama (v)", 0.2, 5.0, 1.2, 0.1)
    st.subheader("Multifatorial setorial")
    run_mf = st.checkbox("Ativar modelo multifatorial (exige coluna setor)", value=True)
    rho_intra = st.slider("ρ default alvo intra-setor", 0.01, 0.30, 0.08, 0.01)
    rho_inter = st.slider("ρ default alvo inter-setor", 0.01, 0.30, 0.03, 0.01)
    if rho_inter > rho_intra:
        st.error("O ρ inter-setor não pode exceder o intra-setor.")
    st.subheader("Migração e ECL (multiperíodo)")
    mtx_up = st.file_uploader("Matriz de transição anual (CSV)", type=["csv"],
                              key="mtx")
    usar_mtx_demo = st.checkbox("Usar matriz de exemplo", value=False)
    horizonte = st.slider("Horizonte multiperíodo (anos)", 1, 10, 5)
    taxa_desc = st.slider("Taxa de desconto (a.a.)", 0.0, 0.30, 0.10, 0.01)
    block_sz = st.select_slider("Bloco de simulação (memória)",
                                options=[2_000, 5_000, 10_000, 25_000], value=10_000)
    st.subheader("Contribuições de risco")
    run_contrib = st.checkbox("Calcular contribuições de VaR/ES", value=True)
    contrib_S = st.slider("Iterações da decomposição (S)", 5, 30, 10)
    contrib_alpha = st.selectbox("Nível α das contribuições", [0.95, 0.99, 0.999], index=1)
    seed = st.number_input("Semente aleatória", value=42, step=1)

# ---------------- Upload ----------------
st.subheader("1. Carregue a carteira")
col1, col2 = st.columns([2, 1])
with col1:
    up = st.file_uploader("CSV no padrão CreditLens", type=["csv"])
with col2:
    template_path = os.path.join(os.path.dirname(__file__), "templates", "carteira_modelo.csv")
    if os.path.exists(template_path):
        st.download_button("⬇ Baixar modelo de carteira",
                           open(template_path, "rb").read(),
                           file_name="carteira_modelo.csv", mime="text/csv")
    demo = st.checkbox("Usar carteira de teste (250 contrapartes)")

with st.expander("Padrão do arquivo"):
    st.markdown(
        "- **Obrigatórias:** `id_contraparte`, `exposicao` (EAD > 0), `pd` ∈ (0,1)\n"
        "- **Opcionais:** `lgd` ∈ (0,1] *(default 1,0)*, `prazo_anos` *(default 2,5; "
        "truncado a [1,5] para o IRB)*, `rating`, `setor`, `nome`\n"
        "- Separador vírgula, decimal ponto, UTF-8. Severidade Bolder: c = EAD × LGD."
    )

st.markdown("**Ou parta de uma base analítica** (contrato a contrato; será "
            "agregada em grandes exposições + grupos homogêneos × setor antes dos modelos):")
ca1, ca2, ca3 = st.columns([2, 1, 1])
with ca1:
    up_analitica = st.file_uploader("Base analítica (CSV)", type=["csv"],
                                    key="analitica")
with ca2:
    demo_analitica = st.checkbox("Gerar base analítica demo (300 mil contratos)")
with ca3:
    max_grandes = st.number_input("Máx. grandes exposições", 0, 2000, 200, 50)

source = None
analitica_src = None
if up_analitica is not None:
    analitica_src = up_analitica
elif demo_analitica:
    with st.spinner("Gerando 300 mil contratos sintéticos..."):
        from gerar_base_analitica import gerar
        analitica_src = gerar(path=None)
elif up is not None:
    source = up
elif demo:
    source = os.path.join(os.path.dirname(__file__), "data", "carteira_teste.csv")

if source is None and analitica_src is None:
    st.info("Carregue um CSV (padrão ou base analítica) ou marque uma das "
            "opções de demonstração para começar.")
    st.stop()

pooled_mode = False
if analitica_src is not None:
    from engine.aggregation import aggregate_portfolio
    import pandas as _pd, io as _io
    try:
        df_ana = analitica_src if isinstance(analitica_src, _pd.DataFrame) \
            else _pd.read_csv(analitica_src)
        with st.spinner(f"Agregando {len(df_ana):,} linhas..."):
            df_agg, rel = aggregate_portfolio(df_ana, corte_grandes=None,
                                              max_grandes=int(max_grandes))
        buf = _io.StringIO(); df_agg.to_csv(buf, index=False); buf.seek(0)
        port = load_portfolio(buf)
        pooled_mode = True
        st.success(f"Agregação: {rel['linhas_analitica']:,} contratos → "
                   f"{rel['linhas_agregada']} linhas "
                   f"({rel['n_grandes']} grandes exposições + "
                   f"{rel['n_buckets']} buckets GH×setor).")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("EL analítica", f"{rel['el_analitica']:,.1f}")
        q2.metric("EL agregada", f"{rel['el_agregada']:,.1f}",
                  f"dif. {rel['el_diferenca_pct']:.4%}", delta_color="off")
        q3.metric("Exposição nos grandes", f"{rel['exposicao_grandes_pct']:.1%}")
        q4.metric("CV de PD máx. no bucket", f"{rel['pd_cv_max_bucket']:.2f}",
                  f"{rel['buckets_cv_alto']} buckets com CV>0,5", delta_color="off")
        if rel['buckets_cv_alto'] > 0:
            st.warning("Há buckets com dispersão alta de PD — considere refinar "
                       "os grupos homogêneos (coluna 'gh') para reduzir o viés "
                       "de Jensen nas funções de capital.")
    except ValueError as e:
        st.error(f"Base analítica inválida: {e}")
        st.stop()
else:
    try:
        port = load_portfolio(source)
    except ValueError as e:
        st.error(f"Carteira inválida: {e}")
        st.stop()
    pooled_mode = "n_contratos" in port.df.columns and \
        (port.df["n_contratos"] > 1).any()

for w in port.warnings:
    st.warning(w)

# ---------------- Visão geral ----------------
st.subheader("2. Visão geral da carteira")
s = port.summary()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Contrapartes", f"{s['n_contrapartes']:,}")
c2.metric("EAD total", f"{s['ead_total']:,.0f}")
c3.metric("Perda esperada", f"{s['perda_esperada']:,.1f}",
          f"{s['perda_esperada_pct']:.2%} da exposição", delta_color="off")
c4.metric("PD média ponderada", f"{s['pd_media_ponderada']:.2%}")
c5.metric("HHI (N efetivo)", f"{s['hhi']:.4f}", f"{s['n_efetivo']:.0f} nomes",
          delta_color="off")

with st.expander("Ver carteira carregada"):
    st.dataframe(port.df, width='stretch', height=280)

# ---------------- Execução ----------------
st.subheader("3. Análise")
if st.button("▶ Executar modelos", type="primary"):
    prog = st.progress(0.0, text="Iniciando...")
    steps = {"n": 0}

    def tick(msg):
        steps["n"] += 1
        prog.progress(min(steps["n"] / 12, 0.99), text=msg)

    with st.spinner("Rodando modelos..."):
        if pooled_mode:
            from engine.analytics import run_pooled_analysis
            results = run_pooled_analysis(
                port, M=M, rho_target=rho_target,
                rho_intra=rho_intra, rho_inter=rho_inter, nu=float(nu),
                contrib_alpha=contrib_alpha, seed=int(seed), progress=tick)
        else:
            results = run_full_analysis(
            port, M=M, rho_target=rho_target, nu=float(nu),
            crp_w=crp_w, crp_v=crp_v,
            run_contributions=run_contrib, contrib_S=contrib_S,
            contrib_alpha=contrib_alpha, seed=int(seed), progress=tick,
            run_multifactor=run_mf and rho_inter <= rho_intra
                and "setor" in port.df.columns,
            rho_intra=rho_intra, rho_inter=rho_inter,
        )
    # --- migração & ECL ---
    mtx_src = mtx_up if mtx_up is not None else (
        os.path.join(os.path.dirname(__file__), "templates",
                     "matriz_migracao_modelo.csv") if usar_mtx_demo else None)
    st.session_state.pop("mig", None)
    if mtx_src is not None and "rating" in port.df.columns:
        try:
            from engine.migration import (load_transition_matrix, compute_ecl,
                                          simulate_multiperiod)
            P_m, ratings_m = load_transition_matrix(mtx_src)
            ecl_df, cum_m, _ = compute_ecl(port, P_m, ratings_m,
                                           discount_rate=taxa_desc)
            if pooled_mode:
                from engine.migration import simulate_multiperiod_pooled
                cal_mf = results.get("multifator", {}).get("calibracao")
                if cal_mf:
                    _, sector_idx = np.unique(port.df["setor"].astype(str),
                                              return_inverse=True)
                    sim_m = simulate_multiperiod_pooled(port, P_m, ratings_m,
                        M=M, T=horizonte, rho_g=cal_mf["rho_ativos_inter"],
                        rho_s=cal_mf["rho_ativos_intra"], sector_idx=sector_idx,
                        discount_rate=taxa_desc, block_size=block_sz,
                        seed=int(seed), progress=tick)
                else:
                    sim_m = simulate_multiperiod_pooled(port, P_m, ratings_m,
                        M=M, T=horizonte,
                        rho_g=results["calibracao_gaussiana"]["rho_ativos"],
                        discount_rate=taxa_desc, block_size=block_sz,
                        seed=int(seed), progress=tick)
                st.session_state["mig"] = {"ecl": ecl_df, "cum": cum_m,
                                           "ratings": ratings_m, "sim": sim_m}
                raise StopIteration
            cal_mf = results.get("multifator", {}).get("calibracao")
            if cal_mf:
                _, sector_idx = np.unique(port.df["setor"].astype(str),
                                          return_inverse=True)
                sim_m = simulate_multiperiod(port, P_m, ratings_m, M=M,
                    T=horizonte, rho_g=cal_mf["rho_ativos_inter"],
                    rho_s=cal_mf["rho_ativos_intra"], sector_idx=sector_idx,
                    discount_rate=taxa_desc, block_size=block_sz,
                    seed=int(seed), progress=tick)
            else:
                sim_m = simulate_multiperiod(port, P_m, ratings_m, M=M,
                    T=horizonte,
                    rho_g=results["calibracao_gaussiana"]["rho_ativos"],
                    discount_rate=taxa_desc, block_size=block_sz,
                    seed=int(seed), progress=tick)
            st.session_state["mig"] = {"ecl": ecl_df, "cum": cum_m,
                                       "ratings": ratings_m, "sim": sim_m}
        except StopIteration:
            pass
        except ValueError as e:
            st.error(f"Migração/ECL: {e}")
    prog.progress(1.0, text="Concluído")
    st.session_state["results"] = results

if "results" not in st.session_state:
    st.stop()

results = st.session_state["results"]
total = results["resumo"]["exposicao_liquida_total"]

tabs = st.tabs(["Medidas de risco", "Distribuições", "Capital IRB",
                "Contribuições", "Setores", "Migração & ECL",
                "Calibração", "Download"])

# --- Medidas ---
with tabs[0]:
    med = results["medidas"].copy()
    fmt = {c: "{:,.1f}" for c in med.columns if c not in ("modelo", "tempo_s")}
    fmt["tempo_s"] = "{:.2f}"
    st.dataframe(med.style.format(fmt, na_rep="—"), width='stretch')
    col_var = [c for c in med.columns if c.startswith("VaR 99.90")][0]
    dfp = med.dropna(subset=[col_var]).sort_values(col_var)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(dfp["modelo"], dfp[col_var] / total * 100, color="#1f4e79")
    ax.set_xlabel("VaR 99,90% (% da exposição líquida)")
    st.pyplot(fig)

# --- Distribuições ---
with tabs[1]:
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
    ax[0].set_title("Corpo (até o quantil 99,9%)"); ax[0].legend(fontsize=7)
    ax[1].set_title("Cauda P(L > x)"); ax[1].set_ylim(1e-5, 1)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.2, which="both")
    ax[1].set_xlabel("Perda (% exposição líquida)")
    st.pyplot(fig)

# --- IRB ---
with tabs[2]:
    irb_res = results["irb"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital IRB (99,90%)", f"{irb_res['capital_total_999']:,.1f}",
              f"{irb_res['capital_pct_exposicao']:.2%} da exposição", delta_color="off")
    c2.metric("Ajuste de granularidade", f"{irb_res['ajuste_granularidade']:,.1f}")
    c3.metric("Capital + GA", f"{irb_res['capital_com_ga']:,.1f}")
    c4.metric("RWA equivalente (12,5×)", f"{irb_res['rwa_equivalente']:,.0f}")
    df_irb = port.df[["id_contraparte", "exposicao", "pd", "lgd", "prazo_anos"]].copy()
    df_irb["K"] = irb_res["coeficiente_k"]
    df_irb["capital_IRB"] = irb_res["capital_por_contraparte"]
    st.dataframe(
        df_irb.sort_values("capital_IRB", ascending=False).head(50)
        .style.format({"exposicao": "{:,.1f}", "pd": "{:.4%}", "lgd": "{:.0%}",
                       "prazo_anos": "{:.1f}", "K": "{:.4f}", "capital_IRB": "{:,.2f}"}),
        width='stretch', height=350)

# --- Contribuições ---
with tabs[3]:
    if results.get("pooled") and results.get("contrib_bucket") is not None:
        cb = results["contrib_bucket"]
        st.caption(f"Contribuições ao ES {results['contrib_alpha']:.0%} por linha "
                   f"agregada ({results['modelo_contrib']}; ES total "
                   f"{results['es_total_contrib']:,.1f}). Grandes exposições têm "
                   f"n_contratos = 1; buckets GH×setor agregam o varejo pulverizado.")
        st.dataframe(cb.head(40).style.format(
            {"exposicao": "{:,.1f}", "pd": "{:.4%}", "contrib_ES": "{:,.2f}",
             "contrib_ES_pct": "{:.2%}", "n_contratos": "{:,.0f}"}),
            width='stretch', height=380)
    elif "contribuicoes" not in results:
        st.info("Contribuições não calculadas nesta execução.")
    else:
        cc = results["contribuicoes"]
        st.caption(f"Decomposição Monte Carlo do modelo threshold gaussiano "
                   f"(α = {cc['alpha']:.0%}; VaR médio {cc['var_medio']:,.1f}, "
                   f"ES médio {cc['es_medio']:,.1f})")
        df_c = port.df[["id_contraparte", "exposicao", "pd"]].copy()
        if "rating" in port.df.columns:
            df_c["rating"] = port.df["rating"]
        df_c["contrib_VaR"] = cc["var_contrib"]
        df_c["contrib_ES"] = cc["es_contrib"]
        df_c["contrib_ES_%"] = df_c["contrib_ES"] / cc["es_medio"]
        top = df_c.sort_values("contrib_ES", ascending=False)
        st.dataframe(top.head(30).style.format(
            {"exposicao": "{:,.1f}", "pd": "{:.4%}", "contrib_VaR": "{:,.2f}",
             "contrib_ES": "{:,.2f}", "contrib_ES_%": "{:.2%}"}),
            width='stretch', height=350)

# --- Setores (multifatorial) ---
with tabs[4]:
    if "multifator" not in results:
        st.info("Modelo multifatorial não executado (ative na barra lateral; "
                "a carteira precisa da coluna 'setor').")
    else:
        mf = results["multifator"]
        cal = mf["calibracao"]
        c1, c2, c3 = st.columns(3)
        c1.metric("ρ ativos intra-setor", f"{cal['rho_ativos_intra']:.2%}",
                  f"alvo default {cal['rho_default_intra_alvo']:.0%}", delta_color="off")
        c2.metric("ρ ativos inter-setor", f"{cal['rho_ativos_inter']:.2%}",
                  f"alvo default {cal['rho_default_inter_alvo']:.0%}", delta_color="off")
        c3.metric("HHI setorial", f"{mf['hhi_setorial']:.4f}",
                  f"{1/mf['hhi_setorial']:.1f} setores efetivos", delta_color="off")
        st.caption(f"Contribuições ao ES {mf['contrib_alpha']:.0%} do modelo "
                   f"gaussiano multifatorial (ES médio {mf['es_medio']:,.1f}). "
                   "A razão risco/exposição > 1 indica setor que concentra "
                   "proporcionalmente mais cauda do que carteira.")
        df_s = mf["contrib_por_setor"].copy()
        st.dataframe(df_s.style.format({
            "exposicao_liquida": "{:,.1f}", "perda_esperada": "{:,.1f}",
            "contrib_ES": "{:,.1f}", "exposicao_pct": "{:.1%}",
            "contrib_ES_pct": "{:.1%}", "razao_risco_exposicao": "{:.2f}"}),
            width='stretch')
        fig, ax = plt.subplots(figsize=(8, 3.6))
        x = np.arange(len(df_s))
        ax.bar(x - 0.2, df_s["exposicao_pct"] * 100, width=0.4,
               label="Exposição", color="#8a8a8a")
        ax.bar(x + 0.2, df_s["contrib_ES_pct"] * 100, width=0.4,
               label="Contribuição ao ES", color="#b02418")
        ax.set_xticks(x); ax.set_xticklabels(df_s["setor"], rotation=30, ha="right")
        ax.set_ylabel("% do total"); ax.legend(fontsize=8)
        st.pyplot(fig)

# --- Migração & ECL ---
with tabs[5]:
    if "mig" not in st.session_state:
        st.info("Carregue a matriz de transição na barra lateral (ou marque a "
                "matriz de exemplo) e execute a análise. A carteira precisa "
                "da coluna 'rating'.")
    else:
        mig = st.session_state["mig"]
        ecl_df, sim_m = mig["ecl"], mig["sim"]
        if sim_m is None:
            sim_m = {"T": 0, "M": 0, "block_size": 0, "EL": float("nan"),
                     "VaR": {"99.90%": float("nan")},
                     "ES": {"99.90%": float("nan")}, "perdas": np.array([0.0])}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ECL 12 meses", f"{ecl_df['ECL_12m'].sum():,.1f}")
        c2.metric("ECL lifetime", f"{ecl_df['ECL_lifetime'].sum():,.1f}",
                  f"{ecl_df['ECL_lifetime'].sum()/ecl_df['ECL_12m'].sum():.2f}x o 12m",
                  delta_color="off")
        if "provisao" in ecl_df:
            c3.metric("Estágio 2", f"{int((ecl_df['estagio']==2).sum())} nomes")
            c4.metric("Provisão por estágio", f"{ecl_df['provisao'].sum():,.1f}")
        st.caption(f"Multiperíodo ({sim_m['T']} anos, M={sim_m['M']:,}, blocos de "
                   f"{sim_m['block_size']:,}): EL {sim_m['EL']:,.1f} | "
                   f"VaR 99,9% {sim_m['VaR']['99.90%']:,.1f} | "
                   f"ES 99,9% {sim_m['ES']['99.90%']:,.1f}")
        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            anos = np.arange(1, mig["cum"].shape[1] + 1)
            for i, r in enumerate(mig["ratings"][:-1]):
                ax.semilogy(anos, mig["cum"][i] * 100, marker="o", ms=3, label=r)
            ax.set_xlabel("Anos"); ax.set_ylabel("PD acumulada (%)")
            ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25, which="both")
            st.pyplot(fig)
        with colB:
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            ax.hist(sim_m["perdas"], bins=80, density=True, color="#1f4e79")
            ax.set_xlabel("Perda descontada acumulada")
            ax.set_title(f"Distribuição multiperíodo ({sim_m['T']} anos)")
            st.pyplot(fig)
        st.dataframe(ecl_df.sort_values("ECL_lifetime", ascending=False)
            .head(50).style.format({"exposicao": "{:,.1f}", "pd": "{:.4%}",
                "lgd": "{:.0%}", "prazo_anos": "{:.1f}", "pd_matriz_1a": "{:.4%}",
                "ECL_12m": "{:,.2f}", "ECL_lifetime": "{:,.2f}",
                "razao_lifetime_12m": "{:.2f}", "provisao": "{:,.2f}"},
                na_rep="—"), width='stretch', height=330)

# --- Calibração ---
with tabs[6]:
    st.json({
        "parametros": results["parametros"],
        "beta_binomial": results.get("calibracao_beta", {}),
        "threshold_gaussiano": results.get("calibracao_gaussiana", {}),
        "multifatorial": results.get("multifator", {}).get("calibracao", {}),
    })

# --- Download ---
with tabs[7]:
    buf = io.BytesIO()
    results_to_excel(results, port, buf)
    st.download_button("⬇ Baixar resultados (Excel)", buf.getvalue(),
                       file_name="resultados_analise.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
