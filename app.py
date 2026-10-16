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


def _recarregar_engine():
    """O Streamlit recarrega o app.py a cada atualização do código, mas
    mantém em memória os módulos de engine/ já importados. Recarrega cada
    módulo cujo arquivo mudou desde a última carga (ou que ainda não foi
    marcado por esta versão do app), evitando chamar funções antigas."""
    import importlib, sys
    ordem = ["portfolio", "irb_bcb", "multifactor", "pooled", "aggregation",
             "migration", "report_paper", "report_tex", "analytics"]
    for nome in ordem:
        mod = sys.modules.get(f"engine.{nome}")
        arq = getattr(mod, "__file__", None)
        if mod is None or arq is None or not os.path.exists(arq):
            continue
        mt = os.path.getmtime(arq)
        if getattr(mod, "_creditlens_mtime", None) != mt:
            mod = importlib.reload(mod)
            mod._creditlens_mtime = mt


_recarregar_engine()

from engine.portfolio import load_portfolio
from engine.analytics import run_full_analysis, results_to_excel

st.set_page_config(page_title="CreditLens", page_icon="📊", layout="wide")


def _mi(v):
    """Entrada em reais; saída em R$ milhões, duas casas."""
    try:
        return f"{float(v) / 1e6:,.2f}"
    except (TypeError, ValueError):
        return "—"


def _FM(v):            # formatador de coluna monetária (Styler)
    return _mi(v) if v == v else "—"


st.title("CreditLens — Análise de Carteira de Crédito")
st.caption(
    "Motor quantitativo baseado em Bolder (2018), *Credit-Risk Modelling* (Springer): "
    "modelos independentes, mixture models (beta-binomial, CreditRisk+), threshold models "
    "(Gaussiano/t, ASRF), capital regulatório pelas abordagens IRB (Res. BCB 303/2023, com "
    "ajuste de granularidade) e padronizada (Res. BCB 229/2022), com o piso de 72,5%, e "
    "decomposição de contribuições de risco."
)

# ---------------- Sidebar: parâmetros ----------------
with st.sidebar:
    st.header("Parâmetros dos modelos")
    st.caption("Os dados de entrada ficam na área central (passo 1).")
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
    st.subheader("Contribuições de risco")
    run_contrib = st.checkbox("Calcular contribuições de VaR/ES", value=True)
    contrib_S = st.slider("Iterações da decomposição (S)", 5, 30, 10)
    contrib_alpha = st.selectbox("Nível α das contribuições", [0.95, 0.99, 0.999], index=1)
    seed = st.number_input("Semente aleatória", value=42, step=1)
    # entrada sempre em reais (R$); relatórios em R$ milhões
    _unid, escala_reais = "R$", 1.0

# ---------------- 1. Dados de entrada (só a escolha; nada é processado) ----
_TDIR = os.path.join(os.path.dirname(__file__), "templates")
_DEMO250 = os.path.join(os.path.dirname(__file__), "data", "carteira_teste.csv")
_MTX_DEMO = os.path.join(_TDIR, "matriz_migracao_modelo.csv")


def _baixar_modelo(arquivo, rotulo, key):
    st.download_button(rotulo, open(os.path.join(_TDIR, arquivo), "rb").read(),
                       file_name=arquivo, mime="text/csv", key=key,
                       type="tertiary", icon=":material/download:")


def _escolha(rotulo, opcoes, key, default):
    """Seletor segmentado que nunca fica vazio."""
    v = st.segmented_control(rotulo, opcoes, default=default, key=key,
                             selection_mode="single", width="stretch")
    return v if v is not None else default


st.subheader("1. Dados de entrada")
st.caption("Escolha a carteira e, se quiser, a matriz de transição. Nada é "
           "calculado até você clicar em **▶ Executar** (passo 3). Valores "
           "monetários em reais (R\\$); resultados em R\\$ milhões.")

# cartões de mesma largura e mesma altura (o menor estica até o maior)
st.markdown("""<style>
div[data-testid="stLayoutWrapper"]:has(> [class*="st-key-cartao_"]) {
    flex: 1 1 auto; display: flex; flex-direction: column; }
[class*="st-key-cartao_"] { flex: 1 1 auto; }
[class*="st-key-cartao_"] [data-testid="stButtonGroup"] button {
    flex: 1 1 0; min-width: 0; }   /* segmentos de mesma largura */
</style>""", unsafe_allow_html=True)
_cc, _cm = st.columns(2, gap="medium")

# --- cartão: carteira ---
_ORIG_ARQ, _ORIG_EX = "Enviar arquivo", "Usar exemplo"
_TIPO_CP, _TIPO_CT = "Por contraparte", "Por contrato"
with _cc, st.container(border=True, key="cartao_carteira"):
    st.markdown("##### :material/account_balance: Carteira de crédito")
    tipo_sel = _escolha("Granularidade da base", [_TIPO_CP, _TIPO_CT],
                        "tipo_base", _TIPO_CP)
    orig_sel = _escolha("Origem", [_ORIG_ARQ, _ORIG_EX], "orig_base",
                        _ORIG_ARQ)
    analitica = tipo_sel == _TIPO_CT
    if analitica:
        st.caption("Uma linha por contrato, até milhões de linhas. Ao "
                   "executar, as maiores exposições passam individualmente "
                   "e o restante é agregado em grupos homogêneos (GH × setor "
                   "× categoria IRB), preservando a perda esperada.")
    else:
        st.caption("Uma linha por contraparte, até alguns milhares de nomes; "
                   "os modelos rodam nome a nome.")

    fonte = None        # descreve a escolha; o conteúdo só é processado ao executar
    if orig_sel == _ORIG_ARQ:
        _up = st.file_uploader(
            "CSV da base analítica" if analitica else "CSV da carteira",
            type=["csv"], key="up_analitica" if analitica else "up_carteira")
        if _up is not None:
            fonte = {"tipo": "analitica" if analitica else "padrao",
                     "nome": _up.name, "conteudo": _up.getvalue()}
    else:
        if analitica:
            st.info("**Base demo** — 300 mil contratos simulados (R$ 5 bi em 8 setores; "
                    "varejo, PME e grandes empresas), gerada ao executar.",
                    icon=":material/dataset:")
            fonte = {"tipo": "analitica",
                     "nome": "base analítica demo (300 mil contratos)",
                     "conteudo": None}
        else:
            st.info("**Carteira de teste** — 250 contrapartes em 8 setores "
                    "(R$ 10 bi).", icon=":material/dataset:")
            fonte = {"tipo": "padrao",
                     "nome": "carteira de teste (250 contrapartes)",
                     "conteudo": None}
    if analitica:
        max_grandes = st.number_input(
            "Máximo de grandes exposições individuais", 0, 2000, 200, 50,
            help="As maiores exposições ficam fora da agregação, "
                 "preservando o risco de concentração de nomes.")
    else:
        max_grandes = 200
    if analitica:
        _baixar_modelo("base_analitica_modelo.csv",
                       "Baixar modelo de base analítica", "dl_mod_analitica")
    else:
        _baixar_modelo("carteira_modelo.csv", "Baixar modelo de carteira",
                       "dl_mod_carteira")

# --- cartão: matriz de transição ---
with _cm, st.container(border=True, key="cartao_matriz"):
    st.markdown("##### :material/swap_vert: Migração e ECL *(opcional)*")
    mtx_sel = st.segmented_control(
        "Matriz de transição anual", [_ORIG_ARQ, _ORIG_EX], default=None,
        key="orig_mtx", selection_mode="single", width="stretch",
        help="Opcional. Clique de novo na opção escolhida para desmarcar.")
    mtx_up, usar_mtx_demo = None, False
    if mtx_sel == _ORIG_ARQ:
        mtx_up = st.file_uploader("CSV da matriz de transição", type=["csv"],
                                  key="mtx")
    elif mtx_sel == _ORIG_EX:
        usar_mtx_demo = True
        st.caption(":material/grid_on: Matriz de exemplo com 8 graus "
                   "(AAA a C) e o estado de default.")
    if mtx_sel is None:
        st.caption("Com a matriz, o CreditLens calcula a estrutura a termo "
                   "de PD, o ECL de 12 meses e *lifetime* (Res. CMN 4.966) "
                   "e o capital multiperíodo. Exige a coluna `rating`.")
        horizonte, taxa_desc, block_sz = 5, 0.10, 10_000
    else:
        h1, h2 = st.columns(2)
        horizonte = h1.slider("Horizonte (anos)", 1, 10, 5)
        taxa_desc = h2.slider("Desconto (a.a.)", 0, 30, 10, 1,
                              format="%d%%") / 100
        block_sz = st.select_slider(
            "Bloco de simulação (memória)",
            options=[2_000, 5_000, 10_000, 25_000], value=10_000,
            help="Cenários simulados por bloco no multiperíodo; blocos "
                 "menores usam menos memória.")
    _baixar_modelo("matriz_migracao_modelo.csv",
                   "Baixar modelo de matriz", "dl_mod_matriz")

with st.expander("Layouts dos arquivos de entrada — dicionário de "
                 "colunas e exemplos preenchidos", icon=":material/description:"):
    from engine.layouts import dicionario, dicionario_analitica, CARTEIRA, MATRIZ
    st.caption("Convenções: CSV separado por vírgula, decimal com ponto, "
               "UTF-8. Valores monetários **em reais (R$)**, sem escala — "
               "exceto colunas terminadas em `_mi` (R$ milhões). Taxas e "
               "frações em decimal (5% = 0.05). Os resultados são "
               "apresentados em R$ milhões.")
    _lt1, _lt2, _lt3 = st.tabs(["Carteira (por contraparte)",
                                "Base analítica (por contrato)",
                                "Matriz de transição anual"])
    with _lt1:
        st.markdown("Uma linha por **contraparte**. Para carteiras de até "
                    "alguns milhares de nomes; os modelos rodam nome a nome. "
                    "O exemplo cobre atacado, PME, instituição financeira, "
                    "varejo (imobiliário, rotativo e demais) e um ativo "
                    "problemático.")
        _f = os.path.join(_TDIR, "carteira_modelo.csv")
        st.download_button("⬇ Baixar modelo de carteira", open(_f, "rb").read(),
                           file_name="carteira_modelo.csv", mime="text/csv")
        st.dataframe(pd.read_csv(_f), hide_index=True, width='stretch')
        st.markdown("**Dicionário de colunas**")
        st.dataframe(dicionario(CARTEIRA), hide_index=True, width='stretch')
    with _lt2:
        st.markdown("Uma linha por **contrato**, para bases grandes (até "
                    "milhões de linhas). Ao executar, as maiores exposições "
                    "passam individualmente e o restante é agregado em grupos "
                    "homogêneos (GH × setor × categoria IRB), preservando a "
                    "perda esperada. O mesmo cliente pode ter vários "
                    "contratos — repita o `id_contraparte`. O modelo traz dois "
                    "clientes com mais de um contrato.")
        _f = os.path.join(_TDIR, "base_analitica_modelo.csv")
        st.download_button("⬇ Baixar modelo de base analítica",
                           open(_f, "rb").read(),
                           file_name="base_analitica_modelo.csv", mime="text/csv")
        st.dataframe(pd.read_csv(_f), hide_index=True, width='stretch')
        st.markdown("**Dicionário de colunas** (as demais colunas seguem o "
                    "layout da carteira)")
        st.dataframe(dicionario_analitica(), hide_index=True, width='stretch')
    with _lt3:
        st.markdown(
            "Probabilidades de migração **em um ano**: a linha é o rating de "
            "origem, a coluna o rating de destino. Ordene do melhor ao pior "
            "rating e termine com o estado de **default**, que é absorvente. "
            "Os rótulos precisam ser os mesmos da coluna `rating` da carteira. "
            "Usada para a estrutura a termo de PD (PDs acumuladas por "
            "potências da matriz), o ECL de 12 meses e *lifetime* e o "
            "capital multiperíodo.")
        _f = os.path.join(_TDIR, "matriz_migracao_modelo.csv")
        st.download_button("⬇ Baixar modelo de matriz de transição",
                           open(_f, "rb").read(),
                           file_name="matriz_migracao_modelo.csv", mime="text/csv")
        _mx = pd.read_csv(_f, index_col=0)
        st.dataframe(_mx.style.format("{:.4f}").background_gradient(
            cmap="Blues", axis=None), width='stretch')
        st.caption("Leitura: um cliente BBB hoje tem 88,5% de chance de "
                   "continuar BBB em um ano, 5,0% de cair para BB e 0,5% de "
                   "entrar em default. A última coluna é a PD de 12 meses do "
                   "grau.")
        st.markdown("**Regras do layout**")
        st.dataframe(dicionario(MATRIZ), hide_index=True, width='stretch')


@st.cache_data(show_spinner=False, max_entries=4)
def _ler_base(tipo, nome, conteudo):
    """Lê a base escolhida (CSV enviado, carteira de teste ou base demo)."""
    if conteudo is not None:
        return pd.read_csv(io.BytesIO(conteudo))
    if tipo == "analitica":
        from gerar_base_analitica import gerar
        return gerar(path=None)
    return pd.read_csv(_DEMO250)


@st.cache_data(show_spinner=False, max_entries=4)
def _resumo_setores(tipo, nome, conteudo):
    """Prévia leve: linhas, colunas e totais por setor (exposição líquida e
    PD média ponderada) — o que a leitura das correlações precisa."""
    df = _ler_base(tipo, nome, conteudo)
    out = {"linhas": len(df), "colunas": list(df.columns), "setores": None}
    if {"setor", "exposicao", "pd"} <= set(df.columns):
        lgd = df["lgd"] if "lgd" in df.columns else 1.0
        c = pd.to_numeric(df["exposicao"], errors="coerce") * lgd
        g = pd.DataFrame({"setor": df["setor"].astype(str), "c": c,
                          "cp": c * pd.to_numeric(df["pd"], errors="coerce")}
                         ).groupby("setor").sum()
        g["p"] = g["cp"] / g["c"]
        out["setores"] = g[["c", "p"]].reset_index()
    return out


@st.cache_data(show_spinner=False, max_entries=4)
def _preparar_carteira(tipo, nome, conteudo, max_grandes, escala_reais=1.0):
    """Carga, validação e (base analítica) agregação — feitas ao executar."""
    if tipo == "analitica":
        df = _ler_base(tipo, nome, conteudo)
        from engine.aggregation import aggregate_portfolio
        df_agg, rel = aggregate_portfolio(df, corte_grandes=None,
                                          max_grandes=int(max_grandes),
                                          escala_reais=escala_reais)
        buf = io.StringIO(); df_agg.to_csv(buf, index=False); buf.seek(0)
        return load_portfolio(buf), rel, True
    port = load_portfolio(io.BytesIO(conteudo) if conteudo is not None
                          else _DEMO250)
    pooled = "n_contratos" in port.df.columns and \
        bool((port.df["n_contratos"] > 1).any())
    return port, None, pooled


@st.cache_data(show_spinner=False, max_entries=4)
def _descrever_base(tipo, nome, conteudo):
    """Análise descritiva da base de entrada original (apêndice do paper)."""
    from engine.descritiva import descrever
    return descrever(_ler_base(tipo, nome, conteudo))


if fonte is None:
    st.info("Envie o CSV da carteira ou escolha **Usar exemplo** para "
            "começar.", icon=":material/upload_file:")
    st.subheader("3. Execute")
    st.caption("O botão é habilitado quando a carteira estiver escolhida "
               "no passo 1.")
    st.button("▶ Executar", type="primary", disabled=True,
              key="executar_inativo")
    st.stop()

try:
    with st.spinner("Lendo a base..."):
        previa = _resumo_setores(fonte["tipo"], fonte["nome"], fonte["conteudo"])
except Exception as e:                       # CSV ilegível
    st.error(f"Não foi possível ler a base: {e}")
    st.stop()
st.success(
    f"Base selecionada: **{fonte['nome']}** — {previa['linhas']:,} linhas"
    + (" (será agregada ao executar)" if fonte["tipo"] == "analitica" else "")
    + (f"; {len(previa['setores'])} setores" if previa["setores"] is not None
       else "; sem coluna `setor` (modelo multifatorial indisponível)") + ".")


# ---------------- 2. Correlação por setor (multifatorial) ----------------
def _mostrar_leitura_setores(cal, rho_intra, rho_inter):
    """Leitura do que os alvos por setor significam, a partir da calibração
    feita na execução (nada é calculado antes de Executar)."""
    from engine.multifactor import calibrate_rho, leitura_setores
    pbar = cal["pbar"]
    ps, pares, res = leitura_setores(
        cal, rho_intra, rho_inter,
        calibrate_rho(pbar, rho_intra), calibrate_rho(pbar, rho_inter))
    st.markdown("##### O que estas escolhas significam")
    st.markdown(
        "- **ρ intra** responde: *se um devedor do setor quebra, quanto "
        "aumenta a chance de outro do mesmo setor quebrar?*\n"
        "- **ρ inter** responde: *quanto o setor quebra junto com a economia "
        "— e, por ela, com os outros setores?*\n"
        "- A **fração global** (ρ ativos inter ÷ ρ ativos intra) diz de onde "
        "vem a dependência: perto de 1, o setor é **cíclico** (move-se com a "
        "economia e contagia os demais); perto de 0, é **concentrado** (um "
        "choque próprio derruba muitos nomes juntos, mas pouco se espalha).")
    _q = ps[["setor", "pd_media", "rho_default_intra_alvo",
             "rho_default_inter_alvo", "fracao_global", "perfil",
             "leitura"]].rename(columns={
        "pd_media": "PD média", "rho_default_intra_alvo": "ρ intra",
        "rho_default_inter_alvo": "ρ inter",
        "fracao_global": "fração global"})
    st.dataframe(_q.style.format({"PD média": "{:.3%}", "ρ intra": "{:.3%}",
                                  "ρ inter": "{:.3%}",
                                  "fração global": "{:.3f}"}),
                 width='stretch', hide_index=True)
    (v1, a1, b1), (v2, a2, b2) = pares[0], pares[1]
    vz, az, bz = pares[-1]
    st.markdown(
        f"**Pares de setores resultantes** (correlação de default): os mais "
        f"ligados são **{a1}–{b1}** ({v1:.3%}) e **{a2}–{b2}** ({v2:.3%}); o "
        f"menos ligado, **{az}–{bz}** ({vz:.3%}). Os pares não são digitados: "
        "saem da carga de cada setor no fator global.")
    dir_txt = {"acima": "**acima** do padrão uniforme — espere VaR e ES "
                        "multifatoriais maiores",
               "abaixo": "**abaixo** do padrão uniforme — espere VaR e ES "
                         "multifatoriais menores",
               "em linha com": "**em linha** com o padrão uniforme"}
    st.markdown(
        f"**Efeito esperado no capital:** correlações de ativos médias "
        f"(ponderadas pela exposição) de {res['rho_s_medio']:.3f} intra e "
        f"{res['rho_g_medio']:.3f} inter, contra {res['rho_s_ref']:.3f} e "
        f"{res['rho_g_ref']:.3f} no uniforme — "
        f"{dir_txt[res['direcao']]}. Setores grandes pesam mais nessa média.")
    with st.expander("Matriz de correlação de default entre setores"):
        st.dataframe(cal["matriz_default"].style.format("{:.3%}")
                     .background_gradient(cmap="Reds", axis=None),
                     width='stretch')
    st.caption("Os alvos são hipóteses do usuário. Num uso formal, cada ρ "
               "precisa de fonte (séries de inadimplência por setor ou "
               "estudo de referência); o paper registra os valores e "
               "verifica a calibração, mas não justifica a escolha.")


rho_setor = None
setor_invalido = False
if run_mf and previa["setores"] is not None:
    _tot = previa["setores"]
    st.subheader("2. Correlação por setor (opcional)")
    usar_setor = st.checkbox(
        "Calibrar correlações por setor", value=False,
        help="Cada setor recebe os próprios alvos de correlação de "
             "default, calibrados na PD média do setor. Desmarcado, "
             "todos os setores usam os alvos da barra lateral, "
             "calibrados na PD média da carteira.")
    if usar_setor:
        st.caption(
            "ρ intra: correlação de default entre dois devedores do setor. "
            "ρ inter: correlação de default do setor com os demais via fator "
            "global. Entre os setores s e t, a correlação de ativos é "
            "√(ρ_G,s·ρ_G,t): pares diferem pela carga de cada setor, mas "
            "blocos de setores (ex.: commodities × consumo) exigiriam um "
            "segundo fator global. Ajuste os alvos da barra lateral antes de "
            "editar: eles preenchem a tabela.")
        _sets = _tot["setor"].tolist()
        _base = pd.DataFrame({"setor": _sets,
                              "rho_intra": [float(rho_intra)] * len(_sets),
                              "rho_inter": [float(rho_inter)] * len(_sets)})
        _ed = st.data_editor(
            _base, key="rho_setor_" + "|".join(_sets), hide_index=True,
            disabled=["setor"], width='stretch',
            column_config={
                "rho_intra": st.column_config.NumberColumn(
                    "ρ default intra", min_value=0.001, max_value=0.5,
                    step=0.005, format="%.3f"),
                "rho_inter": st.column_config.NumberColumn(
                    "ρ default inter", min_value=0.001, max_value=0.5,
                    step=0.005, format="%.3f")})
        _bad = _ed[_ed["rho_inter"] > _ed["rho_intra"]]
        if len(_bad):
            setor_invalido = True
            st.error("O ρ inter não pode exceder o intra nos setores: "
                     + ", ".join(_bad["setor"]) + ".")
        else:
            rho_setor = _ed
            st.caption("A leitura destas escolhas — perfil de cada setor, "
                       "pares mais e menos ligados e efeito esperado no "
                       "capital — é gerada ao executar e aparece na aba "
                       "**Setores**.")


# ---------------- 3. Execução: o pipeline inteiro, com a configuração atual ---
def _config_atual():
    """Tudo o que define uma execução — base e parâmetros."""
    import hashlib
    h = hashlib.md5(fonte["conteudo"]).hexdigest() if fonte["conteudo"] else ""
    return {
        "base": (fonte["tipo"], fonte["nome"], h,
                 int(max_grandes) if fonte["tipo"] == "analitica" else None),
        "M": M, "rho_target": rho_target, "nu": nu, "crp": (crp_w, crp_v),
        "mf": (run_mf, rho_intra, rho_inter),
        "rho_setor": None if rho_setor is None else
        rho_setor.round(6).to_dict(orient="records"),
        "mig": (getattr(mtx_up, "name", None),
                hashlib.md5(mtx_up.getvalue()).hexdigest() if mtx_up else None,
                usar_mtx_demo, horizonte, taxa_desc, block_sz),
        "contrib": (run_contrib, contrib_S, contrib_alpha), "seed": int(seed),
        "escala_reais": escala_reais,
    }


cfg_atual = _config_atual()
st.subheader("3. Execute")
_bloqueio = (run_mf and rho_inter > rho_intra) or setor_invalido
if _bloqueio:
    st.caption("Corrija as correlações sinalizadas para executar.")
if st.button("▶ Executar", type="primary", disabled=_bloqueio):
    prog = st.progress(0.0, text="Preparando a carteira...")
    steps = {"n": 0}

    def tick(msg):
        steps["n"] += 1
        prog.progress(min(steps["n"] / 14, 0.99), text=msg)

    try:
        with st.spinner("Carregando e preparando a carteira..."):
            port, rel_agg, pooled_mode = _preparar_carteira(
                fonte["tipo"], fonte["nome"], fonte["conteudo"],
                int(max_grandes), escala_reais)
    except ValueError as e:
        prog.empty()
        st.error(f"Base inválida: {e}")
        st.stop()
    tick("Carteira pronta")
    with st.spinner("Descrevendo a base de entrada..."):
        descritiva = _descrever_base(fonte["tipo"], fonte["nome"],
                                     fonte["conteudo"])

    with st.spinner("Rodando modelos..."):
        if pooled_mode:
            from engine.analytics import run_pooled_analysis
            results = run_pooled_analysis(
                port, M=M, rho_target=rho_target,
                rho_intra=rho_intra, rho_inter=rho_inter, nu=float(nu),
                crp_w=crp_w, crp_v=crp_v,
                contrib_alpha=contrib_alpha, seed=int(seed), progress=tick,
                rho_setor=rho_setor, escala_reais=escala_reais,
                rel_agg=rel_agg)
        else:
            results = run_full_analysis(
                port, M=M, rho_target=rho_target, nu=float(nu),
                crp_w=crp_w, crp_v=crp_v,
                run_contributions=run_contrib, contrib_S=contrib_S,
                contrib_alpha=contrib_alpha, seed=int(seed), progress=tick,
                run_multifactor=run_mf and rho_inter <= rho_intra
                    and "setor" in port.df.columns,
                rho_intra=rho_intra, rho_inter=rho_inter,
                rho_setor=rho_setor, escala_reais=escala_reais,
            )
    # --- migração & ECL ---
    mtx_src = mtx_up if mtx_up is not None else (
        os.path.join(os.path.dirname(__file__), "templates",
                     "matriz_migracao_modelo.csv") if usar_mtx_demo else None)
    st.session_state.pop("mig", None)
    if mtx_src is not None and "rating" in port.df.columns:
        try:
            from engine.migration import (load_transition_matrix, compute_ecl,
                                          simulate_multiperiod,
                                          simulate_multiperiod_pooled)
            from engine.multifactor import rhos_para_motor
            if hasattr(mtx_src, "seek"):
                mtx_src.seek(0)
            P_m, ratings_m = load_transition_matrix(mtx_src)
            ecl_df, cum_m, _ = compute_ecl(port, P_m, ratings_m,
                                           discount_rate=taxa_desc)
            cal_mf = results.get("multifator", {}).get("calibracao")
            if cal_mf:
                _, sector_idx = np.unique(port.df["setor"].astype(str),
                                          return_inverse=True)
                kw = dict(rho_g=rhos_para_motor(cal_mf)[0],
                          rho_s=rhos_para_motor(cal_mf)[1],
                          sector_idx=sector_idx)
            else:
                kw = dict(rho_g=results["calibracao_gaussiana"]["rho_ativos"])
            if pooled_mode:
                sim_m = simulate_multiperiod_pooled(
                    port, P_m, ratings_m, M=M, T=horizonte,
                    discount_rate=taxa_desc, block_size=block_sz,
                    seed=int(seed), progress=tick, rho_d_alvo=rho_target, **kw)
            else:
                sim_m = simulate_multiperiod(
                    port, P_m, ratings_m, M=M, T=horizonte,
                    discount_rate=taxa_desc, block_size=block_sz,
                    seed=int(seed), progress=tick, **kw)
            st.session_state["mig"] = {"ecl": ecl_df, "cum": cum_m,
                                       "ratings": ratings_m, "sim": sim_m}
        except ValueError as e:
            st.error(f"Migração/ECL: {e}")
    prog.progress(1.0, text="Concluído")
    import datetime as _dt
    st.session_state.update({
        "results": results, "port": port, "rel_agg": rel_agg,
        "pooled_mode": pooled_mode, "cfg": cfg_atual,
        "executado_em": _dt.datetime.now().strftime("%H:%M:%S"),
        "params_ui": {"contrib_alpha": contrib_alpha, "contrib_S": contrib_S,
                      "unidade": _unid,
                      "max_grandes": int(max_grandes), "horizonte": horizonte,
                      "taxa_desc": taxa_desc},
        "nome_base": fonte["nome"],
        "descritiva": descritiva,
    })

if "results" not in st.session_state:
    st.stop()

# ---------------- 4. Resultados da última execução ----------------
if st.session_state.get("cfg") != cfg_atual:
    st.warning(
        f"A configuração mudou desde a última execução "
        f"({st.session_state.get('executado_em', '')}). Clique em "
        "**▶ Executar** para rodar o pipeline com a configuração atual.")
    if not st.toggle("Mostrar os resultados da última execução", value=False):
        st.stop()

port = st.session_state["port"]
rel_agg = st.session_state.get("rel_agg")
st.subheader("4. Resultados")
st.caption(f"Execução de {st.session_state.get('executado_em', '')} — base "
           f"{st.session_state.get('nome_base', '')}.")
for w in port.warnings:
    st.warning(w)
if rel_agg is not None:
    rel = rel_agg
    st.success(f"Agregação: {rel['linhas_analitica']:,} contratos → "
               f"{rel['linhas_agregada']} linhas "
               f"({rel['n_grandes']} grandes exposições + "
               f"{rel['n_buckets']} buckets GH×setor).")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("EL analítica (R$ mi)", _mi(rel['el_analitica']))
    q2.metric("EL agregada (R$ mi)", _mi(rel['el_agregada']),
              f"dif. {rel['el_diferenca_pct']:.4%}", delta_color="off")
    q3.metric("Exposição nos grandes", f"{rel['exposicao_grandes_pct']:.1%}")
    q4.metric("CV de PD máx. no bucket", f"{rel['pd_cv_max_bucket']:.2f}",
              f"{rel['buckets_cv_alto']} buckets com CV>0,5", delta_color="off")
    if rel['buckets_cv_alto'] > 0:
        st.warning("Há buckets com dispersão alta de PD — considere refinar "
                   "os grupos homogêneos (coluna 'gh') para reduzir o viés "
                   "de Jensen nas funções de capital.")
s = port.summary()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Contrapartes", f"{s['n_contrapartes']:,}")
c2.metric("EAD total (R$ mi)", _mi(s['ead_total']))
c3.metric("Perda esperada (R$ mi)", _mi(s['perda_esperada']),
          f"{s['perda_esperada_pct']:.2%} da exposição", delta_color="off")
c4.metric("PD média ponderada", f"{s['pd_media_ponderada']:.2%}")
c5.metric("HHI (N efetivo)", f"{s['hhi']:.4f}", f"{s['n_efetivo']:.0f} nomes",
          delta_color="off")
with st.expander("Ver carteira executada"):
    st.dataframe(port.df, width='stretch', height=280)

_desc = st.session_state.get("descritiva")
if _desc:
    with st.expander("Análise descritiva da base de entrada "
                     "(apêndice do paper de validação)"):
        _g, _cc = _desc["geral"], _desc["concentracao"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Linhas", f"{_g['linhas']:,}",
                  f"{_g['contrapartes']:,} contrapartes", delta_color="off")
        d2.metric("Exposição total (R$ mi)", _mi(_g['ead_total']))
        d3.metric("10 maiores exposições", f"{_cc['top10']:.3%}",
                  f"Gini {_cc['gini']:.3f}", delta_color="off")
        d4.metric("Nº efetivo de exposições", f"{_cc['n_efetivo']:,.3f}")
        if _desc.get("ausentes"):
            st.caption("Colunas opcionais ausentes: "
                       + ", ".join(f"`{a}`" for a in _desc["ausentes"]) + ".")
        st.markdown("**Completude e domínio das colunas**")
        st.dataframe(_desc["qualidade"].rename(columns={
            "coluna": "Coluna", "tipo": "Tipo",
            "preenchimento": "Preenchimento",
            "nao_preenchidos": "Não preenchidos",
            "observacao": "Observação"}).style.format(
            {"Preenchimento": "{:.3%}", "Não preenchidos": "{:,}"}),
            width='stretch', hide_index=True)
        st.markdown("**Variáveis de risco**")
        _num = _desc["numericas"].copy()
        _cols_q = ["media", "desvio", "min", "p01", "p25", "mediana", "p75",
                   "p99", "max"]
        _ex = _num["variavel"] == "Exposição (EAD)"
        _num.loc[_ex, _cols_q] = _num.loc[_ex, _cols_q] / 1e3
        _num.loc[_ex, "variavel"] = "Exposição por contrato (R$ mil)"
        st.dataframe(_num.style.format({c: "{:,.3f}" for c in _cols_q}
                                       | {"n": "{:,}"}),
                     width='stretch', hide_index=True)
        for _c, (_rot, _t) in _desc["categoricas"].items():
            st.markdown(f"**Composição por {_rot.lower() if _rot != 'Categoria IRB' else _rot}**")
            st.dataframe(_t.style.format({
                "linhas": "{:,}", "ead": _FM, "el": _FM,
                "ead_pct": "{:.3%}", "pd_media": "{:.3%}",
                "lgd_media": "{:.3%}", "el_pct": "{:.3%}"}),
                width='stretch', hide_index=True)

results = st.session_state["results"]
total = results["resumo"]["exposicao_liquida_total"]

tabs = st.tabs(["Medidas de risco", "Distribuições", "Capital regulatório",
                "Contribuições", "Setores", "Migração & ECL",
                "Calibração", "Download"],
            key="aba_resultados", on_change="rerun")   # mantém a aba ativa

# --- Medidas ---
with tabs[0]:
    med = results["medidas"].copy()
    st.caption("Valores em R$ milhões (base de entrada em reais).")
    fmt = {c: _FM for c in med.columns if c not in ("modelo", "tempo_s")}
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
    def _rotulo(n):
        for suf in [" (pooled)", " (setores)", " (setores, nu=10)",
                    " (pooled, nu=10)", " (nu=10)"]:
            n = n.replace(suf, "")
        return n.replace(" multifatorial", " MF")

    PAL = {"Binomial independente": "#9aa0a6",
           "Beta-binomial": "#6a5acd",
           "CreditRisk+": "#0e8a8a",
           "Threshold Gaussiano 1F": "#1f4e79",
           "Threshold t-Student 1F": "#b02418",
           "Threshold Gaussiano MF": "#2e8b57",
           "Threshold t-Student MF": "#e07b00"}
    dists = {k: np.asarray(v) / 1e6 for k, v in results["distribuicoes"].items()}
    xmax = max(np.quantile(d, 0.999) for d in dists.values())
    bins = np.linspace(0, xmax, 90)

    # escala do corpo governada pelos modelos COM dependência: o binomial
    # independente (quase degenerado pela lei dos grandes números) é
    # tracejado e pode estourar o topo -- com anotação do pico.
    picos = {}
    for nome, dist in dists.items():
        h, _ = np.histogram(dist, bins=bins, density=True)
        picos[nome] = float(h.max())
    dep = [v for k, v in picos.items() if "independente" not in k.lower()]
    ylim = (max(dep) if dep else max(picos.values())) * 1.25

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for nome, dist in dists.items():
        rot = _rotulo(nome)
        indep = "independente" in nome.lower()
        cor = PAL.get(rot, "#555")
        ax[0].hist(dist, bins=bins, density=True, histtype="step",
                   label=rot, color=cor, lw=1.1 if indep else 1.6,
                   ls="--" if indep else "-")
        srt = np.sort(dist)
        tail = 1.0 - np.arange(1, len(srt) + 1) / len(srt)
        ax[1].semilogy(srt * 1e6 / total * 100, np.maximum(tail, 1e-6),
                       label=rot, color=cor, lw=1.1 if indep else 1.6,
                       ls="--" if indep else "-")
    ax[0].set_xlim(0, xmax)
    ax[0].set_ylim(0, ylim)
    for nome, pico in picos.items():
        if "independente" in nome.lower() and pico > ylim:
            el_x = float(np.mean(dists[nome]))
            ax[0].annotate("Binomial indep.: pico fora da escala\n"
                           "(risco idiossincrático eliminado pela LLN)",
                           xy=(el_x, ylim),
                           xytext=(el_x + 0.18 * xmax, ylim * 0.72),
                           fontsize=7, color="#555",
                           arrowprops=dict(arrowstyle="->",
                                           color="#9aa0a6", lw=1.0))
    ax[0].set_title("Corpo (até 99,9%) — escala dos modelos com dependência",
                    fontsize=10)
    ax[0].legend(fontsize=7)
    ax[0].set_xlabel("Perda (R$ milhões)")
    ax[1].set_title("Cauda P(L > x)"); ax[1].set_ylim(1e-5, 1)
    ax[1].axhline(1e-3, color="k", ls=":", lw=0.8)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.2, which="both")
    ax[1].set_xlabel("Perda (% exposição líquida)")
    st.pyplot(fig)

# --- IRB ---
with tabs[2]:
    irb_res = results["irb"]
    cp = results.get("padronizado")
    if cp:
        ps_ = cp["piso"]
        st.markdown("##### Requerimento de capital — visão consolidada")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("RWA IRB (modelos internos, R$ mi)", _mi(ps_['rwa_irb']))
        k2.metric("RWA padronizado (RWACPAD, R$ mi)", _mi(ps_['rwa_cpad']),
                  f"IRB/padronizado {ps_['razao_irb_cpad']:.3f}",
                  delta_color="off")
        k3.metric("Piso: 72,5% × RWACPAD (R$ mi)", _mi(ps_['piso']),
                  "vinculante" if ps_["vinculante"] else "não vinculante",
                  delta_color="inverse" if ps_["vinculante"] else "off")
        k4.metric("RWA final (art. 34, Res. BCB 303; R$ mi)",
                  _mi(ps_['rwa_final']),
                  f"capital R$ {_mi(ps_['capital_final'])} mi (8%)",
                  delta_color="off")
        st.caption("RWA final = max(RWA IRB; 72,5% × RWACPAD), calculados "
                   "sobre as mesmas exposições da carteira ativa. Quando o "
                   "piso é vinculante, o capital é determinado pela "
                   "abordagem padronizada, não pelos modelos internos.")
        st.divider()
    st.markdown("##### Abordagem IRB (modelos internos) — Res. BCB 303/2023")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital IRB (99,90%; R$ mi)", _mi(irb_res['capital_total_999']),
              f"{irb_res['capital_pct_exposicao']:.2%} da exposição", delta_color="off")
    c2.metric("Ajuste de granularidade (R$ mi)", _mi(irb_res['ajuste_granularidade']))
    c3.metric("Capital + GA (R$ mi)", _mi(irb_res['capital_com_ga']))
    c4.metric("RWA equivalente (12,5×; R$ mi)", _mi(irb_res['rwa_equivalente']))
    st.caption("Função de capital da Resolução BCB nº 303/2023: correlação por "
               "categoria (atacado, SME, HVCRE, IF, varejo residencial, rotativo "
               "qualificado e demais varejo), ajuste de maturidade só no atacado, "
               "pisos de PD de 0,05% (0,10% no rotativo).")
    seg_tab = irb_res.get("por_segmento")
    if seg_tab is not None and len(seg_tab):
        if not irb_res.get("segmentado"):
            st.warning("A carteira não traz a coluna `segmento_irb`: todas as "
                       "exposições foram tratadas como atacado (corporate). Em "
                       "carteiras de varejo isso superestima o capital.")
        if irb_res.get("sme_sem_receita"):
            st.warning(f"{irb_res['sme_sem_receita']} exposição(ões) SME sem "
                       "`receita_bruta_mi`: aplicada correlação sem redução de "
                       "porte (conservador).")
        if irb_res.get("n_pd_piso"):
            st.info(f"{irb_res['n_pd_piso']} PD(s) elevada(s) ao piso "
                    "regulatório (art. 61).")
        st.markdown("**Capital por categoria IRB**")
        st.dataframe(seg_tab.drop(columns=["segmento"]).style.format({
            "ead": _FM, "exposicao_liquida": _FM,
            "R_ponderado": "{:.3%}", "capital": _FM,
            "capital_pct": "{:.3%}", "rwa": _FM}),
            width='stretch', hide_index=True)
    cols_irb = ["id_contraparte", "exposicao", "pd", "lgd", "prazo_anos"]
    cols_irb += [c for c in ["segmento_irb"] if c in port.df.columns]
    df_irb = port.df[cols_irb].copy()
    df_irb["K"] = irb_res["coeficiente_k"]
    df_irb["capital_IRB"] = irb_res["capital_por_contraparte"]
    st.dataframe(
        df_irb.sort_values("capital_IRB", ascending=False).head(50)
        .style.format({"exposicao": _FM, "pd": "{:.4%}", "lgd": "{:.0%}",
                       "prazo_anos": "{:.1f}", "K": "{:.4f}", "capital_IRB": _FM}),
        width='stretch', height=350)

    if cp:
        st.divider()
        st.markdown("##### Abordagem padronizada (RWACPAD) — Res. BCB 229/2022")
        q1, q2, q3 = st.columns(3)
        q1.metric("RWACPAD (R$ mi)", _mi(cp['rwa_total']))
        q2.metric("Capital padronizado (8%; R$ mi)", _mi(cp['capital']))
        q3.metric("FPR médio", f"{cp['fpr_medio']:.3%}")
        st.caption("RWA = EAD × FPR, com o fator de ponderação de risco de "
                   "cada exposição pela sua categoria (artigos da Res. BCB "
                   "229). A LGD não entra; na base analítica, o FPR é "
                   "atribuído contrato a contrato antes da agregação.")
        av = cp.get("avisos") or {}
        _msgs = {
            "residencial_sem_ltv": "exposição(ões) imobiliária(s) residencial(is) "
                                   "sem `ltv`: aplicada a faixa acima de 100% "
                                   "(conservador)",
            "pj_sem_receita": "pessoa(s) jurídica(s) sem `receita_bruta_mi`: "
                              "FPR de 100% (art. 41)",
            "pme_sem_ativo_total": "PME(s) sem `ativo_total_mi`: o critério de "
                                   "ativo do art. 36 foi presumido atendido",
            "varejo_reprovado_limites": "exposição(ões) de varejo fora dos "
                                        "limites do art. 46 (R$ 5 mi ou 0,2%)",
            "if_categoria_padrao_A": "exposição(ões) a IF sem `categoria_if`: "
                                     "presumida categoria A",
            "ativos_problematicos": "ativo(s) problemático(s) ponderado(s) pelo "
                                    "art. 66",
            "fpr_informado": "exposição(ões) com FPR informado pelo usuário",
        }
        for k, txt in _msgs.items():
            if av.get(k):
                st.info(f"{av[k]:,} {txt}.")
        if cp.get("por_artigo") is not None:
            st.markdown("**RWACPAD por fator de ponderação**")
            st.dataframe(cp["por_artigo"][["descricao", "linhas", "ead", "rwa",
                                           "fpr_medio"]].style.format({
                "linhas": "{:,}", "ead": _FM, "rwa": _FM,
                "fpr_medio": "{:.3%}"}), width='stretch', hide_index=True)
        if cp.get("comparacao") is not None:
            st.markdown("**IRB × padronizado por categoria**")
            comp = cp["comparacao"].copy()
            cols_c = [c for c in ["descricao", "segmento", "ead", "rwa_irb",
                                  "rwa_cpad", "razao_irb_cpad"]
                      if c in comp.columns]
            st.dataframe(comp[cols_c].style.format({
                "ead": _FM, "rwa_irb": _FM, "rwa_cpad": _FM,
                "razao_irb_cpad": "{:.3f}"}), width='stretch', hide_index=True)
            st.caption("Razão abaixo de 1: o modelo interno exige menos "
                       "capital que o padronizado na categoria; abaixo de "
                       "0,725 em toda a carteira, o piso passa a vincular.")

# --- Contribuições ---
with tabs[3]:
    if results.get("pooled") and results.get("contrib_bucket") is not None:
        cb = results["contrib_bucket"]
        st.caption(f"Contribuições ao ES {results['contrib_alpha']:.0%} por linha "
                   f"agregada ({results['modelo_contrib']}; ES total "
                   f"R$ {_mi(results['es_total_contrib'])} mi). Grandes exposições têm "
                   f"n_contratos = 1; buckets GH×setor agregam o varejo pulverizado.")
        st.dataframe(cb.head(40).style.format(
            {"exposicao": _FM, "pd": "{:.4%}", "contrib_ES": _FM,
             "contrib_ES_pct": "{:.2%}", "n_contratos": "{:,.0f}"}),
            width='stretch', height=380)
    elif "contribuicoes" not in results:
        st.info("Contribuições não calculadas nesta execução.")
    else:
        cc = results["contribuicoes"]
        st.caption(f"Decomposição Monte Carlo do modelo threshold gaussiano "
                   f"(α = {cc['alpha']:.0%}; VaR médio R$ {_mi(cc['var_medio'])} mi, "
                   f"ES médio R$ {_mi(cc['es_medio'])} mi)")
        df_c = port.df[["id_contraparte", "exposicao", "pd"]].copy()
        if "rating" in port.df.columns:
            df_c["rating"] = port.df["rating"]
        df_c["contrib_VaR"] = cc["var_contrib"]
        df_c["contrib_ES"] = cc["es_contrib"]
        df_c["contrib_ES_%"] = df_c["contrib_ES"] / cc["es_medio"]
        top = df_c.sort_values("contrib_ES", ascending=False)
        st.dataframe(top.head(30).style.format(
            {"exposicao": _FM, "pd": "{:.4%}", "contrib_VaR": _FM,
             "contrib_ES": _FM, "contrib_ES_%": "{:.2%}"}),
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
        if cal.get("cargas_por_setor"):
            st.caption("Correlações calibradas por setor — as métricas acima "
                       "são médias ponderadas pela exposição.")
            _mostrar_leitura_setores(cal, cal["rho_default_intra_alvo"],
                                     cal["rho_default_inter_alvo"])
            with st.expander("Calibração detalhada por setor"):
                st.dataframe(cal["por_setor"][[
                  "setor", "pd_media", "rho_default_intra_alvo",
                  "rho_default_inter_alvo", "rho_ativos_intra",
                  "rho_ativos_inter", "carga_global", "carga_setorial"]
              ].style.format({k: "{:.3f}" for k in [
                  "rho_default_intra_alvo", "rho_default_inter_alvo",
                  "rho_ativos_intra", "rho_ativos_inter", "carga_global",
                  "carga_setorial"]} | {"pd_media": "{:.3%}"}),
                  width='stretch', hide_index=True)
        st.caption(f"Contribuições ao ES {mf['contrib_alpha']:.0%} do modelo "
                   f"gaussiano multifatorial (ES médio R$ {_mi(mf['es_medio'])} mi). "
                   "A razão risco/exposição > 1 indica setor que concentra "
                   "proporcionalmente mais cauda do que carteira.")
        df_s = mf["contrib_por_setor"].copy()
        st.dataframe(df_s.style.format({
            "exposicao_liquida": _FM, "perda_esperada": _FM,
            "contrib_ES": _FM, "exposicao_pct": "{:.1%}",
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
        c1.metric("ECL 12 meses (R$ mi)", _mi(ecl_df['ECL_12m'].sum()))
        c2.metric("ECL lifetime (R$ mi)", _mi(ecl_df['ECL_lifetime'].sum()),
                  f"{ecl_df['ECL_lifetime'].sum()/ecl_df['ECL_12m'].sum():.2f}x o 12m",
                  delta_color="off")
        if "provisao" in ecl_df:
            c3.metric("Estágio 2", f"{int((ecl_df['estagio']==2).sum())} nomes")
            c4.metric("Provisão por estágio (R$ mi)", _mi(ecl_df['provisao'].sum()))
        st.caption(f"Multiperíodo ({sim_m['T']} anos, M={sim_m['M']:,}, blocos de "
                   f"{sim_m['block_size']:,}), em R$ milhões: EL {_mi(sim_m['EL'])} | "
                   f"VaR 99,9% {_mi(sim_m['VaR']['99.90%'])} | "
                   f"ES 99,9% {_mi(sim_m['ES']['99.90%'])}")
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
            .head(50).style.format({"exposicao": _FM, "pd": "{:.4%}",
                "lgd": "{:.0%}", "prazo_anos": "{:.1f}", "pd_matriz_1a": "{:.4%}",
                "ECL_12m": _FM, "ECL_lifetime": _FM,
                "razao_lifetime_12m": "{:.2f}", "provisao": _FM},
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
    _exec = st.session_state.get("executado_em", "")
    if st.session_state.get("_dl_exec") != _exec:      # nova execução: refaz
        st.session_state.pop("_paper", None)
        st.session_state.pop("_excel", None)
        st.session_state["_dl_exec"] = _exec

    def _gerar_paper():
        from engine.report_tex import latex_disponivel, gerar_paper_latex
        args = dict(mig=st.session_state.get("mig"),
                    nome_base=st.session_state.get("nome_base", ""),
                    extra=st.session_state.get("params_ui"),
                    rel_agg=st.session_state.get("rel_agg"),
                    descritiva=st.session_state.get("descritiva"))
        pdf_bytes, tex_zip, aviso = None, None, None
        if latex_disponivel():
            try:
                pdf_bytes, tex_zip = gerar_paper_latex(port, results, **args)
            except Exception as e:
                aviso = ("A compilação LaTeX falhou; gerada a versão "
                         f"simplificada. Detalhe: {str(e)[:300]}")
        else:
            aviso = ("LaTeX indisponível neste ambiente: paper gerado na "
                     "versão simplificada (matplotlib).")
        if pdf_bytes is None:
            from engine.report_paper import gerar_paper
            buf_pdf = io.BytesIO()
            gerar_paper(buf_pdf, port, results, **args)
            pdf_bytes = buf_pdf.getvalue()
        return {"pdf": pdf_bytes, "zip": tex_zip, "aviso": aviso}

    st.caption(f"Arquivos da execução de {_exec}. Valores do Excel em reais "
               "(para auditoria); paper em R$ milhões.")
    d1, d2 = st.columns(2, gap="medium")
    with d1, st.container(border=True, key="cartao_excel"):
        st.markdown("##### :material/table_view: Resultados em Excel")
        st.caption("Todas as tabelas da execução: medidas por modelo, "
                   "capital IRB e padronizado, contribuições, setores, "
                   "migração e calibração.")
        if "_excel" not in st.session_state:
            _bx = io.BytesIO()
            results_to_excel(results, port, _bx)
            st.session_state["_excel"] = _bx.getvalue()
        st.download_button(
            "Baixar Excel", st.session_state["_excel"],
            file_name="resultados_analise.xlsx", on_click="ignore",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet",
            type="primary", icon=":material/download:", key="dl_excel")
    with d2, st.container(border=True, key="cartao_paper"):
        st.markdown("##### :material/description: Paper de validação (PDF)")
        st.caption("Resultados por modelo com a fundamentação teórica, os "
                   "parâmetros desta execução, as referências e o apêndice "
                   "descritivo da base. Leva alguns segundos para compor.")
        _pp = st.session_state.get("_paper")
        if _pp is None:
            if st.button("Gerar paper", icon=":material/play_arrow:",
                         key="gerar_paper"):
                with st.spinner("Compondo o paper (LaTeX)..."):
                    st.session_state["_paper"] = _gerar_paper()
                st.rerun()
        else:
            if _pp["aviso"]:
                st.info(_pp["aviso"])
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "Baixar PDF", _pp["pdf"], file_name="paper_validacao.pdf",
                    mime="application/pdf", on_click="ignore", type="primary",
                    icon=":material/download:", key="dl_paper")
            if _pp["zip"] is not None:
                with b2:
                    st.download_button(
                        "Fonte LaTeX (.zip)", _pp["zip"],
                        file_name="paper_validacao_latex.zip",
                        mime="application/zip", on_click="ignore",
                        icon=":material/folder_zip:", key="dl_tex")
            st.caption(":material/check_circle: Paper pronto. Uma nova "
                       "execução descarta este paper e pede um novo.")
