"""
Capital padronizado de risco de crédito (RWACPAD) — Resolução BCB nº 229/2022,
com o piso de saída (output floor) da Resolução BCB nº 303/2023, art. 34.

    RWA_CPAD = soma_i  EAD_i x FPR_i

em que EAD_i é a exposição (já com FCC aplicado, quando houver; líquida da
provisão, se a coluna `provisao` for informada) e FPR_i o fator de ponderação
de risco da exposição, definido pela sua categoria. A LGD não entra no
padronizado. O capital é F x RWA, com F = 8%.

Fatores implementados (Res. BCB 229, texto consolidado; artigos citados):

  Instituições financeiras (art. 33), pela categoria A/B/C (arts. 29-32) e
  pelo prazo original (até 90 dias / acima):
      A: 20% / 40%      B: 50% / 75%      C: 150%
  Pessoa jurídica não financeira:
      art. 35  grande empresa de baixo risco ............... 65%
      art. 36  PME: ativo total < R$ 240 mi e receita bruta
               anual < R$ 300 mi .............................. 85%
      art. 41  demais ........................................ 100%
  Varejo (art. 46): pessoa natural ou PJ de pequeno porte (receita
  bruta anual < R$ 15 mi), sem garantia imobiliária, exposição agregada
  por contraparte <= R$ 5 mi e < 0,2% do total de varejo .... 75%
      art. 47  pós-pago ou limite sem atraso, parcelamento ou
               saque nos últimos 360 dias ("transactor") ...... 45%
      art. 48  pessoa natural fora dos arts. 46 e 47 ......... 100%
  Garantidas por imóvel residencial, por LTV (saldo / valor de avaliação):
      art. 50  sem dependência do fluxo do imóvel:
               <=50%: 20 | <=60%: 25 | <=80%: 30 | <=90%: 40 | <=100%: 50 | >100%: 70
      art. 51  com dependência do fluxo do imóvel:
               <=50%: 30 | <=60%: 35 | <=80%: 45 | <=90%: 60 | <=100%: 75 | >100%: 105
  Imóvel não residencial com dependência do fluxo (art. 53):
               <=60%: 70 | <=80%: 90 | >80%: 110
  Outras exposições imobiliárias (art. 54) ................... 150%
  Ativo problemático (art. 66; art. 22, II — prevalece o maior entre o
  FPR do art. 66 e o específico): provisão < 20% do saldo: 150%;
  20% a 50%: 100%; >= 50%: 50%. Residencial sem dependência do fluxo
  do imóvel: 100%.

Mapeamento a partir da categoria IRB (`segmento_irb`) e de colunas
opcionais; na ausência de informação, aplica-se o fator conservador e o
caso é contado nos avisos:

  coluna                    uso
  ------------------------  ---------------------------------------------
  ltv                       imobiliário (fração; ex. 0,65)
  imovel_dependente_fluxo   0/1 — arts. 51 e 53 em vez de 50 e 52
  transactor                0/1 — art. 47 (rotativo sem uso em 360 dias)
  ativo_problematico        0/1 (ou dias_atraso >= 90)
  dias_atraso               dias de atraso
  provisao                  provisão da exposição (mesma unidade de exposicao)
  categoria_if              A, B ou C (instituições financeiras; padrão A)
  prazo_original_dias       prazo original (IFs; padrão: prazo_anos x 365)
  grande_baixo_risco        0/1 — art. 35 (critérios verificados pelo usuário)
  ativo_total_mi            ativo total em R$ milhões (art. 36)
  receita_bruta_mi          receita bruta anual em R$ milhões (arts. 36 e 46)
  fpr                       FPR informado pelo usuário (fração), prevalece

Não implementado: descasamento cambial (art. 55), mitigação de risco de
crédito (Circular 3.809), fatores de conversão de exposições fora do
balanço (a coluna `exposicao` é tomada como a EAD final), financiamentos
especializados (arts. 37-40), participações e cotas de fundos.
"""
import numpy as np
import pandas as pd

F = 0.08
PISO_SAIDA = 0.725                      # Res. BCB 303, art. 34

ARTIGOS = {
    "33-A-curto": ("IF categoria A, prazo original até 90 dias (art. 33)", 0.20),
    "33-A": ("IF categoria A (art. 33)", 0.40),
    "33-B-curto": ("IF categoria B, prazo original até 90 dias (art. 33)", 0.50),
    "33-B": ("IF categoria B (art. 33)", 0.75),
    "33-C": ("IF categoria C (art. 33)", 1.50),
    "35": ("Grande empresa de baixo risco (art. 35)", 0.65),
    "36": ("PME com ativo < R$ 240 mi e receita < R$ 300 mi (art. 36)", 0.85),
    "41": ("Demais pessoas jurídicas (art. 41)", 1.00),
    "46": ("Varejo (art. 46)", 0.75),
    "47": ("Varejo transactor, sem uso em 360 dias (art. 47)", 0.45),
    "48": ("Pessoa natural fora do varejo (art. 48)", 1.00),
    "50": ("Imobiliário residencial, sem dependência do fluxo (art. 50)", None),
    "51": ("Imobiliário residencial, com dependência do fluxo (art. 51)", None),
    "53": ("Imóvel não residencial, com dependência do fluxo (art. 53)", None),
    "54": ("Outras exposições imobiliárias (art. 54)", 1.50),
    "66": ("Ativo problemático (art. 66)", None),
    "informado": ("FPR informado pelo usuário", None),
}

_LTV_50 = ([0.50, 0.60, 0.80, 0.90, 1.00], [0.20, 0.25, 0.30, 0.40, 0.50, 0.70])
_LTV_51 = ([0.50, 0.60, 0.80, 0.90, 1.00], [0.30, 0.35, 0.45, 0.60, 0.75, 1.05])
_LTV_53 = ([0.60, 0.80], [0.70, 0.90, 1.10])


def _faixa(ltv, tabela):
    lim, fpr = tabela
    return np.asarray(fpr)[np.searchsorted(lim, ltv, side="left")]


def _col(df, nome, padrao=np.nan, tipo=float):
    if nome in df.columns:
        return pd.to_numeric(df[nome], errors="coerce").to_numpy(tipo)
    return np.full(len(df), padrao, dtype=tipo)


def fpr_por_exposicao(df, escala_reais=1.0):
    """FPR, artigo e EAD padronizada de cada linha de uma base contrato a
    contrato (ou de uma carteira granular). `escala_reais` converte a
    unidade de `exposicao` em reais (1 = R$; 1.000 = R$ mil), para os
    limites monetários do varejo (art. 46)."""
    n = len(df)
    seg = (df["segmento_irb"].astype(str).str.strip().str.lower().to_numpy()
           if "segmento_irb" in df.columns
           else np.full(n, "corporate", dtype=object))
    ead = pd.to_numeric(df["exposicao"], errors="coerce").to_numpy(float)
    prov = _col(df, "provisao", 0.0)
    prov = np.where(np.isnan(prov), 0.0, prov)
    ead_cpad = np.maximum(ead - prov, 0.0)
    ltv = _col(df, "ltv")
    dep = _col(df, "imovel_dependente_fluxo", 0.0) > 0.5
    transactor = _col(df, "transactor", 0.0) > 0.5
    receita = _col(df, "receita_bruta_mi")
    ativo = _col(df, "ativo_total_mi")
    grande = _col(df, "grande_baixo_risco", 0.0) > 0.5
    atraso = _col(df, "dias_atraso", 0.0)
    probl = (_col(df, "ativo_problematico", 0.0) > 0.5) | (atraso >= 90)
    fpr_inf = _col(df, "fpr")
    cat_if = (df["categoria_if"].astype(str).str.strip().str.upper().to_numpy()
              if "categoria_if" in df.columns else np.full(n, "A", dtype=object))
    prazo_d = _col(df, "prazo_original_dias")
    if "prazo_anos" in df.columns:
        prazo_d = np.where(np.isnan(prazo_d),
                           pd.to_numeric(df["prazo_anos"], errors="coerce")
                           .to_numpy(float) * 365.0, prazo_d)

    fpr = np.full(n, 1.00)
    art = np.full(n, "41", dtype=object)
    avisos = {}

    # ---- varejo: critérios de elegibilidade do art. 46 ----
    peq = (receita < 15.0)                                   # art. 46, § 3º
    cand_varejo = np.isin(seg, ["varejo_demais", "varejo_rotativo"]) | \
        ((seg == "sme") & peq)
    ids = (df["id_contraparte"].astype(str).to_numpy()
           if "id_contraparte" in df.columns else np.arange(n).astype(str))
    exp_reais = ead * escala_reais
    tot_varejo = exp_reais[cand_varejo].sum()
    agreg = pd.Series(exp_reais[cand_varejo]).groupby(ids[cand_varejo]).sum()
    por_cp = np.zeros(n)
    por_cp[cand_varejo] = agreg.reindex(ids[cand_varejo]).to_numpy()
    passa = cand_varejo & (por_cp <= 5_000_000.0) & \
        (por_cp < 0.002 * max(tot_varejo, 1e-12))
    avisos["varejo_reprovado_limites"] = int((cand_varejo & ~passa).sum())

    # pessoa natural fora do varejo (art. 48) e varejo (arts. 46/47)
    pf = np.isin(seg, ["varejo_demais", "varejo_rotativo"])
    m = pf & ~passa
    fpr[m], art[m] = 1.00, "48"
    m = passa
    fpr[m], art[m] = 0.75, "46"
    m = passa & (seg == "varejo_rotativo") & transactor
    fpr[m], art[m] = 0.45, "47"

    # ---- pessoa jurídica não financeira ----
    pj = np.isin(seg, ["corporate", "sme"]) & ~passa
    pme = pj & (receita < 300.0) & (np.isnan(ativo) | (ativo < 240.0))
    fpr[pme], art[pme] = 0.85, "36"
    avisos["pme_sem_ativo_total"] = int((pme & np.isnan(ativo)).sum())
    m = pj & grande
    fpr[m], art[m] = 0.65, "35"
    avisos["pj_sem_receita"] = int((pj & np.isnan(receita) & ~grande).sum())

    # ---- imobiliário ----
    res = seg == "varejo_residencial"
    sem_ltv = np.isnan(ltv)
    ltv_r = np.where(sem_ltv, 9.99, ltv)                    # sem LTV: faixa > 100%
    m = res & ~dep
    fpr[m], art[m] = _faixa(ltv_r[m], _LTV_50), "50"
    m = res & dep
    fpr[m], art[m] = _faixa(ltv_r[m], _LTV_51), "51"
    avisos["residencial_sem_ltv"] = int((res & sem_ltv).sum())
    hv = seg == "hvcre"
    m = hv & ~sem_ltv
    fpr[m], art[m] = _faixa(ltv[m], _LTV_53), "53"
    m = hv & sem_ltv
    fpr[m], art[m] = 1.50, "54"

    # ---- instituições financeiras ----
    ifs = np.isin(seg, ["if", "if_s1s2"])
    curto = prazo_d <= 90.0
    for c in ["A", "B", "C"]:
        m = ifs & (cat_if == c)
        if c == "C":
            fpr[m], art[m] = 1.50, "33-C"
            continue
        v_curto, v_longo = {"A": (0.20, 0.40), "B": (0.50, 0.75)}[c]
        fpr[m & curto], art[m & curto] = v_curto, f"33-{c}-curto"
        fpr[m & ~curto], art[m & ~curto] = v_longo, f"33-{c}"
    avisos["if_categoria_padrao_A"] = int(
        (ifs & ("categoria_if" not in df.columns)).sum())

    # ---- ativo problemático (art. 66; art. 22, II: prevalece o maior) ----
    pct = np.where(ead > 0, prov / np.maximum(ead, 1e-12), 0.0)
    f66 = np.where(pct >= 0.50, 0.50, np.where(pct >= 0.20, 1.00, 1.50))
    f66 = np.where(res & ~dep, 1.00, f66)
    m = probl & (f66 > fpr)
    fpr[m], art[m] = f66[m], "66"
    avisos["ativos_problematicos"] = int(probl.sum())

    # ---- FPR informado prevalece ----
    m = ~np.isnan(fpr_inf)
    fpr[m], art[m] = fpr_inf[m], "informado"
    avisos["fpr_informado"] = int(m.sum())

    return fpr, art, ead_cpad, avisos


def resumo(fpr, art, ead_cpad, seg=None):
    """Quadros por artigo e por categoria IRB."""
    rwa = ead_cpad * fpr
    d = pd.DataFrame({"artigo": art, "ead": ead_cpad, "rwa": rwa})
    pa = d.groupby("artigo").agg(linhas=("ead", "size"), ead=("ead", "sum"),
                                 rwa=("rwa", "sum")).reset_index()
    pa["fpr_medio"] = pa["rwa"] / pa["ead"].where(pa["ead"] > 0)
    pa["descricao"] = pa["artigo"].map(lambda a: ARTIGOS.get(a, (a,))[0])
    pa = pa.sort_values("rwa", ascending=False)
    ps = None
    if seg is not None:
        d["segmento"] = seg
        ps = d.groupby("segmento").agg(ead=("ead", "sum"),
                                       rwa_cpad=("rwa", "sum")).reset_index()
    return pa, ps


def calcular(port, escala_reais=1.0, rel_agg=None):
    """RWA padronizado da carteira de modelagem.

    Na carteira agregada, o RWA de cada linha vem da coluna `rwa_cpad`,
    calculada contrato a contrato na agregação (o FPR é linear na
    exposição, então a soma por grupo é exata); os quadros por artigo vêm
    do relatório de agregação. Na carteira granular, calcula linha a linha."""
    df = port.df
    seg = (df["segmento_irb"].astype(str).to_numpy()
           if "segmento_irb" in df.columns
           else np.full(len(df), "corporate", dtype=object))
    if "rwa_cpad" in df.columns:
        rwa = df["rwa_cpad"].to_numpy(float)
        ead = df["ead_cpad"].to_numpy(float) if "ead_cpad" in df.columns \
            else df["exposicao"].to_numpy(float)
        cp = (rel_agg or {}).get("cpad", {})
        pa = cp.get("por_artigo")
        avisos = cp.get("avisos", {})
        d = pd.DataFrame({"segmento": seg, "ead": ead, "rwa_cpad": rwa})
        ps = d.groupby("segmento").agg(ead=("ead", "sum"),
                                       rwa_cpad=("rwa_cpad", "sum")).reset_index()
    else:
        fpr, art, ead, avisos = fpr_por_exposicao(df, escala_reais)
        rwa = ead * fpr
        pa, ps = resumo(fpr, art, ead, seg)
    return {"rwa": rwa, "rwa_total": float(rwa.sum()),
            "capital": float(rwa.sum() * F), "por_artigo": pa,
            "por_segmento": ps, "avisos": avisos,
            "fpr_medio": float(rwa.sum() / max(ead.sum(), 1e-12))}


def piso_saida(rwa_irb, rwa_cpad):
    """Res. BCB 303, art. 34: RWA = max(RWA modelos internos;
    72,5% x RWA_CPAD), para as mesmas exposições."""
    piso = PISO_SAIDA * rwa_cpad
    final = max(rwa_irb, piso)
    return {"rwa_irb": rwa_irb, "rwa_cpad": rwa_cpad, "piso": piso,
            "rwa_final": final, "capital_final": final * F,
            "vinculante": bool(piso > rwa_irb),
            "razao_irb_cpad": rwa_irb / rwa_cpad if rwa_cpad > 0 else np.nan}
