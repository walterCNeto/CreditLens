"""
Padrão de carteira do CreditLens e validação de upload.

PADRÃO DO ARQUIVO (CSV, separador vírgula, decimal ponto, UTF-8)
----------------------------------------------------------------
Colunas obrigatórias:
    id_contraparte : identificador único (texto)
    exposicao      : EAD em unidade monetária (> 0)
    pd             : probabilidade de default em 1 ano, em (0, 1)

Colunas opcionais:
    lgd            : perda dado o default, em (0, 1]. Default: 1.0
    prazo_anos     : prazo efetivo (M) em anos, para ajuste de
                     maturidade do IRB. Default: 2.5 (piso 1, teto 5)
    rating         : rótulo de rating interno/externo (texto)
    setor          : setor/segmento (texto ou inteiro)
    nome           : nome da contraparte (texto)

Convenção Bolder: a severidade por contraparte é c_n = EAD_n × LGD_n
(exposição líquida sujeita a perda), usada em todos os modelos.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLS = ["id_contraparte", "exposicao", "pd"]
OPTIONAL_DEFAULTS = {"lgd": 1.0, "prazo_anos": 2.5}


@dataclass
class Portfolio:
    df: pd.DataFrame
    warnings: list = field(default_factory=list)

    # ---- vetores no formato esperado pelas libs do Bolder ----
    @property
    def N(self) -> int:
        return len(self.df)

    @property
    def p(self) -> np.ndarray:
        """Vetor de PDs."""
        return self.df["pd"].to_numpy(dtype=float)

    @property
    def ead(self) -> np.ndarray:
        return self.df["exposicao"].to_numpy(dtype=float)

    @property
    def lgd(self) -> np.ndarray:
        return self.df["lgd"].to_numpy(dtype=float)

    @property
    def c(self) -> np.ndarray:
        """Severidade Bolder: c_n = EAD_n * LGD_n."""
        return self.ead * self.lgd

    @property
    def tenor(self) -> np.ndarray:
        return self.df["prazo_anos"].to_numpy(dtype=float)

    # ---- estatísticas descritivas ----
    def summary(self) -> dict:
        c = self.c
        w = c / c.sum()
        pbar = float(np.dot(w, self.p))          # PD média ponderada por exposição
        el = float(np.dot(self.p, c))            # perda esperada
        hhi = float(np.sum(w ** 2))              # índice de concentração
        return {
            "n_contrapartes": self.N,
            "ead_total": float(self.ead.sum()),
            "exposicao_liquida_total": float(c.sum()),
            "pd_media_simples": float(self.p.mean()),
            "pd_media_ponderada": pbar,
            "lgd_media": float(self.lgd.mean()),
            "perda_esperada": el,
            "perda_esperada_pct": el / c.sum(),
            "hhi": hhi,
            "n_efetivo": 1.0 / hhi,
            "maior_exposicao_pct": float(w.max()),
            "top10_exposicao_pct": float(np.sort(w)[-10:].sum()) if self.N >= 10 else 1.0,
        }


def load_portfolio(path_or_buffer, sep: str = ",", decimal: str = ".") -> Portfolio:
    """Carrega e valida uma carteira no padrão CreditLens."""
    df = pd.read_csv(path_or_buffer, sep=sep, decimal=decimal)
    df.columns = [c.strip().lower() for c in df.columns]
    warnings = []

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {missing}. "
            f"O padrão exige: {REQUIRED_COLS} (ver templates/carteira_modelo.csv)."
        )

    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            warnings.append(f"Coluna opcional '{col}' ausente — usando default {default}.")

    # coerção numérica
    for col in ["exposicao", "pd", "lgd", "prazo_anos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = df[["exposicao", "pd", "lgd", "prazo_anos"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} linha(s) com valores numéricos inválidos "
            f"(ids: {df.loc[bad, 'id_contraparte'].head(5).tolist()}...)."
        )

    # validações de domínio
    if (df["exposicao"] <= 0).any():
        raise ValueError("Há exposições não positivas na carteira.")
    if ((df["pd"] <= 0) | (df["pd"] >= 1)).any():
        n_edge = int(((df["pd"] <= 0) | (df["pd"] >= 1)).sum())
        # PD == 0 é comum em carteiras reais; trata com piso regulatório
        floor = 0.0003  # piso IRB de 3 bps
        df.loc[df["pd"] <= 0, "pd"] = floor
        df.loc[df["pd"] >= 1, "pd"] = 0.9999
        warnings.append(
            f"{n_edge} PD(s) fora de (0,1) ajustadas (piso {floor:.4%} / teto 99,99%)."
        )
    if ((df["lgd"] <= 0) | (df["lgd"] > 1)).any():
        raise ValueError("LGD deve estar em (0, 1].")

    # ajuste de maturidade IRB: 1 <= M <= 5
    clipped = ((df["prazo_anos"] < 1) | (df["prazo_anos"] > 5)).sum()
    if clipped:
        df["prazo_anos"] = df["prazo_anos"].clip(1.0, 5.0)
        warnings.append(f"{int(clipped)} prazo(s) truncado(s) ao intervalo IRB [1, 5] anos.")

    if df["id_contraparte"].duplicated().any():
        raise ValueError("Há id_contraparte duplicados na carteira.")

    return Portfolio(df=df.reset_index(drop=True), warnings=warnings)
