"""
Gera uma base ANALÍTICA de demonstração: 300 mil contratos, mistura de
varejo pulverizado e corporate, para exercitar a camada de agregação.

Composição:
  - ~299.700 contratos de varejo/PME: exposições log-normais pequenas,
    8 bandas de rating (GH = rating), 8 setores;
  - 300 contratos corporate grandes: exposições log-normais com média
    duas ordens de grandeza acima — são os candidatos naturais ao corte
    de grandes exposições.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260716)
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "C"]
PD_CENTRAL = np.array([0.0003, 0.0008, 0.0020, 0.0050, 0.0150,
                       0.0450, 0.1200, 0.2500])
PROB_RATING = np.array([0.02, 0.06, 0.14, 0.26, 0.27, 0.17, 0.06, 0.02])
SETORES = ["Agro", "Energia", "Indústria", "Varejo", "Construção",
           "Serviços", "Petroquímica", "Logística"]


def gerar(n_varejo=299_700, n_corp=300, path="data/base_analitica_demo.csv"):
    # composição setorial distinta: varejo bem espalhado; corporate
    # concentrado em Energia/Indústria/Petroquímica (concentração realista)
    PESO_SETOR = {
        "VJ": np.array([0.14, 0.08, 0.12, 0.22, 0.14, 0.16, 0.04, 0.10]),
        "CO": np.array([0.05, 0.32, 0.24, 0.05, 0.08, 0.06, 0.15, 0.05]),
    }
    partes = []
    for bloco, n_b, mu, sig in [("VJ", n_varejo, -2.3, 1.0),
                                ("CO", n_corp, 2.6, 0.9)]:
        idx = RNG.choice(len(RATINGS), size=n_b, p=PROB_RATING)
        pdv = np.clip(PD_CENTRAL[idx] * np.exp(RNG.normal(0, 0.30, n_b)),
                      1e-4, 0.6)
        ead = RNG.lognormal(mu, sig, n_b)
        lgd = np.clip(RNG.normal(0.45, 0.12, n_b), 0.10, 0.90)
        partes.append(pd.DataFrame({
            "id_contraparte": [f"{bloco}{i+1:07d}" for i in range(n_b)],
            "rating": [RATINGS[i] for i in idx],
            "gh": [RATINGS[i] for i in idx],
            "setor": RNG.choice(SETORES, size=n_b, p=PESO_SETOR[bloco]),
            "exposicao": ead.round(4),
            "pd": pdv.round(6),
            "lgd": lgd.round(4),
            "prazo_anos": RNG.uniform(1, 5, n_b).round(2),
        }))
    df = pd.concat(partes, ignore_index=True)
    # normaliza o total para 100.000 (unidades monetárias)
    df["exposicao"] = (df["exposicao"] / df["exposicao"].sum()
                       * 100_000).round(6)
    if path:
        df.to_csv(path, index=False)
        print(f"Base analítica: {path} ({len(df):,} linhas, "
              f"EAD {df['exposicao'].sum():,.0f})")
    return df


if __name__ == "__main__":
    gerar()
