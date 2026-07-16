"""
Gera uma carteira de teste sintética no padrão CreditLens.

250 contrapartes corporativas, escala de rating interna de 8 graus
(AAA→C) com PDs crescentes, exposições log-normais (carteira
concentrada, como é típico em crédito corporativo), LGD por classe
de garantia e prazos entre 1 e 5 anos.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260715)

N = 250
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "C"]
PD_CENTRAL = np.array([0.0003, 0.0008, 0.0020, 0.0050, 0.0150, 0.0450, 0.1200, 0.2500])
PROB_RATING = np.array([0.04, 0.08, 0.16, 0.26, 0.24, 0.14, 0.06, 0.02])
SETORES = ["Agro", "Energia", "Indústria", "Varejo", "Construção",
           "Serviços", "Petroquímica", "Logística"]
LGD_CLASSES = {"Sênior garantida": 0.25, "Sênior": 0.45, "Subordinada": 0.75}


def gerar(n=N, path="data/carteira_teste.csv"):
    idx_rating = RNG.choice(len(RATINGS), size=n, p=PROB_RATING)
    # PD com dispersão em torno do central do grau (lognormal multiplicativa)
    pd_vec = PD_CENTRAL[idx_rating] * np.exp(RNG.normal(0, 0.25, n))
    pd_vec = np.clip(pd_vec, 1e-4, 0.5)

    # exposições log-normais — cauda pesada, carteira concentrada
    ead = RNG.lognormal(mean=2.2, sigma=1.1, size=n)
    ead = np.round(ead / ead.sum() * 10_000, 4)  # normaliza p/ total 10.000 (R$ mi)

    classes = RNG.choice(list(LGD_CLASSES.keys()), size=n, p=[0.35, 0.45, 0.20])
    lgd = np.array([LGD_CLASSES[k] for k in classes]) + RNG.normal(0, 0.05, n)
    lgd = np.clip(lgd, 0.10, 0.90).round(4)

    prazo = np.round(RNG.uniform(1.0, 5.0, n), 2)
    setor = RNG.choice(SETORES, size=n)

    df = pd.DataFrame({
        "id_contraparte": [f"CP{i+1:04d}" for i in range(n)],
        "nome": [f"Empresa {i+1:03d}" for i in range(n)],
        "rating": [RATINGS[i] for i in idx_rating],
        "setor": setor,
        "exposicao": ead,
        "pd": pd_vec.round(6),
        "lgd": lgd,
        "prazo_anos": prazo,
    })
    df.to_csv(path, index=False)
    print(f"Carteira de teste gerada: {path} ({n} contrapartes, "
          f"EAD total {df.exposicao.sum():,.0f})")
    return df


if __name__ == "__main__":
    gerar()
