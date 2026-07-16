"""
Migração de ratings, ECL lifetime e simulação multiperíodo do CreditLens.

Insumo novo: matriz de transição anual P (K x K), em CSV com rótulos de
rating nas linhas (origem) e colunas (destino), último estado = Default
(absorvente). A coluna `rating` da carteira, antes descritiva, passa a
mapear cada contraparte à linha correspondente da matriz.

Blocos funcionais:
  1. Carga e validação da matriz.
  2. Estrutura a termo de PD: PD acumulada até t anos = [P^t]_{r, D}.
  3. ECL 12 meses vs. lifetime por contraparte (CMN 4.966 / IFRS 9),
     com estágios opcionais via coluna `rating_origem`.
  4. Simulação multiperíodo de defaults sobre a MESMA estrutura de
     fatores do modelo threshold (um fator ou multifatorial setorial):
     a linha da matriz do rating corrente vira uma escada de limiares
     K_{r,s} = Phi^{-1}(cumsum_s P_{r,.}) que particiona o latente Y_n
     no rating de destino (CreditMetrics, modo default).

Desempenho (bases grandes):
  A simulação processa as M trajetórias em BLOCOS de tamanho controlado
  (`block_size`), limitando o pico de memória a O(block x N) em vez de
  O(M x N). O gerador PCG64 com SeedSequence por bloco garante
  reprodutibilidade independente do tamanho do bloco. A complexidade de
  tempo é O(M x N x T); ver README para taxas de vazão medidas.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# ----------------------------------------------------------------------
# 1. Matriz de transição
# ----------------------------------------------------------------------

def load_transition_matrix(path_or_buffer, tol: float = 1e-6):
    """Carrega e valida uma matriz de transição anual.

    Formato: CSV com a primeira coluna contendo os ratings de origem e o
    cabeçalho os de destino, na MESMA ordem; último estado = Default.
    Devolve (P, ratings) com P np.ndarray K x K e ratings a lista de
    rótulos (incluindo o estado de default na última posição).
    """
    df = pd.read_csv(path_or_buffer, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]

    if list(df.columns) != list(df.index):
        raise ValueError("A matriz deve ser quadrada com linhas e colunas "
                         "na mesma ordem de ratings.")
    P = df.to_numpy(dtype=float)
    K = P.shape[0]
    if (P < -tol).any():
        raise ValueError("A matriz contém probabilidades negativas.")
    rowsum = P.sum(axis=1)
    if np.abs(rowsum - 1).max() > 1e-3:
        raise ValueError(f"Linhas devem somar 1 (desvio máximo "
                         f"{np.abs(rowsum-1).max():.4f}).")
    P = P / rowsum[:, None]                    # renormaliza resíduos
    if not (P[-1, -1] > 1 - 1e-9 and np.allclose(P[-1, :-1], 0, atol=1e-9)):
        raise ValueError("O último estado deve ser o Default absorvente "
                         "(linha final = [0, ..., 0, 1]).")
    return P, list(df.index)


def map_portfolio_ratings(port_df: pd.DataFrame, ratings: list) -> np.ndarray:
    """Mapeia a coluna `rating` da carteira aos índices da matriz."""
    if "rating" not in port_df.columns:
        raise ValueError("A carteira precisa da coluna 'rating' para o "
                         "módulo de migração.")
    lut = {r: i for i, r in enumerate(ratings)}
    vals = port_df["rating"].astype(str).str.strip()
    unknown = sorted(set(vals) - set(lut))
    if unknown:
        raise ValueError(f"Ratings da carteira ausentes na matriz: {unknown}.")
    idx = vals.map(lut).to_numpy(dtype=np.int64)
    if (idx == len(ratings) - 1).any():
        raise ValueError("Há contrapartes já no estado de Default.")
    return idx


# ----------------------------------------------------------------------
# 2. Estrutura a termo de PD
# ----------------------------------------------------------------------

def pd_term_structure(P: np.ndarray, T: int):
    """PD acumulada e marginal por rating e ano.

    Devolve (cum, mrg), ambos (K-1) x T: cum[r, t-1] = [P^t]_{r, D};
    mrg[r, t-1] = cum_t - cum_{t-1} (probabilidade de defaultar
    exatamente no ano t, visto de hoje).
    """
    K = P.shape[0]
    cum = np.zeros((K - 1, T))
    Pt = np.eye(K)
    for t in range(T):
        Pt = Pt @ P
        cum[:, t] = Pt[:-1, -1]
    mrg = np.diff(np.concatenate([np.zeros((K - 1, 1)), cum], axis=1), axis=1)
    return cum, mrg


# ----------------------------------------------------------------------
# 3. ECL 12 meses vs. lifetime
# ----------------------------------------------------------------------

def compute_ecl(port, P, ratings, discount_rate: float = 0.0,
                horizon_cap: int = 10, stage_notches: int = 2):
    """ECL 12m e lifetime por contraparte; estágios se houver rating_origem.

    Convenções: perdas ao fim de cada ano, descontadas a `discount_rate`;
    horizonte lifetime = ceil(prazo_anos), limitado a `horizon_cap`.
    Estágio 2 quando o rating atual está `stage_notches` ou mais degraus
    abaixo do de origem (deterioração significativa); provisão = ECL 12m
    no estágio 1 e ECL lifetime no estágio 2.
    """
    r_idx = map_portfolio_ratings(port.df, ratings)
    T = int(min(horizon_cap, np.ceil(port.tenor.max())))
    cum, mrg = pd_term_structure(P, T)
    disc = (1.0 + discount_rate) ** -np.arange(1, T + 1)

    life_T = np.minimum(np.ceil(port.tenor).astype(int), T)   # horizonte por nome
    c = port.c
    mrg_n = mrg[r_idx]                                        # N x T
    mask = np.arange(1, T + 1)[None, :] <= life_T[:, None]    # anos dentro do prazo
    ecl_life = ((mrg_n * disc) * mask).sum(axis=1) * c
    ecl_12m = mrg_n[:, 0] * disc[0] * c

    out = port.df[["id_contraparte", "rating", "exposicao", "pd", "lgd",
                   "prazo_anos"]].copy()
    out["pd_matriz_1a"] = cum[r_idx, 0]
    out["ECL_12m"] = ecl_12m
    out["ECL_lifetime"] = ecl_life
    out["razao_lifetime_12m"] = np.where(ecl_12m > 0, ecl_life / ecl_12m, np.nan)

    if "rating_origem" in port.df.columns:
        lut = {r: i for i, r in enumerate(ratings)}
        orig = port.df["rating_origem"].astype(str).str.strip().map(lut)
        downg = r_idx - orig.to_numpy()
        out["estagio"] = np.where(downg >= stage_notches, 2, 1)
        out["provisao"] = np.where(out["estagio"] == 2, ecl_life, ecl_12m)
    return out, cum, T


# ----------------------------------------------------------------------
# 4. Simulação multiperíodo (chunked)
# ----------------------------------------------------------------------

def _threshold_bands(P: np.ndarray) -> np.ndarray:
    """Escada de limiares K x K: bands[r, s] = Phi^{-1}(cumsum_s P_{r,.}).
    Clipping em [1e-12, 1-1e-12] mantém os limiares finitos (|.| < 7,1),
    o que permite o truque de achatamento com offset por linha."""
    cs = np.clip(np.cumsum(P, axis=1), 1e-12, 1.0 - 1e-12)
    return norm.ppf(cs)


def simulate_multiperiod(
    port, P, ratings,
    M: int = 50_000,
    T: int = 5,
    rho_g: float = 0.17,
    rho_s: float | None = None,
    sector_idx: np.ndarray | None = None,
    discount_rate: float = 0.0,
    alphas=np.array([0.95, 0.99, 0.999]),
    block_size: int = 10_000,
    seed: int = 42,
    dtype=np.float64,
    progress=None,
):
    """Distribuição de perdas multiperíodo por migração (modo default).

    Para bases muito grandes, `dtype=np.float32` reduz pela metade a banda
    de memória (o gargalo medido) com erro numérico ordens de magnitude
    abaixo do ruído de Monte Carlo.

    A cada ano, o latente Y_n é sorteado da mesma estrutura de fatores do
    modelo threshold — um fator (rho_s=None) ou multifatorial setorial —
    e classificado na escada de limiares da linha do rating corrente:
    destino = Default acumula a perda c_n descontada; demais destinos
    atualizam o rating para o ano seguinte. Processamento em blocos de
    `block_size` trajetórias limita a memória a O(block x N).
    """
    N, p, c = port.N, port.p, port.c
    r0 = map_portfolio_ratings(port.df, ratings)
    bands = _threshold_bands(P)                # K x K
    D = len(ratings) - 1                       # índice do default
    Kst = bands.shape[1]
    _OFFSET = 100.0                            # >> amplitude dos limiares
    flat_bands = (bands + np.arange(Kst)[:, None] * _OFFSET).ravel().astype(dtype)
    assert np.all(np.diff(flat_bands) >= 0), "escada achatada deve ser não decrescente"
    disc = (1.0 + discount_rate) ** -np.arange(1, T + 1)
    multi = rho_s is not None and sector_idx is not None
    n_sec = int(sector_idx.max()) + 1 if multi else 0

    losses = np.empty(M)
    defaults_por_ano = np.zeros(T)
    seeds = np.random.SeedSequence(seed).spawn(int(np.ceil(M / block_size)))

    done = 0
    for b, ss in enumerate(seeds):
        rng = np.random.default_rng(ss)
        mb = min(block_size, M - done)
        rating = np.tile(r0, (mb, 1))          # mb x N (int)
        alive = np.ones((mb, N), dtype=bool)
        loss_b = np.zeros(mb)

        for t in range(T):
            # --- latente do ano t (fatores independentes entre anos) ---
            G = rng.standard_normal((mb, 1), dtype=dtype)
            eps = rng.standard_normal((mb, N), dtype=dtype)
            if multi:
                F = rng.standard_normal((mb, n_sec), dtype=dtype)
                Y = (np.sqrt(rho_g) * G
                     + np.sqrt(max(rho_s - rho_g, 0.0)) * F[:, sector_idx]
                     + np.sqrt(1.0 - rho_s) * eps)
            else:
                Y = np.sqrt(rho_g) * G + np.sqrt(1.0 - rho_g) * eps

            # --- classificação vetorizada na escada de limiares ---
            # Truque de achatamento: desloca a linha r da escada por r*OFFSET
            # (limiares são finitos e |.| < 8, então as linhas deslocadas não
            # se sobrepõem) e resolve TODAS as células num único searchsorted
            # global: destino = #limiares da linha do rating corrente < Y.
            y_shift = Y + (rating * _OFFSET).astype(dtype)
            ss_idx = np.searchsorted(flat_bands, y_shift.ravel(),
                                     side="left").reshape(Y.shape)
            new_rating = np.minimum(ss_idx - rating * Kst, D)
            new_rating = np.where(alive, new_rating, rating)
            newly_def = alive & (new_rating == D)
            loss_b += (newly_def * c).sum(axis=1) * disc[t]
            defaults_por_ano[t] += newly_def.sum()
            alive &= ~newly_def
            rating = new_rating

        losses[done:done + mb] = loss_b
        done += mb
        if progress:
            progress(f"Multiperíodo: bloco {b+1}/{len(seeds)}")

    losses.sort()
    el = float(losses.mean())
    ul = float(losses.std())
    var = np.quantile(losses, alphas)
    idx = np.searchsorted(losses, var)
    es = np.array([losses[i:].mean() for i in idx])
    return {
        "perdas": losses,
        "EL": el, "UL": ul,
        "VaR": dict(zip([f"{a:.2%}" for a in alphas], map(float, var))),
        "ES": dict(zip([f"{a:.2%}" for a in alphas], map(float, es))),
        "defaults_por_ano": defaults_por_ano / (M * N),  # fração média da carteira
        "T": T, "M": M, "block_size": block_size,
    }


def simulate_multiperiod_pooled(
    port, P, ratings,
    M: int = 100_000, T: int = 5,
    rho_g: float = 0.17, rho_s: float | None = None,
    sector_idx: np.ndarray | None = None,
    discount_rate: float = 0.0,
    alphas=np.array([0.95, 0.99, 0.999]),
    block_size: int = 20_000, seed: int = 42,
    dtype=np.float32, progress=None,
):
    """Multiperíodo para carteiras AGREGADAS (pools + grandes exposições).

    Pools (n_contratos > 1): o estado do bucket é o VETOR DE FRAÇÕES por
    rating, que evolui pela matriz de transição condicional ao cenário
    sistemático do ano — as probabilidades de banda da escada de limiares
    avaliadas no fator sorteado (lei dos grandes números dentro do pool;
    exata no limite, mesma aproximação large-pool do motor de um período).
    Grandes exposições (n = 1): trajetória de rating DISCRETA, como no
    nível contraparte, compartilhando os mesmos fatores do ano.
    Perdas de default descontadas ao ano de ocorrência."""
    df = port.df
    n = df["n_contratos"].to_numpy(int) if "n_contratos" in df.columns \
        else np.ones(port.N, int)
    C = port.c.astype(dtype)
    r0 = map_portfolio_ratings(df, ratings)
    bands = _threshold_bands(P).astype(dtype)          # Kst x Kst
    Kst = bands.shape[1]
    D = len(ratings) - 1
    disc = ((1.0 + discount_rate) ** -np.arange(1, T + 1)).astype(dtype)

    multi = rho_s is not None and sector_idx is not None
    if multi:
        n_sec = int(sector_idx.max()) + 1
        denom = dtype(np.sqrt(1.0 - rho_s))
        load_g = dtype(np.sqrt(rho_g))
        load_s = dtype(np.sqrt(max(rho_s - rho_g, 0.0)))
        sec_of = sector_idx
    else:
        n_sec = 1
        denom = dtype(np.sqrt(1.0 - rho_g))
        load_g = dtype(np.sqrt(rho_g))
        load_s = dtype(0.0)
        sec_of = np.zeros(port.N, int)

    pool = n > 1
    single = ~pool
    idx_pool = np.where(pool)[0]
    idx_sing = np.where(single)[0]
    Gp, Ns = len(idx_pool), len(idx_sing)
    # offset de achatamento p/ classificação discreta dos singles
    _OFFSET = dtype(100.0)
    flat_bands = (bands + np.arange(Kst, dtype=dtype)[:, None] * _OFFSET).ravel()

    losses = np.empty(M)
    defaults_por_ano = np.zeros(T)
    seeds = np.random.SeedSequence(seed).spawn(int(np.ceil(M / block_size)))
    done = 0
    for b, ss in enumerate(seeds):
        rng = np.random.default_rng(ss)
        mb = min(block_size, M - done)
        # pools: frações por rating
        frac = np.zeros((mb, Gp, Kst), dtype=dtype)
        frac[:, np.arange(Gp), r0[idx_pool]] = 1.0
        # singles: estado discreto
        ratingS = np.tile(r0[idx_sing], (mb, 1))
        aliveS = np.ones((mb, Ns), dtype=bool)
        loss_b = np.zeros(mb, dtype=np.float64)

        for t in range(T):
            G = rng.standard_normal((mb, 1), dtype=dtype)
            F = rng.standard_normal((mb, n_sec), dtype=dtype)
            syst_sec = load_g * G + load_s * F          # mb x n_sec

            # ---- pools: matriz condicional por setor ----
            for sec in range(n_sec):
                sel = idx_pool[sec_of[idx_pool] == sec]
                if len(sel) == 0:
                    continue
                gcols = np.searchsorted(idx_pool, sel)
                z = (bands[None, :, :] - syst_sec[:, sec, None, None]) / denom
                cdf = norm.cdf(z).astype(dtype)          # mb x Kst x Kst
                probs = np.diff(cdf, axis=2, prepend=0.0)
                probs[:, :, -1] += 1.0 - cdf[:, :, -1]   # fecha a última banda
                f_old = frac[:, gcols, :]
                f_new = np.einsum("mgk,mkj->mgj", f_old, probs)
                new_def = f_new[:, :, D] - f_old[:, :, D]
                loss_b += (C[sel] * new_def).sum(axis=1) * disc[t]
                defaults_por_ano[t] += float((n[sel] * new_def).sum())
                frac[:, gcols, :] = f_new

            # ---- singles: trajetória discreta ----
            if Ns:
                eps = rng.standard_normal((mb, Ns), dtype=dtype)
                Y = syst_sec[:, sec_of[idx_sing]] + denom * eps
                y_shift = Y + (ratingS * _OFFSET).astype(dtype)
                ss_idx = np.searchsorted(flat_bands, y_shift.ravel(),
                                         side="left").reshape(Y.shape)
                new_rating = np.minimum(ss_idx - ratingS * Kst, D)
                new_rating = np.where(aliveS, new_rating, ratingS)
                newly = aliveS & (new_rating == D)
                loss_b += (newly * C[idx_sing]).sum(axis=1) * disc[t]
                defaults_por_ano[t] += float(newly.sum())
                aliveS &= ~newly
                ratingS = new_rating

        losses[done:done + mb] = loss_b
        done += mb
        if progress:
            progress(f"Multiperíodo pooled: bloco {b+1}/{len(seeds)}")

    losses.sort()
    var = np.quantile(losses, alphas)
    idx = np.searchsorted(losses, var)
    return {
        "perdas": losses,
        "EL": float(losses.mean()),
        "UL": float(losses.std()),
        "VaR": {f"{a:.2%}": float(v) for a, v in zip(alphas, var)},
        "ES": {f"{a:.2%}": float(losses[i:].mean()) for a, i in zip(alphas, idx)},
        "defaults_por_ano": defaults_por_ano / (M * n.sum()),  # fração média dos contratos
        "T": T, "M": M, "block_size": block_size,
    }
