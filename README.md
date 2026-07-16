# CreditLens — Análise de Risco de Carteiras de Crédito

Software de análise de carteiras de crédito construído sobre as bibliotecas
oficiais de **David Jamieson Bolder, *Credit-Risk Modelling: Theoretical
Foundations, Diagnostic Tools, Practical Examples, and Numerical Recipes in
Python* (Springer, 2018)** — o referencial teórico do projeto (`model.pdf`).

## O que o software calcula

| Módulo | Modelos | Referência no livro |
|---|---|---|
| Independentes | Binomial (analítico homogêneo + Monte Carlo) | Cap. 2 |
| Misturas | Beta-binomial calibrado; CreditRisk+ (Poisson-gama 1 fator) | Cap. 3 |
| Limiar | Gaussiano 1 fator; t-Student 1 fator; ASRF analítico | Cap. 4 |
| Regulatório | Capital IRB de Basileia por contraparte (VaR 99,90%), RWA equivalente, ajuste de granularidade Gordy–Lütkebohmert (via CreditRisk+) | Cap. 6 |
| Contribuições | Decomposição Monte Carlo de VaR/ES por contraparte (limiar gaussiano) | Cap. 7 |
| Multifatorial | Limiar gaussiano/t com fatores setoriais (dois níveis: global + setor), calibração intra/inter-setor, contribuições de ES por setor | Cap. 4 (estrutura de correlação) |
| Migração & ECL | Matriz de transição anual (upload), estrutura a termo de PD por potências, ECL 12m vs. lifetime com estágios IFRS 9/CMN 4.966, simulação multiperíodo com escada de limiares sobre a estrutura de fatores — inclusive em modo pooled (frações por rating × matriz condicional) | Caps. 4 (pMap) e sobre cadeias de Markov |
| Agregação & pooled | Base analítica (milhões de contratos) → grandes exposições + buckets GH×setor com EL preservada exatamente; motor pooled (Binomial condicional / large-pool de Vasicek), GA com c² → C²/n, relatório de qualidade (CV de PD intra-bucket) | Gordy–Lütkebohmert (pool-based) |

Medidas em cada modelo: **EL, UL (volatilidade da perda), VaR e Expected
Shortfall** nos níveis 95% / 99% / 99,9% / 99,97% (configuráveis).

**Estratégia de calibração** (seguindo o livro): todos os modelos com
dependência são calibrados ao mesmo par (p̄, ρ-alvo) — PD média ponderada por
exposição e correlação de default alvo — permitindo comparação direta das
caudas entre famílias de modelos.

## Padrão de carteira (upload)

CSV com separador vírgula, decimal ponto, UTF-8
(modelo em `templates/carteira_modelo.csv`):

| Coluna | Obrigatória | Descrição |
|---|---|---|
| `id_contraparte` | sim | identificador único |
| `exposicao` | sim | EAD > 0 |
| `pd` | sim | probabilidade de default 1 ano ∈ (0,1); PD=0 recebe piso de 3 bps |
| `lgd` | não | ∈ (0,1]; default 1,0 — severidade Bolder c = EAD × LGD |
| `prazo_anos` | não | prazo efetivo p/ ajuste de maturidade IRB; default 2,5; truncado a [1,5] |
| `setor` | não | descritiva nos modelos 1F; mapeia a contraparte ao fator setorial no modo multifatorial |
| `rating` | não* | funcional nos módulos de migração/ECL (mapeia à linha da matriz) |
| `rating_origem` | não | rating na originação; habilita a classificação de estágios (≥2 notches → estágio 2) |
| `nome` | não | descritiva |

## Como executar

```bash
pip install -r requirements.txt

# Interface web (upload da carteira, parâmetros, gráficos, download Excel)
streamlit run app.py

# Linha de comando (lote)
python run_analysis.py data/carteira_teste.csv --M 100000 --rho 0.05 --out output/

# Com o modelo multifatorial setorial
python run_analysis.py data/carteira_teste.csv --M 100000 --multifator --rho-intra 0.08 --rho-inter 0.03 --out output/

# Base analítica (contrato a contrato): agrega e roda o motor pooled
python gerar_base_analitica.py   # gera demo de 300 mil contratos
python run_analysis.py data/base_analitica_demo.csv --analitica --M 100000 --max-grandes 200 --out output/

# Com migração de ratings, ECL lifetime e capital multiperíodo
python run_analysis.py data/carteira_teste.csv --M 100000 --multifator --migracao templates/matriz_migracao_modelo.csv --horizonte 5 --taxa-desconto 0.10 --out output/

# Regenerar a base de teste sintética (250 contrapartes)
python gerar_base_teste.py
```

## Estrutura

```
carteira-credito/
├── app.py                  # interface Streamlit
├── run_analysis.py         # CLI em lote (Excel + gráficos)
├── gerar_base_teste.py     # gerador da carteira sintética
├── engine/
│   ├── portfolio.py        # padrão de carteira + validação
│   ├── analytics.py        # orquestração dos modelos do Bolder
│   └── bolder/             # bibliotecas originais do livro (patcheadas*)
├── templates/carteira_modelo.csv
├── data/carteira_teste.csv # 250 contrapartes, 8 graus de rating, EAD 10.000
└── output/                 # resultados da rodada de teste
```

\* Patches mínimos de compatibilidade aplicados ao código de 2018:
`scipy.misc.factorial` → `scipy.special` (SciPy ≥ 1.0 removeu o módulo),
import de `rpy2/GIGrvg` tornado opcional (só a variante variance-gamma o usa),
e atribuição escalar em `varContributions` compatível com NumPy 2.x.
Nenhuma lógica de modelo foi alterada.

## Interpretação (base de teste)

Com ρ-alvo = 5% e ν = 10: o modelo binomial independente subestima
drasticamente a cauda (VaR 99,9% ≈ 3,7% da exposição) porque ignora o risco
sistemático; os modelos de limiar com dependência elevam o VaR 99,9% para
9–14% da exposição, e o t-Student produz a cauda mais pesada (dependência de
cauda). O capital IRB (16,1% + GA de 0,9 p.p. pela concentração — HHI 0,011,
N efetivo ≈ 89 nomes) situa-se entre o Gaussiano e o t, como esperado para a
calibração regulatória (ρ 12–24% > ρ dos modelos internos aqui calibrados).

## Deploy na nuvem (Streamlit Community Cloud — gratuito)

1. Crie um repositório no GitHub (ex.: `creditlens`) e publique este diretório:
   ```bash
   git init && git add . && git commit -m "CreditLens v0.1"
   git branch -M main
   git remote add origin https://github.com/walterCNeto/creditlens.git
   git push -u origin main
   ```
2. Acesse https://share.streamlit.io, conecte sua conta GitHub,
   **New app** → repositório `creditlens` → branch `main` → arquivo `app.py` → Deploy.
3. O app fica vivo em `https://<nome>.streamlit.app` e redeploya sozinho a
   cada `git push`.

Limites do plano gratuito: ~1 GB de RAM (mantenha M ≤ 100 mil para carteiras
com centenas de nomes), app público e hiberna após inatividade (acorda no
primeiro acesso).

**Atenção — dados sensíveis:** um app no plano gratuito é público (a URL é
acessível por qualquer pessoa e os dados enviados são processados na
infraestrutura da Streamlit/AWS). Não carregue carteiras reais/confidenciais
nesse ambiente — use a base de teste ou dados anonimizados. Para uso interno
com dados reais, rode localmente ou em infraestrutura privada.

Alternativas: Hugging Face Spaces (Streamlit nativo, gratuito), Fly.io ou
Railway (containers, planos pagos baratos, permitem autenticação e URL privada).
