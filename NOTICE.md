# Atribuição e Licença

Este projeto (CreditLens) incorpora, em `engine/bolder/`, as bibliotecas de
David Jamieson Bolder que acompanham o livro *Credit-Risk Modelling:
Theoretical Foundations, Diagnostic Tools, Practical Examples, and Numerical
Recipes in Python* (Springer, 2018), publicadas em
https://github.com/djbolder/credit-risk-modelling sob GPL-3.0.

Por consequência, todo o CreditLens é distribuído sob **GPL-3.0** (ver LICENSE).

Modificações feitas ao código original (sem alteração de lógica de modelo):
- `cmUtilities.py` / `mixtureModels.py`: `scipy.misc.factorial` → `scipy.special`
  (módulo removido no SciPy moderno);
- `thresholdModels.py`: import de `rpy2`/`GIGrvg` tornado opcional;
- `varContributions.py`: atribuição escalar compatível com NumPy 2.x.

O livro em si (PDF) é material protegido por copyright da Springer e
**não** é distribuído neste repositório.
