"""
CreditLens — Motor de análise de risco de carteiras de crédito.

Construído sobre as bibliotecas de David Jamieson Bolder,
"Credit-Risk Modelling: Theoretical Foundations, Diagnostic Tools,
Practical Examples, and Numerical Recipes in Python" (Springer, 2018).
"""
import os
import sys

# As bibliotecas do Bolder importam-se mutuamente por nome plano
# (ex.: `import cmUtilities as util`), então o diretório precisa estar no path.
_BOLDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bolder")
if _BOLDER_DIR not in sys.path:
    sys.path.insert(0, _BOLDER_DIR)
