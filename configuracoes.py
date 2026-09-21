from pathlib import Path

try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False
    print("⚠  XGBoost não instalado. Execute: pip install xgboost")

PASTA_SAIDA = Path.cwd() / "saida_modelo"
ARQUIVO_CSV = "dados_das_maratonas.csv"

SEED = 42
N_FOLDS = 5
MIN_MEMBROS = 2