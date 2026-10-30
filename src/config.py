import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
# 默认放在项目目录下；部署时也可以用环境变量指向别处，避免把大文件塞进仓库
MODELS_DIR = Path(os.getenv("NLP_MODELS_DIR", ROOT_DIR / "models"))
PRE_TRAINED_DIR = Path(os.getenv("NLP_PRETRAINED_DIR", ROOT_DIR / "pretrained"))

SEQ_LEN = 128
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-5
EPOCHS = 20
