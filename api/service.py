"""把微调好的 BERT 情感分析模型封装成可复用的推理服务。

这一层只做两件事：加载模型、跑推理，不关心怎么对外暴露。
所以同一份代码既能被 HTTP 接口（main.py）调用，也能被脚本 / 批处理任务直接 import。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# src/ 下的 config.py 与 model.py 用的是绝对导入（import config），
# 这里把目录加进 sys.path 复用它们，避免把模型结构再抄一份导致两边不一致。
for _path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config  # noqa: E402  (位于 src/config.py)
from model import ReviewAnalyzeModel  # noqa: E402  (位于 src/model.py)

POSITIVE = "positive"
NEGATIVE = "negative"


class SentimentService:
    """情感分析推理服务。

    模型加载很慢（BERT 有 1 亿多参数，还要读 400MB 权重），
    所以整个进程只加载一次，后续所有请求复用同一份内存中的模型。
    """

    _instance: Optional["SentimentService"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        pretrained_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_length: int = config.SEQ_LEN,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path or config.MODELS_DIR / "best.pt")
        self.pretrained_dir = Path(pretrained_dir or config.PRE_TRAINED_DIR / "bert-base-chinese")
        self.max_length = max_length
        self.threshold = threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = None
        self.model: Optional[ReviewAnalyzeModel] = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ 加载
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "SentimentService":
        """加载分词器与模型（重复调用只会真正加载一次，且线程安全）。"""
        if self.loaded:
            return self

        with self._load_lock:
            if self.loaded:
                return self

            if not self.model_path.exists():
                raise FileNotFoundError(f"找不到模型权重：{self.model_path}")
            if not self.pretrained_dir.exists():
                raise FileNotFoundError(f"找不到预训练模型目录：{self.pretrained_dir}")

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.pretrained_dir))

            model = ReviewAnalyzeModel()
            # 权重是用 state_dict 保存的，这里按同样的结构读回来
            state_dict = torch.load(str(self.model_path), map_location="cpu")
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model

        return self

    # ------------------------------------------------------------------ 推理
    @torch.inference_mode()
    def predict_batch(self, texts: Sequence[str]) -> List[Dict]:
        """一次处理多条文本，比逐条调用吞吐高得多。"""
        texts = list(texts)
        if not texts:
            return []
        self.load()

        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        logits = self.model(**inputs)          # 形状 [batch_size]
        probs = torch.sigmoid(logits).tolist()  # 概率 = 是好评的可能性

        results = []
        for text, prob in zip(texts, probs):
            label = POSITIVE if prob >= self.threshold else NEGATIVE
            results.append(
                {
                    "text": text,
                    "label": label,
                    # 统一给"预测类别的把握"，好评时是 prob，差评时是 1 - prob
                    "confidence": round(prob if prob >= self.threshold else 1 - prob, 6),
                    "positive_prob": round(prob, 6),
                }
            )
        return results

    def predict(self, text: str) -> Dict:
        """单条预测。"""
        return self.predict_batch([text])[0]


def get_service(**kwargs) -> SentimentService:
    """获取全局唯一的服务实例。"""
    if SentimentService._instance is None:
        with SentimentService._instance_lock:
            if SentimentService._instance is None:
                SentimentService._instance = SentimentService(**kwargs)
    return SentimentService._instance


if __name__ == "__main__":
    # 不启动 Web 服务，直接命令行试一下模型能不能加载并预测
    service = get_service()
    service.load()
    print(f"设备：{service.device}，模型：{service.model_path}")

    samples = [
        "这本书内容非常好，物流也很快，下次还会再来买！",
        "质量太差了，用了两天就坏了，非常失望。",
    ]
    for item in service.predict_batch(samples):
        print(f"[{item['label']:8s}] 置信度 {item['confidence']:.4f}  {item['text']}")
