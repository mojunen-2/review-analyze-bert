"""FastAPI 应用：把 SentimentService 暴露成外部可调用的 HTTP 接口。

启动方式（二选一，工作目录都在 review-analyze-bert 下）：

    uvicorn api.main:app --host 0.0.0.0 --port 8000

或直接运行本文件：

    python api/main.py

启动后浏览器打开 http://127.0.0.1:8000/docs 就能直接调试接口。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

try:  # 以包的形式被导入（uvicorn api.main:app）
    from .service import POSITIVE, NEGATIVE, get_service
except ImportError:  # 直接运行 python api/main.py
    from service import POSITIVE, NEGATIVE, get_service  # type: ignore


# 想让接口只给自己人用，就在启动前设置环境变量 API_KEY（不设置则不校验）
API_KEY = os.getenv("API_KEY", "").strip()


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """可选的鉴权：设置了 API_KEY 环境变量才生效。"""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key 无效")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时先把模型加载好，避免第一个请求等待几十秒
    service = get_service()
    service.load()
    print(f"[启动] 模型加载完成，设备={service.device}，阈值={service.threshold}")
    yield
    print("[关闭] 服务退出")


app = FastAPI(
    title="评论情感分析 API",
    description="基于 bert-base-chinese 微调的电商评论情感二分类接口",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待分析的评论内容")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=256, description="批量评论，最多 256 条")


class PredictResult(BaseModel):
    text: str
    label: str = Field(..., description=f"情感标签：{POSITIVE} / {NEGATIVE}")
    confidence: float = Field(..., description="预测类别的置信度")
    positive_prob: float = Field(..., description="判定为好评的概率")


def _require_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text 不能为空")
    return text


@app.get("/", summary="服务说明")
def root() -> dict:
    return {
        "service": "review-analyze-bert",
        "endpoints": ["/health", "/predict", "/predict/batch", "/docs"],
    }


@app.get("/health", summary="健康检查")
def health() -> dict:
    service = get_service()
    return {
        "status": "ok",
        "model_loaded": service.loaded,
        "device": str(service.device),
        "max_length": service.max_length,
        "threshold": service.threshold,
    }


@app.post("/predict", response_model=PredictResult, summary="单条情感分析", dependencies=[Depends(verify_api_key)])
def predict(payload: PredictRequest) -> PredictResult:
    service = get_service()
    return PredictResult(**service.predict(_require_text(payload.text)))


@app.post(
    "/predict/batch",
    response_model=List[PredictResult],
    summary="批量情感分析",
    dependencies=[Depends(verify_api_key)],
)
def predict_batch(payload: BatchPredictRequest) -> List[PredictResult]:
    texts = [_require_text(t) for t in payload.texts]
    service = get_service()
    return [PredictResult(**item) for item in service.predict_batch(texts)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
