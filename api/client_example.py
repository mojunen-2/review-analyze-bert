"""调用上面那个 API 的示例（相当于"外部调用方"）。

先启动服务，再运行本文件：

    python api/main.py                      # 终端 1
    python api/client_example.py            # 终端 2
"""

import requests

BASE_URL = "http://127.0.0.1:8000"


def health() -> dict:
    response = requests.get(f"{BASE_URL}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def predict(text: str) -> dict:
    response = requests.post(f"{BASE_URL}/predict", json={"text": text}, timeout=30)
    response.raise_for_status()
    return response.json()


def predict_batch(texts: list) -> list:
    response = requests.post(f"{BASE_URL}/predict/batch", json={"texts": texts}, timeout=60)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("健康检查：", health())

    print("\n单条预测：")
    for review in [
        "这本书内容非常好，物流也很快，下次还会再来买！",
        "质量太差了，用了两天就坏了，非常失望。",
    ]:
        result = predict(review)
        print(f"  [{result['label']}] 置信度 {result['confidence']:.4f}  {review}")

    print("\n批量预测：")
    for result in predict_batch(
        [
            "包装很仔细，东西和描述的一模一样。",
            "客服态度恶劣，问题一直不解决。",
            "一般般吧，没有什么惊喜。",
        ]
    ):
        print(f"  [{result['label']}] 置信度 {result['confidence']:.4f}  {result['text']}")
