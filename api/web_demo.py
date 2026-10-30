"""情感分析模型的可视化试用页面（Gradio）。

启动：

    python api/web_demo.py

然后浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.service import NEGATIVE, POSITIVE, get_service  # noqa: E402

HOST = os.getenv("WEB_DEMO_HOST", "127.0.0.1")
PORT = int(os.getenv("WEB_DEMO_PORT", "7860"))
# 部署到服务器（没有桌面环境）时可以设成 1，启动时不弹浏览器
OPEN_BROWSER = os.getenv("WEB_DEMO_NO_BROWSER", "") == ""

LABEL_CN = {POSITIVE: "好评 😊", NEGATIVE: "差评 😠"}

EXAMPLES = [
    "这本书内容非常好，物流也很快，下次还会再来买！",
    "质量太差了，用了两天就坏了，非常失望。",
    "包装很仔细，东西和描述的一模一样。",
    "客服态度恶劣，问题一直不解决。",
    "一般般吧，没有什么惊喜，也不算踩雷。",
]


def analyze_one(text: str):
    """单条分析：返回打分分布 + 结论文案。"""
    text = (text or "").strip()
    if not text:
        return {}, "### 请先输入一段评论内容", ""

    result = get_service().predict(text)
    scores = {
        LABEL_CN[POSITIVE]: result["positive_prob"],
        LABEL_CN[NEGATIVE]: round(1 - result["positive_prob"], 6),
    }
    verdict = (
        f"### 判定为 {LABEL_CN[result['label']]}\n"
        f"置信度 **{result['confidence'] * 100:.2f}%**　"
        f"（好评概率 {result['positive_prob'] * 100:.2f}%）"
    )
    return scores, verdict, ""


def analyze_batch(text: str):
    """批量分析：每行一条评论，返回表格。"""
    lines = [line.strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    rows = []
    for result in get_service().predict_batch(lines):
        rows.append(
            [
                result["text"],
                LABEL_CN[result["label"]],
                f"{result['confidence'] * 100:.2f}%",
                f"{result['positive_prob'] * 100:.2f}%",
            ]
        )
    return rows


with gr.Blocks(title="评论情感分析试用") as demo:
    gr.Markdown(
        "# 评论情感分析试用\n"
        "基于 `bert-base-chinese` 微调的电商评论情感二分类模型，"
        "输出 **好评 / 差评** 以及模型置信度。"
    )

    with gr.Tabs():
        # ------------------------------------------------ 单条
        with gr.Tab("单条分析"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_single = gr.Textbox(
                        label="评论内容",
                        placeholder="在这里粘贴一条商品评论……",
                        lines=4,
                    )
                    with gr.Row():
                        btn_single = gr.Button("开始分析", variant="primary")
                        clear_single = gr.Button("清空")
                    gr.Examples(examples=EXAMPLES, inputs=text_single, label="试试这些例子")

                with gr.Column(scale=2):
                    verdict_single = gr.Markdown("### 等待输入……")
                    label_single = gr.Label(label="模型打分", num_top_classes=2)

            btn_single.click(analyze_one, inputs=text_single, outputs=[label_single, verdict_single])
            text_single.submit(analyze_one, inputs=text_single, outputs=[label_single, verdict_single])
            clear_single.click(
                lambda: ("", {}, "### 等待输入……"),
                outputs=[text_single, label_single, verdict_single],
            )

        # ------------------------------------------------ 批量
        with gr.Tab("批量分析"):
            gr.Markdown("每行填一条评论，点按钮一次性分析。")
            text_batch = gr.Textbox(
                label="评论列表（每行一条）",
                placeholder="这本书内容非常好，物流也很快\n质量太差了，用了两天就坏了",
                lines=8,
            )
            btn_batch = gr.Button("批量分析", variant="primary")
            table_batch = gr.Dataframe(
                headers=["评论内容", "情感倾向", "置信度", "好评概率"],
                label="分析结果",
                wrap=True,
            )
            gr.Examples(examples=["\n".join(EXAMPLES[:4])], inputs=text_batch, label="试试这些例子")
            btn_batch.click(analyze_batch, inputs=text_batch, outputs=table_batch)

    gr.Markdown(
        "---\n"
        "模型判定阈值 0.5：好评概率 ≥ 0.5 判为好评。"
        "输入超过 128 个 token 的部分会被截断。"
    )


if __name__ == "__main__":
    print("正在加载模型……")
    service = get_service()
    service.load()
    print(f"模型加载完成，设备 = {service.device}")
    demo.launch(
        server_name=HOST,
        server_port=PORT,
        inbrowser=OPEN_BROWSER,
        theme=gr.themes.Soft(),  # Gradio 6 起主题在这里传入
    )
