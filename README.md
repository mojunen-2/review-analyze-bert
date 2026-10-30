# 中文电商评论情感分析（BERT）

基于 `bert-base-chinese` 微调的中文评论情感二分类项目，包含数据预处理、训练、评估、
命令行预测、FastAPI 接口和 Gradio 网页演示。

## 项目概览

| 项目 | 说明 |
| --- | --- |
| 数据集 | `online_shopping_10_cats`，62,774 条电商评论，覆盖 10 个品类 |
| 标签分布 | 好评 31,728 / 差评 31,046，基本均衡 |
| 模型结构 | BERT-base（12 层，约 1.02 亿参数）+ 单层线性分类头 |
| 微调方式 | 全量微调，学习率 `1e-5` |
| 损失函数 | `BCEWithLogitsLoss`（输出一个 logit，sigmoid 后为好评概率） |
| 序列长度 | 128 token，超出部分截断 |
| 判定阈值 | 好评概率 ≥ 0.5 判为好评 |

## 目录结构

```text
.
├── src/                        # 训练与推理核心代码
│   ├── config.py               # 超参数与路径配置
│   ├── process.py              # 数据清洗、划分、分词、缓存
│   ├── dataset.py              # DataLoader
│   ├── model.py                # BERT + 线性分类头
│   ├── train.py                # 训练循环
│   ├── evaluate.py             # 测试集准确率
│   └── predict.py              # 命令行交互式预测
├── api/
│   ├── service.py              # 模型加载与推理封装（进程内单例）
│   ├── main.py                 # FastAPI 服务
│   ├── web_demo.py             # Gradio 网页演示
│   └── client_example.py       # HTTP 调用示例
├── scripts/
│   └── download_pretrained.py  # 下载 BERT 基座
├── data/raw/                   # 数据集存放位置（不提交）
├── pretrained/                 # BERT 基座（不提交）
└── models/                     # 微调权重 best.pt（不提交）
```

> 本仓库只包含程序代码。训练数据、BERT 基座和微调权重体积过大，不纳入版本控制。

## 快速开始

### 1. 创建环境

```bash
conda create -n review-bert python=3.12 -y
conda activate review-bert
pip install -r requirements.txt
```

PyTorch 建议按自己的 CUDA 版本从 [pytorch.org](https://pytorch.org/get-started/locally/)
选择合适的安装命令。

### 2. 准备数据与模型

**数据集**：下载 `online_shopping_10_cats.csv`（中文电商评论公开数据集，
可从 [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) 获取），
放到 `data/raw/online_shopping_10_cats.csv`。

**BERT 基座**：

```bash
python scripts/download_pretrained.py
```

国内网络较慢时可以先设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`

**微调权重**：把训练得到的 `best.pt` 放到 `models/best.pt`。

如果权重放在其他位置，可以用环境变量指定，无需改动代码：

```bash
set NLP_MODELS_DIR=D:\models\review-bert
set NLP_PRETRAINED_DIR=D:\models\bert-base-chinese
```

### 3. 训练与评估

```bash
cd src
python process.py      # 数据处理并缓存到 data/processed
python train.py        # 训练，最优权重保存为 models/best.pt
python evaluate.py     # 测试集准确率
python predict.py      # 命令行交互式预测
```

### 4. 启动 HTTP 接口

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/docs> 可以直接调试。

单条预测：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"质量太差了，用了两天就坏了\"}"
```

返回示例：

```json
{
  "text": "质量太差了，用了两天就坏了",
  "label": "negative",
  "confidence": 1.0,
  "positive_prob": 0.0
}
```

接口列表：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查，返回模型是否已加载、设备、阈值 |
| POST | `/predict` | 单条评论分析 |
| POST | `/predict/batch` | 批量分析，最多 256 条 |

需要鉴权时，启动前设置 `API_KEY`，调用时带上 `X-API-Key` 请求头。

### 5. 启动网页演示

```bash
python api/web_demo.py
```

浏览器打开 <http://127.0.0.1:7860>。页面包含单条分析和批量分析两个标签页。

想让同一局域网访问，先设置 `WEB_DEMO_HOST=0.0.0.0` 再启动，并放行防火墙端口。

## 设计说明

**为什么模型只加载一次**：`api/service.py` 用进程内单例持有模型，避免每个请求都读取
几百 MB 权重。启动时预热，后续请求复用同一份显存。

**为什么权重不进仓库**：`best.pt` 约 390MB，超过 GitHub 单文件 100MB 的硬限制。
推荐把它上传到 Hugging Face Hub 或作为 GitHub Release 附件发布：

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <your-username>/review-sentiment-bert ./models --repo-type model
```

**关于可复现性**：训练使用固定的随机划分与 `config.py` 中的超参数，
但不同显卡、驱动和 PyTorch 版本仍可能带来细微差异。

## 说明

本项目用于学习和实践，代码仅包含模型训练与推理流程，不包含任何服务器地址、
访问凭证或训练数据。
