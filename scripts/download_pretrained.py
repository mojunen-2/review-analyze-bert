"""把 BERT 中文基座下载到 pretrained/bert-base-chinese。

用法：

    python scripts/download_pretrained.py

国内网络较慢时，可以先设置镜像再运行：

    set HF_ENDPOINT=https://hf-mirror.com
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT_DIR / "pretrained" / "bert-base-chinese"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"开始下载 bert-base-chinese -> {TARGET_DIR}")
    snapshot_download(
        repo_id="bert-base-chinese",
        local_dir=str(TARGET_DIR),
    )
    print("下载完成。")


if __name__ == "__main__":
    main()
