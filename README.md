# retailnext-ai

※ Japanese follows
AI-powered Demo Application for Retail & Fashion

---

## Overview

`retailnext-ai` is a web application for the retail industry that leverages AI technologies for product search, recommendation, chatbot, and image analysis.  
It utilizes the OpenAI API and various datasets to provide features such as product search, suggestions, and image-based recommendations.

---

## Main Features

- Product search (with suggest/typeahead)
- Similar product recommendation via image upload
- Chatbot for Q&A
- Marketing campaign management
- AI-based product vectorization (embeddings)

---

## Setup

1. **Clone the repository**

   ```sh
   git clone https://github.com/yourname/retailnext-ai.git
   cd retailnext-ai
   ```

2. **Create and activate a Python virtual environment**

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **(If needed) Set environment variables such as your OpenAI API key**

---

## How to Run

```sh
uvicorn app:app --reload
```

- Then open [http://localhost:8000](http://localhost:8000) in your browser

---

## Project Structure

```
retailnext-ai/
├── app.py                # FastAPI main app
├── cookbook_rag.py       # AI embedding & RAG logic
├── utils.py              # Utility functions
├── script.js             # Frontend JS
├── style.css             # Custom CSS
├── index.html            # Frontend HTML
├── examples/data/        # Sample product data & images
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Development & Testing

- You can run tests with `pytest`
- Sample data is located under `examples/data/sample_clothes/`

---

## License

MIT License

---

## Notes

- Manage your OpenAI API key and other external API keys via `.env` or environment variables.
- For more details and customization, see comments in the source code.

---

#　日本誤訳

小売・ファッション向け AI 活用デモアプリケーション

---

## 概要

`retailnext-ai` は、商品検索・レコメンド・チャットボット・画像解析など  
AI 技術を活用した小売業向けの Web アプリケーションです。  
OpenAI API や各種データセットを活用し、商品検索やサジェスト、画像からのレコメンドなどを体験できます。

---

## 主な機能

- 商品検索（サジェスト・タイプアヘッド対応）
- 商品画像アップロードによる類似商品レコメンド
- チャットボットによる質問応答
- マーケティングキャンペーン管理
- 商品データの AI ベクトル化（埋め込み生成）

---

## セットアップ

1. **リポジトリをクローン**

   ```sh
   git clone https://github.com/yourname/retailnext-ai.git
   cd retailnext-ai
   ```

2. **Python 仮想環境の作成・有効化**

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **依存パッケージのインストール**

   ```sh
   pip install -r requirements.txt
   ```

4. **（必要に応じて）OpenAI API キーなど環境変数を設定**

---

## 起動方法

```sh
uvicorn app:app --reload
```

- ブラウザで [http://localhost:8000](http://localhost:8000) にアクセス

---

## ディレクトリ構成

```
retailnext-ai/
├── app.py                # FastAPIメインアプリ
├── cookbook_rag.py       # AIベクトル化・RAG関連
├── utils.py              # ユーティリティ関数
├── script.js             # フロントエンドJS
├── style.css             # カスタムCSS
├── index.html            # フロントエンドHTML
├── examples/data/        # サンプル商品データ・画像
├── requirements.txt      # Python依存パッケージ
└── README.md
```

---

## 開発・テスト

- テストは `pytest` で実行できます
- サンプルデータは `examples/data/sample_clothes/` 以下に配置

---

## ライセンス

MIT License

---

## 補足

- OpenAI API キーや外部 API キーは `.env` などで管理してください
- 詳細な使い方やカスタマイズ方法はソースコード内コメントも参照してください

---
