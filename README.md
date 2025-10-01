# retailnext-ai

※ Japanese follows
AI-powered Demo Application for Retail & Fashion

---

## Overview

`retailnext-ai` is a web application for the retail industry that leverages AI technologies for product search, recommendation, chatbot, and image analysis.  
It utilizes the OpenAI API, OpenSearch for full-text search, and various datasets to provide features such as product search, suggestions, and image-based recommendations.

---

## Main Features

- **Product Search with Full-Text Capabilities**: Advanced search using OpenSearch with suggest/typeahead functionality
- **AI-Powered Chatbot**: Intelligent Q&A system with customer order history and product knowledge
- **Image-Based Recommendations**: Similar product recommendation via image upload using AI vision
- **Real-Time Search Suggestions**: Fast typeahead search with product thumbnails
- **Customer Personalization**: Gender-aware product sorting and personalized recommendations
- **Marketing Campaign Management**: Dynamic banner
- **AI Product Vectorization**: Embeddings generation for semantic search and recommendations

---

## Setup

1. **Clone the repository**

   ```sh
   git clone https://github.com/yourname/retailnext-ai.git
   cd retailnext-ai
   ```

2. **Start OpenSearch server with Docker Compose**

   ```sh
   docker-compose up -d
   ```

   This will start:

   - OpenSearch server on `localhost:9200`
   - OpenSearch Dashboard on `localhost:5601`

3. **Create and activate a Python virtual environment**

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

5. **Set environment variables (if needed)**

   Create a `.env` file and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

6. **Initialize the database and load sample data**

   ```sh
   python init_data.py
   ```

---

## How to Run

1. **Make sure OpenSearch is running**

   ```sh
   docker-compose up -d
   ```

2. **Start the FastAPI application**

   ```sh
   uvicorn app:app --reload
   ```

3. **Open your browser and navigate to**
   - Main application: [http://localhost:8000](http://localhost:8000)
   - OpenSearch Dashboard: [http://localhost:5601](http://localhost:5601)

---

## OpenSearch Configuration

The application uses OpenSearch for:

- Product full-text search
- Search suggestions with completion
- Product data indexing
- Fast query performance

### Default Configuration

- **OpenSearch Server**: `localhost:9200`
- **Index Name**: `products`
- **Dashboard**: `localhost:5601`

### Stopping OpenSearch

```sh
docker-compose down
```

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
├── docker-compose.yml    # OpenSearch container configuration
├── examples/data/        # Sample product data & images
├── requirements.txt      # Python dependencies
├── init_data.py          # Database initialization script
└── README.md
```

---

## API Endpoints

- `GET /` - Main application page
- `POST /search` - Product search
- `POST /suggest` - Search suggestions
- `POST /chatbot_answer` - Chatbot responses
- `POST /recommendwithselected` - Image-based recommendations

---

## Development & Testing

- You can run tests with `pytest`
- Sample data is located under `examples/data/sample_clothes/`
- Monitor OpenSearch status at `http://localhost:5601`

---

## Troubleshooting

### OpenSearch Issues

- Check if Docker is running: `docker ps`
- Restart OpenSearch: `docker-compose restart`
- View logs: `docker-compose logs opensearch-node`

### Application Issues

- Check Python dependencies: `pip list`
- Verify OpenAI API key is set correctly
- Ensure OpenSearch is accessible at `localhost:9200`

---

## License

MIT License

---

## Notes

- Manage your OpenAI API key and other external API keys via `.env` or environment variables
- For production use, secure your OpenSearch cluster with proper authentication
- For more details and customization, see comments in the source code

---

# 日本語説明

小売・ファッション向け AI 活用デモアプリケーション

---

## 概要

`retailnext-ai` は、商品検索・レコメンド・チャットボット・画像解析など  
AI 技術を活用した小売業向けの Web アプリケーションです。  
OpenAI API、全文検索用 OpenSearch、各種データセットを活用し、商品検索やサジェスト、画像からのレコメンドなどを体験できます。

---

## 主な機能

- **全文検索対応商品検索**: OpenSearch を使用した高度な検索とサジェスト・タイプアヘッド機能
- **AI 搭載チャットボット**: 顧客の注文履歴と商品知識を持つインテリジェントな質問応答システム
- **画像ベースレコメンド**: AI ビジョンを使用した画像アップロードによる類似商品推薦
- **リアルタイム検索サジェスト**: 商品サムネイル付きの高速タイプアヘッド検索
- **顧客パーソナライゼーション**: 性別を考慮した商品ソートと個人向けレコメンド
- **マーケティングキャンペーン管理**: 動的バナー
- **AI 商品ベクトル化**: セマンティック検索とレコメンドのための埋め込み生成

---

## セットアップ

1. **リポジトリをクローン**

   ```sh
   git clone https://github.com/yourname/retailnext-ai.git
   cd retailnext-ai
   ```

2. **Docker Compose で OpenSearch サーバーを起動**

   ```sh
   docker-compose up -d
   ```

   これにより以下が起動します:

   - OpenSearch サーバー: `localhost:9200`
   - OpenSearch ダッシュボード: `localhost:5601`

3. **Python 仮想環境の作成・有効化**

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **依存パッケージのインストール**

   ```sh
   pip install -r requirements.txt
   ```

5. **環境変数の設定（必要に応じて）**

   `.env` ファイルを作成して OpenAI API キーを追加:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

6. **データベースの初期化とサンプルデータの読み込み**

   ```sh
   python init_data.py
   ```

---

## 起動方法

1. **OpenSearch が起動していることを確認**

   ```sh
   docker-compose up -d
   ```

2. **FastAPI アプリケーションを起動**

   ```sh
   uvicorn app:app --reload
   ```

3. **ブラウザでアクセス**
   - メインアプリケーション: [http://localhost:8000](http://localhost:8000)
   - OpenSearch ダッシュボード: [http://localhost:5601](http://localhost:5601)

---

## OpenSearch 設定

アプリケーションは以下の用途で OpenSearch を使用します:

- 商品の全文検索
- 補完機能付き検索サジェスト
- 商品データのインデックス化
- 高速クエリパフォーマンス

### デフォルト設定

- **OpenSearch サーバー**: `localhost:9200`
- **インデックス名**: `products`
- **ダッシュボード**: `localhost:5601`

### OpenSearch の停止

```sh
docker-compose down
```

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
├── docker-compose.yml    # OpenSearchコンテナ設定
├── examples/data/        # サンプル商品データ・画像
├── requirements.txt      # Python依存パッケージ
├── init_data.py          # データベース初期化スクリプト
└── README.md
```

---

## API エンドポイント

- `GET /` - メインアプリケーションページ
- `POST /search` - 商品検索
- `POST /suggest` - 検索サジェスト
- `POST /chatbot_answer` - チャットボット応答
- `POST /recommendwithselected` - 画像ベースレコメンド

---

## 開発・テスト

- テストは `pytest` で実行できます
- サンプルデータは `examples/data/sample_clothes/` 以下に配置
- OpenSearch の状態は `http://localhost:5601` で監視できます

---

## トラブルシューティング

### OpenSearch 関連の問題

- Docker が起動しているか確認: `docker ps`
- OpenSearch を再起動: `docker-compose restart`
- ログを確認: `docker-compose logs opensearch-node`

### アプリケーション関連の問題

- Python 依存関係を確認: `pip list`
- OpenAI API キーが正しく設定されているか確認
- OpenSearch が `localhost:9200` でアクセス可能か確認

---

## ライセンス

MIT License

---

## 補足

- OpenAI API キーや外部 API キーは `.env` などで管理してください
- 本番環境では、適切な認証で OpenSearch クラスターを保護してください
- 詳細な使い方やカスタマイズ方法はソースコード内コメントも参照してください

---
