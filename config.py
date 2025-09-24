# AI関連の設定
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_COST_PER_1K_TOKENS = 0.00013  # USD


# データファイルのパス
STYLES_FILEPATH = "examples/data/sample_clothes/sample_styles.csv"
ORDER_HISTORY_FILEPATH = "examples/data/sample_clothes/orderHistory.csv"
CUSTOMERS_FILEPATH = "examples/data/sample_clothes/customerProfile.csv"
QANDA_FILEPATH = "examples/data/sample_clothes/qanda.csv"
CAMPAIGNS_FILEPATH = "examples/data/sample_clothes/campaigns.csv"
BANNER_DIR = "examples/data/sample_clothes/sample_banner/"
IMG_DIR = "examples/data/sample_clothes/sample_images/"

# キャッシュ
CACHE_DIR = "cache"
CACHE_EXPIRE_SECONDS = 24 * 60 * 60  # 24時間


# 派生データ（前処理済みなど）のパス
STYLES_WITH_EMB_FILEPATH = (
    "examples/data/sample_clothes/sample_styles_with_embeddings.csv"
)

# 推論・検証用に使うテスト画像ファイル名（IMG_DIRと連結して使う）
TEST_IMAGE_FILENAMES = ["2133.jpg", "7143.jpg", "4226.jpg"]
