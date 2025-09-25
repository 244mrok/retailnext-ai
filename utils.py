import base64
import os
import json
import time
from config import (
    CACHE_DIR,
    CACHE_EXPIRE_SECONDS,
)


# 画像をBase64エンコードするユーティリティ関数
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


def safe_email(email: str) -> str:
    """メールアドレスをキャッシュキーに安全な形式に変換"""
    return email.replace("@", "_at_").replace(".", "_dot_")


def load_cache(file_name: str, key: str):
    cache_path = os.path.join(CACHE_DIR, f"{file_name}.json")
    """キャッシュファイルが存在し、期限内ならkeyの内容を返す。期限切れ/未登録ならNone"""
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < CACHE_EXPIRE_SECONDS:
            with open(cache_path, "r", encoding="utf-8") as f:
                print("キャッシュから結果を返します")
                cache_data = json.load(f)
                return cache_data.get(key)
    return None


def save_cache(file_name: str, key: str, result):
    cache_path = os.path.join(CACHE_DIR, f"{file_name}.json")
    """キャッシュファイルにkeyで結果を保存（1ファイルに集約）"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            try:
                cache_data = json.load(f)
            except Exception:
                cache_data = {}
    cache_data[key] = result
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
