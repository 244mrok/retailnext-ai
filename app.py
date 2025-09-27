from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import base64
import os
from typing import Optional, Dict
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch
from cookbook_rag import (
    encode_image_to_base64,  # 使っていないなら削除可
    analyze_image,
    find_matching_items_with_rag,
    styles_df,
    get_customer_id_by_email,
    answer_with_order_and_customer_history,
    check_match,
    select_best_banner_for_user,
    recommend_items_for_user,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境はドメインを限定したほうが安全
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 画像ディレクトリ（実ファイルのある場所）
IMG_DIR = Path(__file__).parent / "examples/data/sample_clothes/sample_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)  # 念のため

# /images を静的配信（例: /images/123.jpg でアクセス可能）
app.mount("/images", StaticFiles(directory=str(IMG_DIR)), name="images")


@app.post("/chatbot_answer", response_class=HTMLResponse)
async def chatbot_answer(email: str = Form(...), question: str = Form(...)):
    customer_id = get_customer_id_by_email(email)
    if customer_id is None:
        return "<div style='color:red;'>customer id is not found</div>"
    answer = answer_with_order_and_customer_history(question, customer_id)
    # HTMLで返す（チャットボット内で表示可能）
    return f"<div style='color:#374151; margin:8px 0;'><b>Bot:</b> {answer}</div>"


##本当はviewをjsでやるべきだが、簡易的にここでhtmlを生成して返す
##使われてないのであれば削除可
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    try:
        # 画像をbase64に変換
        contents = await file.read()
        encoded_image = base64.b64encode(contents).decode("utf-8")

        # 候補カテゴリを抽出
        unique_subcategories = styles_df["articleType"].unique()

        # 画像解析
        analysis = analyze_image(encoded_image, unique_subcategories)
        image_analysis = json.loads(analysis)
        item_descs = image_analysis.get("items", [])
        item_category = image_analysis.get("category")
        item_gender = image_analysis.get("gender")

        # データフィルタ（性別と同カテゴリは除外）
        filtered_items = styles_df.loc[
            styles_df["gender"].isin([item_gender, "Unisex"])
        ]
        if item_category:
            filtered_items = filtered_items[
                filtered_items["articleType"] != item_category
            ]

        # 類似アイテム検索
        matching_items = find_matching_items_with_rag(filtered_items, item_descs)

        # 上位4件のみ
        matching_items = matching_items[:4]

        print(
            "*****Matching items*****:", matching_items
        )  # ログ出力（必要に応じて削除）
        if not matching_items:
            return JSONResponse(
                content={"recommendations": [], "analysis": image_analysis},
                media_type="application/json",
            )

        # HTMLとパス（URL）を構築
        paths = []
        html_parts = []
        for item in matching_items:
            item_id = str(item["id"])
            # 実ファイルパス
            file_path = IMG_DIR / f"{item_id}.jpg"
            # 配信URL
            # src_url = f"examples/data/sample_clothes/sample_images/{item_id}.jpg"

            # ファイルが無い場合の簡易フォールバック（必要に応じて削除）
            if not file_path.exists():
                # ここでプレースホルダを使う・スキップするなどの対応
                # continue  # ← 見せたくないならスキップ
                pass

            matched_rows = styles_df.loc[styles_df["id"] == int(item_id)]
            print("matched_rows:", matched_rows)

            if len(matched_rows) > 0:
                productDisplayName = matched_rows["productDisplayName"].values[0]
            else:
                productDisplayName = "不明"

            matched = styles_df.loc[
                styles_df["id"] == int(item_id), "productDisplayName"
            ]
            productDisplayName = matched.values[0] if len(matched) > 0 else "不明"

            paths.append(file_path)
            html_parts.append(
                #    f'<img src="{src_url}" alt="{item.get("productDisplayName", item_id)}" '
                #    f'style="display:inline-block;margin:1px;max-height:180px" />'
                f'<div style="display:inline-block; margin:8px; text-align:center;">'
                f'<img src="{file_path}" alt="{productDisplayName}, {item_id}" '
                f'style="max-height:180px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08);" /><br>'
                f'<span style="font-size:15px; font-weight:bold;">{productDisplayName}</span><br>'
                #                f'<span style="font-size:14px; color:#374151;">¥{productDisplayPrice if productDisplayPrice is not None else "未定"}</span>'
                f"</div>"
            )

        html = "".join(html_parts)

        result = {
            "html": html,  # ← ただの文字列（JSONシリアライズOK）
            "paths": paths,  # ← 画像URLの配列
        }
        print("Recommendations:", result)  # ログ出力（必要に応じて削除）

        return JSONResponse(
            content={"recommendations": result, "analysis": image_analysis},
            media_type="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##本当はviewをjsでやるべきだが、簡易的にここでhtmlを生成して返す
@app.post("/recommendwithselected")
async def recommendWithSelected(file: UploadFile = File(...)):
    try:
        # 画像をbase64に変換
        contents = await file.read()
        encoded_image = base64.b64encode(contents).decode("utf-8")

        # 候補カテゴリを抽出
        unique_subcategories = styles_df["articleType"].unique()

        # 画像解析
        analysis = analyze_image(encoded_image, unique_subcategories)
        image_analysis = json.loads(analysis)
        item_descs = image_analysis.get("items", [])
        item_category = image_analysis.get("category")
        item_gender = image_analysis.get("gender")

        # データフィルタ（性別と同カテゴリは除外）
        filtered_items = styles_df.loc[
            styles_df["gender"].isin([item_gender, "Unisex"])
        ]
        if item_category:
            filtered_items = filtered_items[
                filtered_items["articleType"] != item_category
            ]

        # 類似アイテム検索
        matching_items = []
        cnt = 0
        while not matching_items and cnt < 3:
            matching_items = find_matching_items_with_rag(filtered_items, item_descs)
            print(f"Attempt {cnt}: Matching items found:", matching_items)
            cnt += 1

        html_parts = []
        retry_cnt = 0
        while not html_parts and retry_cnt < 3:
            for i, item in enumerate(matching_items):
                item_id = item["id"]
                # Path to the image file
                image_path = f"examples/data/sample_clothes/sample_images/{item_id}.jpg"
                # Encode the test image to base64
                suggested_image = encode_image_to_base64(image_path)
                # Check if the items match
                match = json.loads(check_match(encoded_image, suggested_image))
                # Display the image and the analysis results
                if match["answer"] == "yes":
                    # 商品名取得
                    matched_rows = styles_df.loc[styles_df["id"] == int(item_id)]
                    if len(matched_rows) > 0:
                        productDisplayName = matched_rows["productDisplayName"].values[
                            0
                        ]
                    else:
                        productDisplayName = ""
                    # 理由と商品名をHTMLで表示
                    html_parts.append(
                        f'<div style="display:inline-block; margin:8px; text-align:center;">'
                        f'<img src="{image_path}" style="display:inline;margin:1px;max-height:180px"/><br>'
                        f'<span style="font-size:15px; font-weight:bold;">{productDisplayName}</span><br>'
                        f'<span style="font-size:14px; color:#374151;">{match["reason"]}</span>'
                        f"</div>"
                    )
            print(f"Attempt {retry_cnt}: Match result:", html_parts)
            retry_cnt += 1

        html = "".join(html_parts)

        result = {
            "html": html,  # ← ただの文字列（JSONシリアライズOK）
            #    "paths": paths,  # ← 画像URLの配列
        }
        print("Recommendations:", result)  # ログ出力（必要に応じて削除）

        return JSONResponse(
            content={"recommendations": result, "analysis": image_analysis},
            media_type="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/select_banner")
async def select_banner(email: str = Form(...)):
    try:
        best_path, reason = select_best_banner_for_user(email)
        if best_path is None:
            return JSONResponse(content={"error": reason}, status_code=404)
        return JSONResponse(content={"banner_path": best_path, "reason": reason})
    except Exception as e:
        # 例外発生時はエラーメッセージを返す
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/recommend_items")
async def recommend_items(email: str = Form(...)):
    try:
        items = recommend_items_for_user(email)
        return JSONResponse(content={"items": items})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 検索エンジン
OS_HOST = os.getenv("OS_HOST", "http://127.0.0.1:9200")
INDEX = "products"

client = OpenSearch(OS_HOST, timeout=10, max_retries=2, retry_on_timeout=True)


class SearchReq(BaseModel):
    q: Optional[str] = None
    filters: Optional[Dict] = None
    sort: Optional[str] = "score"
    page: int = 1
    per_page: int = 24


def build_query(payload: SearchReq):
    must = []
    if payload.q:
        must.append(
            {
                "multi_match": {
                    "query": payload.q,
                    "type": "bool_prefix",
                    "fields": [
                        "productDisplayName_sayt",
                        "productDisplayName_sayt._2gram",
                        "productDisplayName_sayt._3gram",
                    ],
                }
            }
        )
    else:
        must.append({"match_all": {}})

    filters = []
    f = payload.filters or {}
    if "brand" in f:
        filters.append({"terms": {"brand": f["brand"]}})
    if "color" in f:
        filters.append({"terms": {"baseColour": f["color"]}})
    if f.get("in_stock") is not None:
        filters.append({"term": {"in_stock": bool(f["in_stock"])}})
    if "price_min" in f or "price_max" in f:
        rng = {}
        if f.get("price_min") is not None:
            rng["gte"] = f["price_min"]
        if f.get("price_max") is not None:
            rng["lte"] = f["price_max"]
        filters.append({"range": {"price": rng}})

    sort = [{"_score": "desc"}]
    if payload.sort == "price_asc":
        sort = [{"price": "asc"}]
    if payload.sort == "price_desc":
        sort = [{"price": "desc"}]
    if payload.sort == "new":
        sort = [{"updated_at": "desc"}]

    from_ = max(0, (payload.page - 1) * payload.per_page)

    body = {
        "from": from_,
        "size": payload.per_page,
        "query": {"bool": {"must": must, "filter": filters}},
        "aggs": {
            "by_brand": {"terms": {"field": "brand", "size": 50}},
            "by_color": {"terms": {"field": "baseColour", "size": 50}},
        },
        "sort": sort,
    }
    return body


@app.post("/search")
def search(payload: SearchReq):
    try:
        body = build_query(payload)
        resp = client.search(index=INDEX, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    facets = {
        "brand": resp["aggregations"]["by_brand"]["buckets"],
        "color": resp["aggregations"]["by_color"]["buckets"],
    }
    hits = [
        {**h["_source"], "id": h["_id"], "score": h["_score"]}
        for h in resp["hits"]["hits"]
    ]
    return {
        "hits": {
            "hits": [
                {"_source": {**h["_source"], "id": h["_id"], "score": h["_score"]}}
                for h in resp["hits"]["hits"]
            ]
        },
        "total": resp["hits"]["total"]["value"],
        "facets": facets,
        "page": payload.page,
        "per_page": payload.per_page,
    }


class SuggestReq(BaseModel):
    prefix: str


@app.post("/suggest")
def suggest(req: SuggestReq):
    body = {
        "suggest": {
            "name": {
                "prefix": req.prefix,
                "completion": {
                    "field": "productDisplayName_suggest",
                    "skip_duplicates": True,
                    "size": 8,
                },
            }
        }
    }
    r = client.search(index=INDEX, body=body)
    opts = r["suggest"]["name"][0].get("options", [])
    # return {"suggestions": [o["text"] for o in opts]}
    return {"hits": [o["_source"] for o in opts]}


class TypeAheadReq(BaseModel):
    prefix: str


@app.post("/typeahead")
def typeahead(req: TypeAheadReq):
    body = {
        "query": {
            "multi_match": {
                "query": req.prefix,
                "type": "bool_prefix",
                "fields": [
                    "productDisplayName_sayt",
                    "productDisplayName_sayt._2gram",
                    "productDisplayName_sayt._3gram",
                ],
            }
        },
        "size": 8,
    }
    r = client.search(index=INDEX, body=body)
    return {"hits": [h["_source"] for h in r["hits"]["hits"]]}
