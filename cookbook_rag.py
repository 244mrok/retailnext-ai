import pandas as pd
import numpy as np
import json
import ast
import tiktoken
import concurrent
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt
from IPython.display import Image, display, HTML
from typing import List
import os
from dotenv import load_dotenv
import time
import base64
from config import (
    GPT_MODEL,
    EMBEDDING_MODEL,
    CACHE_DIR,
    STYLES_FILEPATH,
    ORDER_HISTORY_FILEPATH,
    CUSTOMERS_FILEPATH,
    QANDA_FILEPATH,
    CAMPAIGNS_FILEPATH,
    BANNER_DIR,
    IMG_DIR,
    STYLES_WITH_EMB_FILEPATH,
    TEST_IMAGE_FILENAMES,
    CACHE_EXPIRE_SECONDS,
    EMBEDDING_COST_PER_1K_TOKENS,
)
from utils import (
    encode_image_to_base64,
    safe_email,
    load_cache,
    save_cache,
)

# os.environ
load_dotenv()
client = OpenAI()


## Load and Prepare Dataset
def load_datasets_with_log():
    """各CSVを読み込み、内容と件数をログ出力し、DataFrameをまとめて返す"""
    datasets = {}
    files = [
        ("styles", STYLES_FILEPATH, "items of clothing"),
        ("order_history", ORDER_HISTORY_FILEPATH, "order history records"),
        ("customers", CUSTOMERS_FILEPATH, "Customer Profile"),
        ("qanda", QANDA_FILEPATH, "Q and A"),
        ("campaigns", CAMPAIGNS_FILEPATH, "Marketing Campaigns"),
    ]
    for key, path, label in files:
        df = pd.read_csv(path, on_bad_lines="skip")
        print(df.head())
        print(f"Opened {key} dataset successfully. Dataset has {len(df)} {label}.")
        datasets[key] = df
    return datasets


datasets = load_datasets_with_log()
styles_df = datasets["styles"]
styles_orderhistory_df = datasets["order_history"]
styles_customers_df = datasets["customers"]
styles_qanda_df = datasets["qanda"]
styles_campaigns_df = datasets["campaigns"]


# Function to generate embeddings for a given column in a DataFrame
def generate_embeddings(df, column_name):
    # Initialize an empty list to store embeddings
    descriptions = df[column_name].astype(str).tolist()
    embeddings = embed_corpus(descriptions)

    # Add the embeddings as a new column to the DataFrame
    df["embeddings"] = embeddings
    print("Embeddings created successfully.")


# Simple function to take in a list of text objects and return them as a list of embeddings
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input: List):
    response = client.embeddings.create(input=input, model=EMBEDDING_MODEL).data
    return [data.embedding for data in response]


# Splits an iterable into batches of size n.
def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


# Function for batching and parallel processing the embeddings
def embed_corpus(
    corpus: List[str],
    batch_size=64,
    num_workers=8,
    max_context_len=8191,
):
    # Encode the corpus, truncating to max_context_len
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_corpus = [
        encoded_article[:max_context_len]
        for encoded_article in encoding.encode_batch(corpus)
    ]

    # Calculate corpus statistics: the number of inputs, the total number of tokens, and the estimated cost to embed
    num_tokens = sum(len(article) for article in encoded_corpus)
    cost_to_embed_tokens = num_tokens / 1000 * EMBEDDING_COST_PER_1K_TOKENS
    print(
        f"num_articles={len(encoded_corpus)}, num_tokens={num_tokens}, est_embedding_cost={cost_to_embed_tokens:.2f} USD"
    )

    # Embed the corpus
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = [
            executor.submit(get_embeddings, text_batch)
            for text_batch in batchify(encoded_corpus, batch_size)
        ]

        with tqdm(total=len(encoded_corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(batch_size)

        embeddings = []
        for future in futures:
            data = future.result()
            embeddings.extend(data)

        return embeddings


generate_embeddings(styles_df, "productDisplayName")
print("Writing embeddings to file ...")
styles_df.to_csv(STYLES_WITH_EMB_FILEPATH, index=False)
print("Embeddings successfully stored in STYLES_WITH_EMB_FILEPATH")


def cosine_similarity_manual(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def find_similar_items(input_embedding, embeddings, threshold=0.5, top_k=2):
    """Find the most similar items based on cosine similarity."""

    # Calculate cosine similarity between the input embedding and all other embeddings
    similarities = [
        (index, cosine_similarity_manual(input_embedding, vec))
        for index, vec in enumerate(embeddings)
    ]

    # Filter out any similarities below the threshold
    filtered_similarities = [
        (index, sim) for index, sim in similarities if sim >= threshold
    ]

    # Sort the filtered similarities by similarity score
    sorted_indices = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    # Return the top-k most similar items
    return sorted_indices


def find_matching_items_with_rag(df_items, item_descs):
    """Take the input item descriptions and find the most similar items based on cosine similarity for each description."""

    # Select the embeddings from the DataFrame.
    embeddings = df_items["embeddings"].tolist()
    print(f"***df_items: {df_items}")

    similar_items = []
    for desc in item_descs:

        # Generate the embedding for the input item
        input_embedding = get_embeddings([desc])

        # Find the most similar items based on cosine similarity
        similar_indices = find_similar_items(input_embedding, embeddings, threshold=0.6)
        similar_items += [df_items.iloc[i] for i in similar_indices]
        # print(f"Input Description: {desc}")
        # print(f"Similar Items Found: {len(similar_indices)}")
        # print(f"Similar_indices: {similar_indices}")
        # print(f"similar_items: {similar_items}")

    return similar_items


def analyze_image(image_base64, subcategories):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given an image of an item of clothing, analyze the item and generate a JSON output with the following fields: "items", "category", and "gender".
                           Use your understanding of fashion trends, styles, and gender preferences to provide accurate and relevant suggestions for how to complete the outfit.
                           The items field should be a list of items that would go well with the item in the picture. Each item should represent a title of an item of clothing that contains the style, color, and gender of the item.
                           The category needs to be chosen between the types in this list: {subcategories}.
                           You have to choose between the genders in this list: [Men, Women, Boys, Girls, Unisex]
                           Do not include the description of the item in the picture. Do not include the ```json ``` tag in the output.

                           Example Input: An image representing a black leather jacket.

                           Example Output: {{"items": ["Fitted White Women's T-shirt", "White Canvas Sneakers", "Women's Black Skinny Jeans"], "category": "Jackets", "gender": "Women"}}
                           """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
    )
    # Extract relevant features from the response
    features = response.choices[0].message.content
    return features


def analyze_banner_image(image_base64):
    """
    バナー画像の特徴をAIで抽出する関数
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                            Analyze the banner image and return the following features in JSON format:
	                            •Main color (e.g., red, blue, black, etc.)
	                            •Mood (e.g., casual, formal, sporty, seasonal, etc.)
	                            •Recommended target (e.g., Men, Women, Unisex, Youth, Adults, etc.)
	                            •Items or motifs in the image (e.g., jacket, dress, autumn leaves, sports gear, etc.)
                            Output example: {"main_color": "red", "mood": "casual", "target": "Women", "motif": ["dress", "autumn leaves"]}
                            Do not include explanations or JSON tags. Return only the JSON.
                            """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    # AIの返答（JSON文字列）をそのまま返す
    return response.choices[0].message.content


# 使用例
# email = "sample@example.com"
# customer_id = get_customer_id_by_email(email)
# print(customer_id)
def get_customer_id_by_email(email: str):
    print(f"***email: {email}")
    matched = styles_customers_df.loc[styles_customers_df["email"] == email]
    print(f"***matched: {matched}")
    if len(matched) > 0:
        return matched["customerID"].values[0]
    else:
        return None


# 使用例
# customer_id = 12345
# question = "秋におすすめのコーディネートは？"
# print(answer_with_order_and_customer_history(question, customer_id))
def answer_with_order_and_customer_history(question: str, customer_id: int):
    # 顧客情報取得
    customer_info = styles_customers_df.loc[
        styles_customers_df["customerID"] == customer_id
    ]
    customer_info_dict = (
        customer_info.to_dict(orient="records")[0] if len(customer_info) > 0 else {}
    )

    # 注文履歴取得
    order_history = styles_orderhistory_df.loc[
        styles_orderhistory_df["customerID"] == customer_id
    ]

    # 注文履歴の詳細情報をリスト化
    order_items_detail = order_history[
        [
            "productDisplayName",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "usage",
        ]
    ].to_dict(orient="records")

    # customer_info_dict["sex"] が "Men" などの場合
    filtered_styles = styles_df.loc[styles_df["gender"] == customer_info_dict["sex"]]

    # AIへのプロンプト作成
    prompt = f"""
                You are a fashion advisor and serve as the chatbot for an e-commerce site.  
                Your role is to provide clear, friendly, and helpful advice to customers.  

                Please use the following information as reference:  
                - Customer Information: {customer_info_dict}  
                - Order History Details: {order_items_detail}  
                - Product Catalog: {filtered_styles[["id","productDisplayName", "masterCategory", "subCategory", "articleType", "baseColour", "season", "usage"]].to_dict(orient="records")}  
                - Q&A Data: {styles_qanda_df.to_dict(orient="records")} 
                - Campaign event Data: {styles_campaigns_df.to_dict(orient="records")} 
                - Customer Question: {question}  

                Response Guidelines:  
                1. Always respond in the **same language as the customer’s question**.  
                2. Keep your answers **concise, clear, and easy to read**, using line breaks or bullet points when helpful.  
                3. Consider the customer’s profile and purchase history to provide the **most relevant product suggestions, style advice, or direct answers**.  
                4. Do not return raw catalog, Q&A data or Campaign event Data. Instead, **summarize, rephrase, or translate** it into natural language.  
                5. Do not include irrelevant information or unsupported assumptions. 
                6. Format the response with line breaks to make it easier to read 
                7. When recommending items, include the corresponding id from filtered_styles and output it together with the suggestion. Replace "item_id" with the the id from filtered_styles in the following image path format: <img src="examples/data/sample_clothes/sample_images/item_id.jpg" style="max-height:180px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08);" /><br>

                Example responses:  
                - “For this season, I recommend __.”  
                - “Based on your previous purchase of __, you may like __ as a matching item.”  
            """

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    return answer


def select_best_banner_for_user(user_email: str):
    fileName = "banner"
    fileKey = safe_email(user_email)
    # キャッシュが存在し、24時間以内ならそれを返す
    cached = load_cache(fileName, fileKey)
    if cached:
        return cached["path"], cached["reason"]

    # 1. バナー画像パスを自動生成
    banner_dir = BANNER_DIR
    banner_image_paths = [
        os.path.join(banner_dir, fname)
        for fname in os.listdir(banner_dir)
        if fname.lower().endswith((".jpg")) and fname.startswith("banner")
    ]
    print(f"1.Found {len(banner_image_paths)} banner images.")

    # 2. ユーザー属性取得
    customer_id = get_customer_id_by_email(user_email)
    if customer_id is None:
        return None, "ユーザー情報が見つかりません"
    customer_info = styles_customers_df.loc[
        styles_customers_df["customerID"] == customer_id
    ]
    customer_info_dict = (
        customer_info.to_dict(orient="records")[0] if len(customer_info) > 0 else {}
    )

    print(f"2.Found customer info: {customer_info_dict}")

    # 3. バナー画像ごとにAIで特徴を抽出
    banner_features = []
    for img_path in banner_image_paths:
        img_base64 = encode_image_to_base64(img_path)
        analysis = analyze_banner_image(img_base64)
        banner_features.append({"path": img_path, "features": analysis})

    print(f"3.Analyzed banner features: {banner_features}")

    # 4. AIに「どのバナーが最適か」質問
    prompt = f"""
                You are a fashion advisor.
                Below is the user information and the features of multiple banner images.

                User information: {customer_info_dict}
                Banner image features: {banner_features}

                Select only one banner image path (“path”) that is the most suitable for this user, and briefly explain the reason.
                Return the answer only in JSON format as: "path": "...", "reason": "..." 
    """

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    print(f"4.AI response: {response.choices[0].message.content}")

    raw_content = response.choices[0].message.content
    if isinstance(raw_content, str):
        cleaned = raw_content.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
    elif isinstance(raw_content, dict):
        result = raw_content
    else:
        raise ValueError("Unexpected response type")

    # キャッシュ保存
    save_cache(fileName, fileKey, result)

    return result["path"], result["reason"]


# ユーザーにおすすめ商品を提案する関数
def recommend_items_for_user(user_email: str):
    fileName = "recommend_items"
    fileKey = safe_email(user_email)
    # キャッシュが存在し、24時間以内ならそれを返す
    cached = load_cache(fileName, fileKey)
    if cached:
        return cached

    # 顧客情報取得
    customer_info = styles_customers_df.loc[styles_customers_df["email"] == user_email]
    customer_info_dict = (
        customer_info.to_dict(orient="records")[0] if len(customer_info) > 0 else {}
    )

    # 注文履歴取得
    order_history = styles_orderhistory_df.loc[
        styles_orderhistory_df["customerID"] == customer_info_dict.get("customerID")
    ]

    # 注文履歴の詳細情報をリスト化
    order_items_detail = order_history[
        [
            "productDisplayName",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "usage",
        ]
    ].to_dict(orient="records")

    # 性別に合った商品を抽出
    filtered_styles = styles_df.loc[styles_df["gender"] == customer_info_dict["sex"]]

    # AIにおすすめアイテムを聞くプロンプト
    prompt = f"""
        You are a fashion advisor for an e-commerce site.

        Please recommend 4 items for the following customer based on their profile and order history.
        - Customer Information: {customer_info_dict}
        - Order History Details: {order_items_detail}
        - Product Catalog: {filtered_styles[["id","productDisplayName", "masterCategory", "subCategory", "articleType", "baseColour", "season", "usage", "price"]].to_dict(orient="records")}
        
        Your response must be a JSON output with the following fields: "id","productDisplayName","Price".
        The "id" field must correspond to the id in the product catalog.
        The "productDisplayName" field must be the name of the recommended item.
        The "Price" field must be the price of the recommended item if available, otherwise "".
        Do not include the ```json ``` tag in the output.
        Example Output: ["id": 1234, "productDisplayName": "Fitted Shirt", "Price": 2990]
        """

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    print("AI Recommended Items:\n", answer)

    # 余計なタグや空白を除去
    cleaned = answer.replace("```json", "").replace("```", "").strip()
    items_list = json.loads(cleaned)  # ここでPythonのリストに変換

    items = []
    for row in items_list:
        items.append(
            {
                "img": f"examples/data/sample_clothes/sample_images/{row['id']}.jpg",
                "name": row["productDisplayName"],
                "price": row.get("Price", ""),
            }
        )

    # キャッシュ保存
    save_cache(fileName, fileKey, items)

    return items


def check_match(reference_image_base64, suggested_image_base64):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """ You will be given two images of two different items of clothing.
                            Your goal is to decide if the items in the images would work in an outfit together.
                            The first image is the reference item (the item that the user is trying to match with another item).
                            You need to decide if the second item would work well with the reference item.
                            Your response must be a JSON output with the following fields: "answer", "reason".
                            The "answer" field must be either "yes" or "no", depending on whether you think the items would work well together.
                            The "reason" field must be a short explanation of your reasoning for your decision. Do not include the descriptions of the 2 images.
                            Do not include the ```json ``` tag in the output.
                           """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{reference_image_base64}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{suggested_image_base64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    # Extract relevant features from the response
    features = response.choices[0].message.content
    print("*******Features:*******", features)
    return features


##########################下記はJyupyter Notebook用のコード########################

# Set the path to the images and select a test image
image_path = IMG_DIR
test_images = TEST_IMAGE_FILENAMES

# Encode the test image to base64
reference_image = image_path + test_images[0]
encoded_image = encode_image_to_base64(reference_image)

# Select the unique subcategories from the DataFrame
unique_subcategories = styles_df["articleType"].unique()

# Analyze the image and return the results
analysis = analyze_image(encoded_image, unique_subcategories)
image_analysis = json.loads(analysis)

# Display the image and the analysis results
display(Image(filename=reference_image))
print("Image Analysis: ", image_analysis)

# Extract the relevant features from the analysis
item_descs = image_analysis["items"]
item_category = image_analysis["category"]
item_gender = image_analysis["gender"]


# Filter data such that we only look through the items of the same gender (or unisex) and different category
filtered_items = styles_df.loc[styles_df["gender"].isin([item_gender, "Unisex"])]
filtered_items = filtered_items[filtered_items["articleType"] != item_category]
print(str(len(filtered_items)) + " Remaining Items")

# Find the most similar items based on the input item descriptions
matching_items = find_matching_items_with_rag(filtered_items, item_descs)

# Display the matching items (this will display 2 items for each description in the image analysis)
html = ""
paths = []
for i, item in enumerate(matching_items):
    item_id = item["id"]

    # Path to the image file
    image_path = f"examples/data/sample_clothes/sample_images/{item_id}.jpg"
    paths.append(image_path)
    html += f'<img src="{image_path}" style="display:inline;margin:1px"/>'

# Print the matching item description as a reminder of what we are looking for
print(item_descs)
# Display the image
display(HTML(html))

# Select the unique paths for the generated images
paths = list(set(paths))

for path in paths:
    # Encode the test image to base64
    suggested_image = encode_image_to_base64(path)

    # Check if the items match
    match = json.loads(check_match(encoded_image, suggested_image))

    # Display the image and the analysis results
    if match["answer"] == "yes":
        display(Image(filename=path))
        print("The items match!")
        print(match["reason"])
