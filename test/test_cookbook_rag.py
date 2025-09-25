import pytest
import pandas as pd
import numpy as np
import json
from cookbook_rag import (
    load_datasets_with_log,
    generate_embeddings,
    get_embeddings,
    batchify,
    embed_corpus,
    cosine_similarity_manual,
    find_similar_items,
    find_matching_items_with_rag,
    analyze_image,
    analyze_banner_image,
    get_customer_id_by_email,
    answer_with_order_and_customer_history,
    select_best_banner_for_user,
    recommend_items_for_user,
    check_match,
)
from config import STYLES_FILEPATH


def test_load_datasets_with_log():
    datasets = load_datasets_with_log()
    assert isinstance(datasets, dict)
    assert "styles" in datasets
    assert isinstance(datasets["styles"], pd.DataFrame)


def test_generate_embeddings():
    df = pd.DataFrame({"desc": ["A", "B", "C"]})

    # embed_corpusをモックするか、短いテキストでテスト
    def dummy_embed_corpus(texts):
        return [[1, 0, 0]] * len(texts)

    # monkeypatch
    orig_embed_corpus = globals()["embed_corpus"]
    globals()["embed_corpus"] = dummy_embed_corpus
    generate_embeddings(df, "desc")
    assert "embeddings" in df.columns
    assert len(df["embeddings"]) == 3
    globals()["embed_corpus"] = orig_embed_corpus


def test_get_embeddings(monkeypatch):
    # モックでAPI呼び出しを回避
    class DummyData:
        embedding = [1, 2, 3]

    class DummyResponse:
        data = [DummyData(), DummyData()]

    monkeypatch.setattr(
        "cookbook_rag.client.embeddings.create", lambda input, model: DummyResponse()
    )
    result = get_embeddings(["test1", "test2"])
    assert isinstance(result, list)
    assert result[0] == [1, 2, 3]


def test_batchify():
    items = list(range(10))
    batches = list(batchify(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_embed_corpus(monkeypatch):
    # モックでAPI呼び出しを回避
    monkeypatch.setattr("cookbook_rag.get_embeddings", lambda x: [[1, 0, 0]] * len(x))
    corpus = ["A", "B", "C"]
    result = embed_corpus(corpus, batch_size=2, num_workers=1)
    assert isinstance(result, list)
    assert len(result) == 3


def test_cosine_similarity_manual():
    v1 = [1, 0, 0]
    v2 = [1, 0, 0]
    v3 = [0, 1, 0]
    assert cosine_similarity_manual(v1, v2) == pytest.approx(1.0)
    assert cosine_similarity_manual(v1, v3) == pytest.approx(0.0)


def test_find_similar_items():
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    v3 = [1, 1, 0]
    embeddings = [v1, v2, v3]
    result = find_similar_items(v1, embeddings, threshold=0.1, top_k=2)
    assert isinstance(result, list)
    assert len(result) == 2


def test_find_matching_items_with_rag(monkeypatch):
    # モックでAPI呼び出しを回避
    class DummyDF(pd.DataFrame):
        @property
        def _constructor(self):
            return DummyDF

    df = DummyDF({"embeddings": [[1, 0, 0], [0, 1, 0]], "id": [1, 2]})
    monkeypatch.setattr("cookbook_rag.get_embeddings", lambda x: [[1, 0, 0]])
    result = find_matching_items_with_rag(df, ["desc"])
    assert isinstance(result, list)


def test_get_customer_id_by_email():
    datasets = load_datasets_with_log()
    df = datasets["customers"]
    email = df.iloc[0]["email"]
    customer_id = get_customer_id_by_email(email)
    assert customer_id is not None


def test_answer_with_order_and_customer_history(monkeypatch):
    # モックでAPI呼び出しを回避
    monkeypatch.setattr(
        "cookbook_rag.client.chat.completions.create",
        lambda **kwargs: type(
            "obj",
            (object,),
            {
                "choices": [
                    type(
                        "msg",
                        (object,),
                        {"message": type("msg", (object,), {"content": "answer"})()},
                    )()
                ]
            },
        )(),
    )
    datasets = load_datasets_with_log()
    df = datasets["customers"]
    email = df.iloc[0]["email"]
    customer_id = get_customer_id_by_email(email)
    answer = answer_with_order_and_customer_history("test", customer_id)
    assert isinstance(answer, str)


def test_select_best_banner_for_user(monkeypatch):
    # モックでAPI呼び出しを回避
    monkeypatch.setattr(
        "cookbook_rag.client.chat.completions.create",
        lambda **kwargs: type(
            "obj",
            (object,),
            {
                "choices": [
                    type(
                        "msg",
                        (object,),
                        {
                            "message": type(
                                "msg",
                                (object,),
                                {
                                    "content": '{"path":"/images/banner.jpg","reason":"test"}'
                                },
                            )()
                        },
                    )()
                ]
            },
        )(),
    )
    result = select_best_banner_for_user("karen.martin@example.com")
    assert isinstance(result, tuple)
    assert isinstance(result[0], str) or result[0] is None


def test_recommend_items_for_user(monkeypatch):
    # モックでAPI呼び出しを回避
    monkeypatch.setattr(
        "cookbook_rag.client.chat.completions.create",
        lambda **kwargs: type(
            "obj",
            (object,),
            {
                "choices": [
                    type(
                        "msg",
                        (object,),
                        {
                            "message": type(
                                "msg",
                                (object,),
                                {
                                    "content": '[{"id":1,"productDisplayName":"A","Price":100}]'
                                },
                            )()
                        },
                    )()
                ]
            },
        )(),
    )
    result = recommend_items_for_user("karen.martin@example.com")
    assert isinstance(result, list)


def test_check_match(monkeypatch):
    # モックでAPI呼び出しを回避
    monkeypatch.setattr(
        "cookbook_rag.client.chat.completions.create",
        lambda **kwargs: type(
            "obj",
            (object,),
            {
                "choices": [
                    type(
                        "msg",
                        (object,),
                        {
                            "message": type(
                                "msg",
                                (object,),
                                {"content": '{"answer":"yes","reason":"good"}'},
                            )()
                        },
                    )()
                ]
            },
        )(),
    )
    result = check_match("dummy_base64", "dummy_base64")
    assert "answer" in result or isinstance(result, str)
