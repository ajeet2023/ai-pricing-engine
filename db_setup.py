from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "ecommerce_pricing.db"


def _list_csv_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []


def _build_synthetic_pricing_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 220
    price = np.clip(rng.uniform(10.0, 120.0, size=n), 5.0, None)
    quantity = np.maximum(1, np.round(4200 * np.exp(-0.018 * price) * (1 + 0.05 * rng.random(n))).astype(int))
    review_scores = rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30], size=n)
    review_comments = [
        "Great value and fast shipping",
        "Excellent product quality",
        "Delivery was slow",
        "Price feels fair",
        "Would buy again",
        "Poor packaging",
        "Very satisfied with the purchase",
        "Needs better support",
        "Arrived damaged",
        "Excellent customer experience",
    ]

    frame = pd.DataFrame(
        {
            "order_id": [f"ord_{i:03d}" for i in range(1, n + 1)],
            "price": price,
            "quantity": quantity,
            "review_score": review_scores,
            "review_comment_message": [review_comments[i % len(review_comments)] for i in range(n)],
            "product_category_name": ["electronics" if i % 2 else "home" for i in range(n)],
        }
    )
    frame["sentiment_score"] = (frame["review_score"] - 3) / 2
    frame["price_band"] = pd.cut(frame["price"], bins=[0, 25, 50, 75, 100, 200], labels=["budget", "mid", "premium", "high", "luxury"])
    return frame


def _load_real_data() -> pd.DataFrame:
    csv_files = _list_csv_files()
    if not csv_files:
        return _build_synthetic_pricing_data()

    expected_names = [
        "olist_order_items_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
    ]
    file_map = {path.name: path for path in csv_files}

    items_path = file_map.get(expected_names[0])
    reviews_path = file_map.get(expected_names[1])

    if not items_path or not reviews_path:
        return _build_synthetic_pricing_data()

    items = pd.read_csv(items_path)
    reviews = pd.read_csv(reviews_path)

    if "order_id" not in items.columns or "price" not in items.columns:
        return _build_synthetic_pricing_data()

    merged = items[["order_id", "price"]].copy()
    merged["quantity"] = 1
    merged["product_category_name"] = "unknown"

    if "order_id" in reviews.columns:
        review_summary = reviews.groupby("order_id", as_index=False).agg(
            review_score=("review_score", "mean"),
            review_comment_message=("review_comment_message", "first"),
        )
        merged = merged.merge(review_summary, on="order_id", how="left")
        merged["review_score"] = merged["review_score"].fillna(3.0)
        merged["review_comment_message"] = merged["review_comment_message"].fillna("")
        merged["sentiment_score"] = (merged["review_score"] - 3) / 2
    else:
        merged["review_score"] = 3.0
        merged["review_comment_message"] = ""
        merged["sentiment_score"] = 0.0

    merged["price_band"] = pd.cut(merged["price"], bins=[0, 25, 50, 75, 100, 200], labels=["budget", "mid", "premium", "high", "luxury"])
    return merged


def ensure_database() -> Path:
    frame = _load_real_data()
    frame = frame.copy()
    frame["price"] = pd.to_numeric(frame.get("price", 0), errors="coerce")
    frame["quantity"] = pd.to_numeric(frame.get("quantity", 0), errors="coerce")
    frame = frame.dropna(subset=["price", "quantity"])
    frame = frame[(frame["price"] > 0) & (frame["quantity"] > 0)]

    engine = create_engine(f"sqlite:///{DB_PATH}")
    frame.to_sql("pricing_snapshot", engine, if_exists="replace", index=False)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pricing_snapshot_price ON pricing_snapshot(price)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pricing_snapshot_band ON pricing_snapshot(price_band)")
    return DB_PATH


def get_pricing_dataframe() -> pd.DataFrame:
    if not DB_PATH.exists():
        ensure_database()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    return pd.read_sql_query("SELECT * FROM pricing_snapshot", engine)


def get_pricing_insights() -> dict[str, float | str | int]:
    frame = get_pricing_dataframe()
    if frame.empty:
        return {"avg_price": 0.0, "avg_quantity": 0.0, "avg_sentiment": 0.0, "insight": "No data available"}

    avg_price = round(float(frame["price"].mean()), 2)
    avg_quantity = round(float(frame["quantity"].mean()), 2)
    avg_sentiment = round(float(frame["sentiment_score"].mean()), 3)
    insight = "Customers are responding positively to current pricing" if avg_sentiment > 0.1 else "Customer sentiment suggests a pricing review may be needed"
    return {
        "avg_price": avg_price,
        "avg_quantity": avg_quantity,
        "avg_sentiment": avg_sentiment,
        "insight": insight,
    }


if __name__ == "__main__":
    db_path = ensure_database()
    print(f"Database ready at {db_path}")
    print(get_pricing_insights())

