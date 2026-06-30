from __future__ import annotations

import pandas as pd
import streamlit as st

from db_setup import ensure_database, get_pricing_dataframe, get_pricing_insights
from llm_agent import analyze_reviews
from ml_pricing import build_profit_curve, compute_optimal_price, estimate_elasticity, summarize_pricing_model

st.set_page_config(page_title="AI Pricing Engine", layout="wide")
st.title("AI Pricing Engine")
st.caption("A pricing intelligence dashboard that turns transaction, sentiment, and elasticity data into actionable recommendations")

ensure_database()
pricing_df = get_pricing_dataframe()
insights = get_pricing_insights()

review_texts = pricing_df.get("review_comment_message", pd.Series([""], dtype="object")).dropna().astype(str).tolist()
sentiment_summary = analyze_reviews(review_texts)

with st.sidebar:
    st.header("Pricing inputs")
    marginal_cost = st.slider("Marginal cost", 5.0, 150.0, 25.0, step=1.0)
    base_elasticity = estimate_elasticity(pricing_df)["elasticity"]
    elasticity = st.slider("Demand elasticity", -4.0, -0.5, round(float(base_elasticity), 2), step=0.1)
    sentiment = st.slider("Review sentiment", -1.0, 1.0, float(sentiment_summary["sentiment_score"]), step=0.05)
    adjusted_elasticity = elasticity + 0.15 * sentiment

st.subheader("Business summary")
col1, col2, col3 = st.columns(3)
col1.metric("Observed rows", len(pricing_df))
col2.metric("Estimated elasticity", f"{adjusted_elasticity:.2f}")
col3.metric("Optimal price", f"${compute_optimal_price(marginal_cost, adjusted_elasticity):.2f}")

st.subheader("Decision insights")
st.write(f"Database insight: {insights['insight']}")
st.write(f"Average price: ${insights['avg_price']:.2f} | Average quantity: {insights['avg_quantity']:.0f} | Average sentiment: {insights['avg_sentiment']:.2f}")
summary = summarize_pricing_model(marginal_cost, adjusted_elasticity)
st.info(f"Recommendation: {summary['recommendation']}")

st.subheader("Profit curve")
base_quantity = max(float(pricing_df.get("quantity", pd.Series([1000])).mean()), 100.0)
profit_curve = build_profit_curve(
    marginal_cost,
    adjusted_elasticity,
    base_quantity=base_quantity,
    max_price=max(marginal_cost * 2.5, marginal_cost + 40.0),
)
chart_data = profit_curve.set_index("price")[["profit"]]
st.line_chart(chart_data)

st.subheader("How the model works")
with st.expander("View details"):
    st.write(
        "The dashboard estimates elasticity through a log-log OLS regression and then solves the profit-maximizing price using the economics rule $P^* = c \cdot \frac{\epsilon}{\epsilon + 1}$, where $c$ is marginal cost and $\epsilon$ is elasticity."
    )
    st.write(f"Current sentiment score: {sentiment_summary['sentiment_score']:.2f} ({sentiment_summary['sentiment_label']})")
    st.write(sentiment_summary["summary"])
    st.write("The database is stored locally as ecommerce_pricing.db and can be refreshed if new CSV files are added to the data folder.")
