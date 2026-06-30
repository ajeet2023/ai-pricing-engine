from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from db_setup import get_pricing_dataframe


def estimate_elasticity(data: pd.DataFrame | None = None) -> dict[str, float | object]:
    frame = data.copy() if data is not None else get_pricing_dataframe()
    frame = frame.copy()
    frame["price"] = pd.to_numeric(frame.get("price", 0), errors="coerce")
    frame["quantity"] = pd.to_numeric(frame.get("quantity", 0), errors="coerce")
    frame["sentiment_score"] = pd.to_numeric(frame.get("sentiment_score", 0), errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["price", "quantity"])
    frame = frame[(frame["price"] > 0) & (frame["quantity"] > 0)]

    if frame.empty:
        return {"elasticity": -1.2, "r_squared": 0.0, "model": None}

    frame["log_price"] = np.log(frame["price"])
    frame["log_quantity"] = np.log(frame["quantity"])
    X = sm.add_constant(frame[["log_price", "sentiment_score"]])
    y = frame["log_quantity"]
    model = sm.OLS(y, X).fit()
    elasticity = float(model.params.get("log_price", -1.2))
    return {"elasticity": elasticity, "r_squared": float(model.rsquared), "model": model}


def compute_optimal_price(cost: float, elasticity: float) -> float:
    cost = float(cost)
    elasticity = float(elasticity)
    if cost <= 0:
        cost = 1.0
    if elasticity < -1.0:
        return round(cost * (elasticity / (elasticity + 1)), 2)
    return round(cost * 1.2, 2)


def summarize_pricing_model(cost: float, elasticity: float) -> dict[str, str | float]:
    optimal_price = compute_optimal_price(cost, elasticity)
    elasticity_class = "elastic" if elasticity < -1.0 else "unitary" if abs(elasticity) < 1.1 else "inelastic"
    recommendation = (
        "Lower the price slightly to stimulate demand and protect margin"
        if elasticity_class == "elastic"
        else "Hold price near the current level and monitor demand closely"
        if elasticity_class == "unitary"
        else "You have room to raise price carefully if customer sentiment remains strong"
    )
    return {
        "optimal_price": optimal_price,
        "elasticity_class": elasticity_class,
        "recommendation": recommendation,
    }


def build_profit_curve(cost: float, elasticity: float, base_quantity: float = 1000.0, max_price: float | None = None) -> pd.DataFrame:
    cost = float(cost)
    elasticity = float(elasticity)
    if max_price is None:
        max_price = max(cost * 2.5, cost + 40.0)

    prices = np.linspace(cost, max_price, 120)
    demand = base_quantity * np.power(prices / max(cost, 1.0), elasticity)
    profit = (prices - cost) * demand
    return pd.DataFrame({"price": prices, "profit": profit, "demand": demand})
