# AI Pricing Engine

An interactive Streamlit dashboard for pricing intelligence that combines transaction data, review sentiment, and elasticity-based economics to recommend pricing actions.

## What this project does

- Loads or builds a pricing dataset into a local SQLite database
- Uses review text to estimate sentiment
- Runs a log-log OLS regression to estimate demand elasticity
- Calculates an optimal price from a simple profit-maximization rule
- Presents the results in a Streamlit dashboard with clear business insights

## Project structure

- app.py - Streamlit dashboard entry point
- db_setup.py - Database creation and data preparation
- llm_agent.py - Sentiment analysis helper
- ml_pricing.py - Elasticity estimation and pricing logic
- test_pricing.py - Regression tests
- ecommerce_pricing.db - Local SQLite database generated on first run

## Tech stack

- Python 3.10+
- Streamlit
- pandas
- SQLAlchemy
- statsmodels
- numpy
- matplotlib
- openai (optional for API-based sentiment)

## Setup locally

```bash
cd ~/Desktop/ai_pricing_engine
source venv/bin/activate
pip install -r requirements.txt
python -m unittest -v test_pricing.py
streamlit run app.py
```

## How to use it

1. Launch the dashboard with Streamlit.
2. Adjust the marginal cost, elasticity, and sentiment sliders.
3. Review the profit curve and recommendation text.
4. Use the insights to evaluate whether to raise, lower, or hold price.

## Optional data source

If you want to use the real Olist dataset, place the relevant CSV files in the data folder and rerun the database setup.

## Deployment notes

This project is ready to be pushed to GitHub and deployed on Streamlit Cloud.



