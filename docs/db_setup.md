# db_setup.py

## Purpose

This file prepares and stores the pricing dataset in a local SQLite database.

## Main responsibilities

- Creates a fallback synthetic dataset if no real CSV files are present
- Loads Olist-style CSV files if available
- Stores the processed data into `ecommerce_pricing.db`
- Produces summary insights such as average price and sentiment

## Why it matters

A solid data foundation is essential for any analytics product. This module makes the project reproducible and robust even when raw data is incomplete.
