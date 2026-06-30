# llm_agent.py

## Purpose

This module handles review sentiment analysis. It combines an LLM-style workflow with a fallback heuristic for local use.

## Main responsibilities

- Reviews customer text input
- Produces a sentiment score and label
- Creates a readable business summary from the sentiment output

## Why it matters

Customer tone often influences pricing acceptance. This module makes the dashboard more realistic by incorporating customer feedback into the recommendation logic.

