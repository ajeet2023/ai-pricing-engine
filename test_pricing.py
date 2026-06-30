import unittest

from ml_pricing import compute_optimal_price, estimate_elasticity, summarize_pricing_model
from db_setup import ensure_database


class PricingEngineTests(unittest.TestCase):
    def test_database_is_created(self) -> None:
        db_path = ensure_database()
        self.assertTrue(db_path.exists())

    def test_optimal_price_is_positive(self) -> None:
        price = compute_optimal_price(20.0, -2.0)
        self.assertGreater(price, 0)

    def test_elasticity_estimation_returns_numeric_value(self) -> None:
        result = estimate_elasticity()
        self.assertIn("elasticity", result)
        self.assertIsInstance(result["elasticity"], float)

    def test_pricing_model_summary_contains_business_guidance(self) -> None:
        summary = summarize_pricing_model(cost=20.0, elasticity=-2.0)
        self.assertIn("recommendation", summary)
        self.assertIn("elasticity_class", summary)
        self.assertTrue(isinstance(summary["recommendation"], str) and len(summary["recommendation"]) > 0)


if __name__ == "__main__":
    unittest.main()

