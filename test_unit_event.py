import unittest
import numpy as np
import polars as pl
from event_contract_engine import build_event_contract_features

class TestEventContract(unittest.TestCase):
    def test_feature_builder(self):
        n = 500
        df = pl.DataFrame({
            'ts': np.arange(n),
            'open': np.linspace(100, 200, n),
            'high': np.linspace(101, 201, n),
            'low': np.linspace(99, 199, n),
            'close': np.linspace(100, 200, n),
            'um_buy_vol': np.ones(n) * 10,
            'um_sell_vol': np.ones(n) * 8
        })
        feat_df, cols = build_event_contract_features(df, target_period=15)
        self.assertIn('stoch_k69', cols)
        self.assertIn('stoch_d69', cols)
        self.assertIn('vol_imbalance', cols)
        self.assertIn('target', feat_df.columns)
        self.assertEqual(len(feat_df), n)

if __name__ == '__main__':
    unittest.main()
