import unittest
import numpy as np
from threshold_engine import (
    EmpiricalBayesEdgeProcessor,
    InSampleQuantileCalibrator,
    RollingQuantileCalibrator,
    MultiScaleConsensusFilter,
    evaluate_causal_performance
)

class TestThresholdEngine(unittest.TestCase):
    def test_empirical_bayes(self):
        eb = EmpiricalBayesEdgeProcessor(prior_weight=20.0, z_crit=1.96)
        clusters = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] * 50)
        targets = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0] * 50)
        edge_map = eb.shrink_edges(clusters, targets)
        self.assertIn(1, edge_map)
        self.assertIn(2, edge_map)

    def test_in_sample_calibrator(self):
        cal = InSampleQuantileCalibrator(target_coverage=0.10)
        train_scores = np.linspace(-1, 1, 1000)
        cal.fit(train_scores)
        self.assertGreater(cal.long_threshold, 0)
        self.assertLess(cal.short_threshold, 0)

        test_scores = np.array([0.95, -0.95, 0.0])
        signals = cal.predict_signals(test_scores)
        self.assertEqual(signals[0], 1)
        self.assertEqual(signals[1], -1)
        self.assertEqual(signals[2], 0)

    def test_consensus_filter(self):
        filt = MultiScaleConsensusFilter(min_consensus_ratio=0.8)
        # 5 尺度: 4正1负 (80%一致)
        edges = np.array([[0.1, 0.2, 0.3, 0.1, -0.05]])
        base_sig = np.array([1])
        filtered = filt.filter(edges, base_sig)
        self.assertEqual(filtered[0], 1)

        # 5 尺度: 3正2负 (60%一致，未达到80%)
        edges_weak = np.array([[0.1, 0.2, 0.3, -0.1, -0.05]])
        filtered_weak = filt.filter(edges_weak, base_sig)
        self.assertEqual(filtered_weak[0], 0)

if __name__ == '__main__':
    unittest.main()
