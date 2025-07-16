import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality_metrics import calculate_quality_metrics


class TestQualityMetrics(unittest.TestCase):
    
    def setUp(self):
        self.reference_text = "The capital of France is Paris."
        self.candidate_text = "Paris is the capital of France."
        
    def test_calculate_quality_metrics_returns_dict(self):
        """Test that calculate_quality_metrics returns a dictionary with expected keys."""
        result = calculate_quality_metrics(self.reference_text, self.candidate_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('bleu', result)
        self.assertIn('rouge-1', result)
        self.assertIn('rouge-2', result)
        self.assertIn('rouge-l', result)
        
    def test_calculate_quality_metrics_score_ranges(self):
        """Test that all scores are within valid ranges (0-1)."""
        result = calculate_quality_metrics(self.reference_text, self.candidate_text)
        
        for metric_name, score in result.items():
            self.assertGreaterEqual(score, 0.0, f"{metric_name} score should be >= 0")
            self.assertLessEqual(score, 1.0, f"{metric_name} score should be <= 1")
            
    def test_identical_texts_high_scores(self):
        """Test that identical texts produce high similarity scores."""
        result = calculate_quality_metrics(self.reference_text, self.reference_text)
        
        # BLEU and ROUGE scores should be high for identical texts
        self.assertGreater(result['bleu'], 0.8)
        self.assertGreater(result['rouge-1'], 0.8)
        self.assertGreater(result['rouge-l'], 0.8)
        
    def test_empty_candidate_low_scores(self):
        """Test that empty candidate text produces low scores."""
        result = calculate_quality_metrics(self.reference_text, "")
        
        # All scores should be 0 for empty candidate
        for score in result.values():
            self.assertEqual(score, 0.0)
            
    def test_completely_different_texts_low_scores(self):
        """Test that completely different texts produce low scores."""
        different_text = "Elephants are large mammals with trunks."
        result = calculate_quality_metrics(self.reference_text, different_text)
        
        # Scores should be low for completely different texts
        self.assertLess(result['bleu'], 0.3)
        self.assertLess(result['rouge-1'], 0.3)
        self.assertLess(result['rouge-2'], 0.3)
        self.assertLess(result['rouge-l'], 0.3)


if __name__ == '__main__':
    unittest.main()