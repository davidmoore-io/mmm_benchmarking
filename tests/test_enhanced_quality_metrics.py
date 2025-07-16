import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import math

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_quality_metrics import (
    PerplexityCalculator, 
    SemanticSimilarityCalculator, 
    TaskSpecificEvaluator,
    EnhancedQualityMetrics
)


class TestPerplexityCalculator(unittest.TestCase):
    
    def setUp(self):
        self.calc = PerplexityCalculator()
        self.reference_text = "The capital of France is Paris."
        self.candidate_text = "Paris is the capital of France."
        
    def test_calculate_perplexity_returns_float(self):
        """Test that perplexity calculation returns a float."""
        result = self.calc.calculate_perplexity(self.reference_text, self.candidate_text)
        self.assertIsInstance(result, float)
        
    def test_calculate_perplexity_empty_candidate(self):
        """Test perplexity calculation with empty candidate text."""
        result = self.calc.calculate_perplexity(self.reference_text, "")
        self.assertEqual(result, float('inf'))
        
    def test_calculate_perplexity_identical_texts(self):
        """Test perplexity calculation with identical texts."""
        result = self.calc.calculate_perplexity(self.reference_text, self.reference_text)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
        
    def test_get_bigrams(self):
        """Test bigram extraction."""
        tokens = ["the", "cat", "sat", "on", "mat"]
        bigrams = self.calc._get_bigrams(tokens)
        expected = [("the", "cat"), ("cat", "sat"), ("sat", "on"), ("on", "mat")]
        self.assertEqual(bigrams, expected)
        
    def test_bigram_probability(self):
        """Test bigram probability calculation."""
        ref_tokens = ["the", "cat", "sat", "on", "the", "mat"]
        ref_bigrams = self.calc._get_bigrams(ref_tokens)
        prob = self.calc._bigram_probability(("the", "cat"), ref_bigrams, ref_tokens)
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1)


class TestSemanticSimilarityCalculator(unittest.TestCase):
    
    def setUp(self):
        # Mock the SentenceTransformer to avoid downloading models during tests
        with patch('enhanced_quality_metrics.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_st.return_value = mock_model
            self.calc = SemanticSimilarityCalculator()
            
    def test_calculate_semantic_similarity_returns_dict(self):
        """Test that semantic similarity calculation returns a dictionary."""
        result = self.calc.calculate_semantic_similarity("test", "test")
        self.assertIsInstance(result, dict)
        
    def test_calculate_semantic_similarity_keys(self):
        """Test that semantic similarity returns expected keys."""
        result = self.calc.calculate_semantic_similarity("test", "test")
        expected_keys = [
            'cosine_similarity', 'euclidean_similarity', 
            'manhattan_similarity', 'semantic_textual_similarity'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
            
    def test_calculate_semantic_similarity_no_model(self):
        """Test semantic similarity calculation when model is None."""
        self.calc.model = None
        result = self.calc.calculate_semantic_similarity("test", "test")
        for value in result.values():
            self.assertIsInstance(value, float)


class TestTaskSpecificEvaluator(unittest.TestCase):
    
    def setUp(self):
        with patch('enhanced_quality_metrics.stopwords') as mock_stopwords:
            mock_stopwords.words.return_value = ['the', 'is', 'a', 'an']
            self.evaluator = TaskSpecificEvaluator()
            
    def test_evaluate_factual_accuracy_returns_dict(self):
        """Test that factual accuracy evaluation returns a dictionary."""
        result = self.evaluator.evaluate_factual_accuracy("Paris is capital", "Paris is capital")
        self.assertIsInstance(result, dict)
        
    def test_evaluate_factual_accuracy_keys(self):
        """Test that factual accuracy evaluation returns expected keys."""
        result = self.evaluator.evaluate_factual_accuracy("Paris is capital", "Paris is capital")
        expected_keys = ['factual_accuracy', 'fact_recall', 'fact_precision']
        for key in expected_keys:
            self.assertIn(key, result)
            
    def test_evaluate_coherence_returns_float(self):
        """Test that coherence evaluation returns a float."""
        result = self.evaluator.evaluate_coherence("This is a test. This is another test.")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)
        
    def test_evaluate_coherence_single_sentence(self):
        """Test coherence evaluation with single sentence."""
        result = self.evaluator.evaluate_coherence("This is a single sentence.")
        self.assertEqual(result, 1.0)
        
    def test_extract_key_facts(self):
        """Test key fact extraction."""
        facts = self.evaluator._extract_key_facts("Paris is the capital of France.")
        self.assertIsInstance(facts, set)
        
    def test_sentence_similarity(self):
        """Test sentence similarity calculation."""
        sim = self.evaluator._sentence_similarity("The cat sat", "The dog sat")
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, 0)
        self.assertLessEqual(sim, 1)


class TestEnhancedQualityMetrics(unittest.TestCase):
    
    def setUp(self):
        # Mock all the sub-components to avoid external dependencies
        with patch('enhanced_quality_metrics.PerplexityCalculator') as mock_perp, \
             patch('enhanced_quality_metrics.SemanticSimilarityCalculator') as mock_sem, \
             patch('enhanced_quality_metrics.TaskSpecificEvaluator') as mock_task, \
             patch('enhanced_quality_metrics.nltk') as mock_nltk:
            
            mock_nltk.data.find.side_effect = [None, None]  # Simulate data exists
            
            self.metrics = EnhancedQualityMetrics()
            
    def test_calculate_all_metrics_returns_dict(self):
        """Test that calculate_all_metrics returns a dictionary."""
        result = self.metrics.calculate_all_metrics("reference", "candidate")
        self.assertIsInstance(result, dict)
        
    @patch('enhanced_quality_metrics.logger')
    def test_calculate_all_metrics_handles_errors(self, mock_logger):
        """Test that calculate_all_metrics handles errors gracefully."""
        # Make perplexity calculation raise an exception
        self.metrics.perplexity_calc.calculate_perplexity = MagicMock(side_effect=Exception("Test error"))
        
        result = self.metrics.calculate_all_metrics("reference", "candidate")
        self.assertIn('perplexity', result)
        self.assertEqual(result['perplexity'], float('inf'))
        

if __name__ == '__main__':
    unittest.main()