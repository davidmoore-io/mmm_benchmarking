import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import json

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced benchmarking system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_modules_import_successfully(self):
        """Test that all main modules can be imported."""
        try:
            from quality_metrics import EnhancedQualityMetrics
            from human_evaluation import HumanEvaluator
            from ab_testing import ABTestManager
            
            # Test instantiation
            metrics = EnhancedQualityMetrics()
            evaluator = HumanEvaluator()
            ab_manager = ABTestManager()
            
            self.assertIsNotNone(metrics)
            self.assertIsNotNone(evaluator)
            self.assertIsNotNone(ab_manager)
            
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")
    
    @patch('quality_metrics.SentenceTransformer')
    def test_enhanced_metrics_workflow(self, mock_transformer):
        """Test the enhanced metrics workflow."""
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model
        
        from quality_metrics import EnhancedQualityMetrics
        
        metrics = EnhancedQualityMetrics()
        results = metrics.calculate_all_metrics(
            "Paris is the capital of France.",
            "The capital of France is Paris."
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'perplexity', 'cosine_similarity', 'euclidean_similarity',
            'manhattan_similarity', 'semantic_textual_similarity',
            'factual_accuracy', 'fact_recall', 'fact_precision', 'coherence'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (int, float))
    
    def test_human_evaluation_workflow(self):
        """Test the human evaluation workflow."""
        from human_evaluation import HumanEvaluator, EvaluationCriteria
        
        # Use temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        evaluator = HumanEvaluator(temp_db.name)
        
        # Test creating evaluation tasks
        benchmark_results = [
            {
                'query': 'Test query',
                'reference': 'Reference text',
                'response': 'Model response',
                'api_name': 'TestAPI',
                'model_name': 'test-model'
            }
        ]
        
        task_ids = evaluator.create_evaluation_tasks(benchmark_results)
        self.assertGreater(len(task_ids), 0)
        
        # Test submitting evaluation
        ratings = {
            EvaluationCriteria.RELEVANCE: (4, "Good response"),
            EvaluationCriteria.COHERENCE: (3, "Decent coherence")
        }
        
        result = evaluator.submit_evaluation(task_ids[0], "test_evaluator", ratings)
        self.assertTrue(result)
        
        # Test statistics
        stats = evaluator.get_evaluation_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_tasks', stats)
        
        # Clean up
        os.unlink(temp_db.name)
    
    def test_ab_testing_workflow(self):
        """Test the A/B testing workflow."""
        from ab_testing import ABTestManager, PreferenceChoice
        
        # Use temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        manager = ABTestManager(temp_db.name)
        
        # Create experiment
        experiment_id = manager.create_experiment(
            "Test Experiment",
            "Test description",
            [("API1", "model1"), ("API2", "model2")]
        )
        self.assertIsNotNone(experiment_id)
        
        # Add comparison tasks
        benchmark_results = [
            {
                'query': 'Test query',
                'reference': 'Reference text',
                'response': 'Response from API1',
                'api_name': 'API1',
                'model_name': 'model1'
            },
            {
                'query': 'Test query',
                'reference': 'Reference text',
                'response': 'Response from API2',
                'api_name': 'API2',
                'model_name': 'model2'
            }
        ]
        
        task_ids = manager.add_comparison_tasks(experiment_id, benchmark_results)
        self.assertGreater(len(task_ids), 0)
        
        # Submit comparison result
        result = manager.submit_comparison_result(
            task_ids[0], "test_evaluator", 
            PreferenceChoice.RESPONSE_A, 4, "A is better"
        )
        self.assertTrue(result)
        
        # Get statistics
        stats = manager.get_experiment_statistics(experiment_id)
        self.assertIsInstance(stats, dict)
        self.assertIn('total_comparisons', stats)
        
        # Clean up
        os.unlink(temp_db.name)
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_files = [
            'config.yaml',
            'requirements.txt',
            '.env.sample',
            'setup.py',
            'health_check.py',
            'deploy.sh'
        ]
        
        for config_file in config_files:
            self.assertTrue(
                os.path.exists(config_file),
                f"Configuration file {config_file} does not exist"
            )
    
    def test_documentation_exists(self):
        """Test that documentation files exist."""
        doc_files = [
            'README.MD',
            'CLAUDE.md',
            'IMPLEMENTATION_GUIDE.md'
        ]
        
        for doc_file in doc_files:
            self.assertTrue(
                os.path.exists(doc_file),
                f"Documentation file {doc_file} does not exist"
            )
    
    def test_all_test_files_exist(self):
        """Test that all test files exist."""
        test_files = [
            'tests/test_quality_metrics.py',
            'tests/test_utils.py',
            'tests/test_enhanced_quality_metrics.py',
            'tests/test_human_evaluation.py',
            'tests/test_ab_testing.py',
            'tests/test_integration.py'
        ]
        
        for test_file in test_files:
            self.assertTrue(
                os.path.exists(test_file),
                f"Test file {test_file} does not exist"
            )


if __name__ == '__main__':
    unittest.main()