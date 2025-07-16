import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
from datetime import datetime

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ab_testing import (
    ComparisonType, PreferenceChoice, ABTestResponse, ABTestTask,
    ABTestResult, ABTestExperiment, ABTestDatabase, ABTestManager,
    ABTestInterface
)


class TestABTestDataClasses(unittest.TestCase):
    
    def test_ab_test_response_creation(self):
        """Test ABTestResponse creation."""
        response = ABTestResponse(
            id="response-id",
            api_name="OpenAI",
            model_name="gpt-3.5-turbo",
            response_text="This is a response",
            metadata={"score": 0.8}
        )
        
        self.assertEqual(response.id, "response-id")
        self.assertEqual(response.api_name, "OpenAI")
        self.assertEqual(response.model_name, "gpt-3.5-turbo")
        self.assertIsInstance(response.created_at, datetime)
    
    def test_ab_test_task_creation(self):
        """Test ABTestTask creation."""
        response_a = ABTestResponse(
            id="response-a", api_name="OpenAI", model_name="gpt-3.5-turbo",
            response_text="Response A", metadata={}
        )
        response_b = ABTestResponse(
            id="response-b", api_name="Anthropic", model_name="claude-2",
            response_text="Response B", metadata={}
        )
        
        task = ABTestTask(
            id="task-id",
            query="Test query",
            reference_text="Reference",
            response_a=response_a,
            response_b=response_b,
            comparison_type=ComparisonType.PAIRWISE
        )
        
        self.assertEqual(task.id, "task-id")
        self.assertEqual(task.query, "Test query")
        self.assertEqual(task.comparison_type, ComparisonType.PAIRWISE)
        self.assertIsInstance(task.created_at, datetime)
    
    def test_ab_test_result_creation(self):
        """Test ABTestResult creation."""
        result = ABTestResult(
            id="result-id",
            task_id="task-id",
            evaluator_id="evaluator-id",
            preference=PreferenceChoice.RESPONSE_A,
            confidence=4,
            reasoning="Response A is better"
        )
        
        self.assertEqual(result.preference, PreferenceChoice.RESPONSE_A)
        self.assertEqual(result.confidence, 4)
        self.assertEqual(result.reasoning, "Response A is better")
        self.assertIsInstance(result.created_at, datetime)
    
    def test_ab_test_experiment_creation(self):
        """Test ABTestExperiment creation."""
        experiment = ABTestExperiment(
            id="experiment-id",
            name="Test Experiment",
            description="Testing models",
            models_under_test=[("OpenAI", "gpt-3.5-turbo"), ("Anthropic", "claude-2")],
            total_tasks=10
        )
        
        self.assertEqual(experiment.name, "Test Experiment")
        self.assertEqual(experiment.total_tasks, 10)
        self.assertEqual(experiment.completed_tasks, 0)
        self.assertIsInstance(experiment.created_at, datetime)


class TestABTestDatabase(unittest.TestCase):
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db = ABTestDatabase(self.temp_db.name)
        
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        import sqlite3
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('ab_experiments', 'ab_tasks', 'ab_responses', 'ab_results')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
        expected_tables = ['ab_experiments', 'ab_tasks', 'ab_responses', 'ab_results']
        for table in expected_tables:
            self.assertIn(table, tables)
    
    def test_save_experiment(self):
        """Test saving an experiment."""
        experiment = ABTestExperiment(
            id="test-experiment",
            name="Test Experiment",
            description="Test description",
            models_under_test=[("OpenAI", "gpt-3.5-turbo")],
            total_tasks=5
        )
        
        result = self.db.save_experiment(experiment)
        self.assertTrue(result)
    
    def test_save_response(self):
        """Test saving a response."""
        response = ABTestResponse(
            id="test-response",
            api_name="OpenAI",
            model_name="gpt-3.5-turbo",
            response_text="Test response",
            metadata={"score": 0.8}
        )
        
        result = self.db.save_response(response)
        self.assertTrue(result)
    
    def test_save_task(self):
        """Test saving a task."""
        # First save responses
        response_a = ABTestResponse(
            id="response-a", api_name="OpenAI", model_name="gpt-3.5-turbo",
            response_text="Response A", metadata={}
        )
        response_b = ABTestResponse(
            id="response-b", api_name="Anthropic", model_name="claude-2",
            response_text="Response B", metadata={}
        )
        
        self.db.save_response(response_a)
        self.db.save_response(response_b)
        
        # Then save task
        task = ABTestTask(
            id="test-task",
            query="Test query",
            reference_text="Reference",
            response_a=response_a,
            response_b=response_b,
            comparison_type=ComparisonType.PAIRWISE
        )
        
        result = self.db.save_task(task, "experiment-id")
        self.assertTrue(result)
    
    def test_save_result(self):
        """Test saving a result."""
        result = ABTestResult(
            id="test-result",
            task_id="test-task",
            evaluator_id="evaluator-1",
            preference=PreferenceChoice.RESPONSE_A,
            confidence=4
        )
        
        save_result = self.db.save_result(result)
        self.assertTrue(save_result)
    
    def test_get_experiment_results(self):
        """Test getting experiment results."""
        results = self.db.get_experiment_results("nonexistent-experiment")
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIsInstance(results['results'], list)


class TestABTestManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test manager."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.manager = ABTestManager(self.temp_db.name)
        
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_create_experiment(self):
        """Test creating an experiment."""
        experiment_id = self.manager.create_experiment(
            name="Test Experiment",
            description="Test description",
            models_under_test=[("OpenAI", "gpt-3.5-turbo"), ("Anthropic", "claude-2")]
        )
        
        self.assertIsNotNone(experiment_id)
        self.assertIsInstance(experiment_id, str)
    
    def test_add_comparison_tasks(self):
        """Test adding comparison tasks."""
        # Create experiment first
        experiment_id = self.manager.create_experiment(
            name="Test Experiment",
            description="Test description",
            models_under_test=[("OpenAI", "gpt-3.5-turbo"), ("Anthropic", "claude-2")]
        )
        
        # Add tasks
        benchmark_results = [
            {
                'query': 'Test query',
                'reference': 'Reference text',
                'response': 'Response from OpenAI',
                'api_name': 'OpenAI',
                'model_name': 'gpt-3.5-turbo',
                'metadata': {}
            },
            {
                'query': 'Test query',
                'reference': 'Reference text',
                'response': 'Response from Anthropic',
                'api_name': 'Anthropic',
                'model_name': 'claude-2',
                'metadata': {}
            }
        ]
        
        task_ids = self.manager.add_comparison_tasks(experiment_id, benchmark_results)
        self.assertGreater(len(task_ids), 0)
    
    def test_submit_comparison_result(self):
        """Test submitting a comparison result."""
        # This test would need a task to exist first
        # For now, just test that the method exists and handles errors
        result = self.manager.submit_comparison_result(
            "nonexistent-task", "evaluator-1", 
            PreferenceChoice.RESPONSE_A, 4, "Test reasoning"
        )
        self.assertIsInstance(result, bool)
    
    def test_get_experiment_statistics(self):
        """Test getting experiment statistics."""
        stats = self.manager.get_experiment_statistics("nonexistent-experiment")
        self.assertIsInstance(stats, dict)
        self.assertIn('total_comparisons', stats)
        self.assertIn('model_win_rates', stats)
    
    @patch('builtins.open', create=True)
    def test_export_experiment_results(self, mock_open):
        """Test exporting experiment results."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = self.manager.export_experiment_results("experiment-id", "output.json")
        self.assertTrue(result)
        mock_open.assert_called_once_with("output.json", 'w')


class TestABTestInterface(unittest.TestCase):
    
    def setUp(self):
        """Set up test interface."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.manager = ABTestManager(self.temp_db.name)
        self.interface = ABTestInterface(self.manager)
        
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_interface_creation(self):
        """Test interface creation."""
        self.assertIsInstance(self.interface, ABTestInterface)
        self.assertEqual(self.interface.manager, self.manager)
    
    @patch('builtins.input')
    def test_collect_preference(self, mock_input):
        """Test collecting user preference."""
        mock_input.side_effect = ['1', '4', 'Response A is better']
        
        preference, confidence, reasoning = self.interface._collect_preference()
        
        self.assertEqual(preference, PreferenceChoice.RESPONSE_A)
        self.assertEqual(confidence, 4)
        self.assertEqual(reasoning, 'Response A is better')
    
    @patch('builtins.input')
    def test_collect_preference_no_reasoning(self, mock_input):
        """Test collecting user preference without reasoning."""
        mock_input.side_effect = ['2', '3', '']
        
        preference, confidence, reasoning = self.interface._collect_preference()
        
        self.assertEqual(preference, PreferenceChoice.RESPONSE_B)
        self.assertEqual(confidence, 3)
        self.assertIsNone(reasoning)


if __name__ == '__main__':
    unittest.main()