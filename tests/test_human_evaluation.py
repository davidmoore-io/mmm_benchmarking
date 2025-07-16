import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
from datetime import datetime

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human_evaluation import (
    EvaluationCriteria, EvaluationTask, EvaluationRating, EvaluationSession,
    EvaluationDatabase, HumanEvaluator, EvaluationInterface
)


class TestEvaluationDataClasses(unittest.TestCase):
    
    def test_evaluation_task_creation(self):
        """Test EvaluationTask creation."""
        task = EvaluationTask(
            id="test-id",
            query="test query",
            reference_text="reference",
            candidate_text="candidate",
            api_name="OpenAI",
            model_name="gpt-3.5-turbo",
            created_at=datetime.now()
        )
        
        self.assertEqual(task.id, "test-id")
        self.assertEqual(task.query, "test query")
        self.assertEqual(task.api_name, "OpenAI")
        self.assertIsNone(task.completed_at)
        self.assertIsNone(task.evaluator_id)
    
    def test_evaluation_rating_creation(self):
        """Test EvaluationRating creation."""
        rating = EvaluationRating(
            id="rating-id",
            task_id="task-id",
            evaluator_id="evaluator-id",
            criterion=EvaluationCriteria.RELEVANCE,
            score=4,
            comment="Good response"
        )
        
        self.assertEqual(rating.criterion, EvaluationCriteria.RELEVANCE)
        self.assertEqual(rating.score, 4)
        self.assertEqual(rating.comment, "Good response")
        self.assertIsInstance(rating.created_at, datetime)
    
    def test_evaluation_session_creation(self):
        """Test EvaluationSession creation."""
        session = EvaluationSession(
            id="session-id",
            evaluator_id="evaluator-id",
            started_at=datetime.now()
        )
        
        self.assertEqual(session.id, "session-id")
        self.assertEqual(session.total_tasks, 0)
        self.assertEqual(session.completed_tasks, 0)
        self.assertIsNone(session.completed_at)


class TestEvaluationDatabase(unittest.TestCase):
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db = EvaluationDatabase(self.temp_db.name)
        
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Database should be created and tables should exist
        import sqlite3
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('evaluation_tasks', 'evaluation_ratings', 'evaluation_sessions')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
        self.assertIn('evaluation_tasks', tables)
        self.assertIn('evaluation_ratings', tables)
        self.assertIn('evaluation_sessions', tables)
    
    def test_save_task(self):
        """Test saving an evaluation task."""
        task = EvaluationTask(
            id="test-task",
            query="test query",
            reference_text="reference",
            candidate_text="candidate",
            api_name="OpenAI",
            model_name="gpt-3.5-turbo",
            created_at=datetime.now()
        )
        
        result = self.db.save_task(task)
        self.assertTrue(result)
    
    def test_save_rating(self):
        """Test saving an evaluation rating."""
        # First save a task
        task = EvaluationTask(
            id="test-task",
            query="test query",
            reference_text="reference",
            candidate_text="candidate",
            api_name="OpenAI",
            model_name="gpt-3.5-turbo",
            created_at=datetime.now()
        )
        self.db.save_task(task)
        
        # Then save a rating
        rating = EvaluationRating(
            id="test-rating",
            task_id="test-task",
            evaluator_id="evaluator-1",
            criterion=EvaluationCriteria.RELEVANCE,
            score=4
        )
        
        result = self.db.save_rating(rating)
        self.assertTrue(result)
    
    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        # Save a task
        task = EvaluationTask(
            id="test-task",
            query="test query",
            reference_text="reference",
            candidate_text="candidate",
            api_name="OpenAI",
            model_name="gpt-3.5-turbo",
            created_at=datetime.now()
        )
        self.db.save_task(task)
        
        # Get pending tasks
        tasks = self.db.get_pending_tasks("evaluator-1")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, "test-task")
    
    def test_get_evaluation_results(self):
        """Test getting evaluation results."""
        # Save a task and rating
        task = EvaluationTask(
            id="test-task",
            query="test query",
            reference_text="reference",
            candidate_text="candidate",
            api_name="OpenAI",
            model_name="gpt-3.5-turbo",
            created_at=datetime.now()
        )
        self.db.save_task(task)
        
        rating = EvaluationRating(
            id="test-rating",
            task_id="test-task",
            evaluator_id="evaluator-1",
            criterion=EvaluationCriteria.RELEVANCE,
            score=4
        )
        self.db.save_rating(rating)
        
        # Get results
        results = self.db.get_evaluation_results("test-task")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 4)


class TestHumanEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test evaluator."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.evaluator = HumanEvaluator(self.temp_db.name)
        
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_create_evaluation_tasks(self):
        """Test creating evaluation tasks from benchmark results."""
        benchmark_results = [
            {
                'query': 'Test query',
                'reference': 'Reference text',
                'response': 'Model response',
                'api_name': 'OpenAI',
                'model_name': 'gpt-3.5-turbo'
            }
        ]
        
        task_ids = self.evaluator.create_evaluation_tasks(benchmark_results)
        self.assertEqual(len(task_ids), 1)
        self.assertIsInstance(task_ids[0], str)
    
    def test_start_evaluation_session(self):
        """Test starting an evaluation session."""
        session_id = self.evaluator.start_evaluation_session("evaluator-1")
        self.assertIsInstance(session_id, str)
        self.assertIsNotNone(self.evaluator.current_session)
    
    def test_submit_evaluation(self):
        """Test submitting an evaluation."""
        # Create a task first
        benchmark_results = [
            {
                'query': 'Test query',
                'reference': 'Reference text',
                'response': 'Model response',
                'api_name': 'OpenAI',
                'model_name': 'gpt-3.5-turbo'
            }
        ]
        task_ids = self.evaluator.create_evaluation_tasks(benchmark_results)
        task_id = task_ids[0]
        
        # Submit evaluation
        ratings = {
            EvaluationCriteria.RELEVANCE: (4, "Good response"),
            EvaluationCriteria.COHERENCE: (3, "Okay coherence")
        }
        
        result = self.evaluator.submit_evaluation(task_id, "evaluator-1", ratings)
        self.assertTrue(result)
    
    def test_get_evaluation_statistics(self):
        """Test getting evaluation statistics."""
        stats = self.evaluator.get_evaluation_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_tasks', stats)
        self.assertIn('completed_tasks', stats)
        self.assertIn('completion_rate', stats)
    
    @patch('builtins.open', create=True)
    def test_export_evaluation_results(self, mock_open):
        """Test exporting evaluation results."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = self.evaluator.export_evaluation_results("test_output.json")
        self.assertTrue(result)
        mock_open.assert_called_once_with("test_output.json", 'w')


class TestEvaluationInterface(unittest.TestCase):
    
    def setUp(self):
        """Set up test interface."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.evaluator = HumanEvaluator(self.temp_db.name)
        self.interface = EvaluationInterface(self.evaluator)
        
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_interface_creation(self):
        """Test interface creation."""
        self.assertIsInstance(self.interface, EvaluationInterface)
        self.assertEqual(self.interface.evaluator, self.evaluator)
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_collect_ratings(self, mock_print, mock_input):
        """Test collecting ratings from user input."""
        # Mock user inputs for each criterion
        mock_input.side_effect = [
            '4', 'Good relevance',  # RELEVANCE
            '3', 'Okay coherence',  # COHERENCE
            '5', 'Very accurate',   # ACCURACY
            '4', 'Complete enough', # COMPLETENESS
            '3', 'Clear enough',    # CLARITY
            '2', 'Not creative',    # CREATIVITY
            '4', 'Helpful'          # HELPFULNESS
        ]
        
        ratings = self.interface._collect_ratings()
        
        self.assertEqual(len(ratings), 7)  # All criteria
        self.assertEqual(ratings[EvaluationCriteria.RELEVANCE][0], 4)
        self.assertEqual(ratings[EvaluationCriteria.RELEVANCE][1], 'Good relevance')


if __name__ == '__main__':
    unittest.main()