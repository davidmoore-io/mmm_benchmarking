"""
A/B Testing Framework

This module implements a comprehensive A/B testing framework for comparing
model performance based on user preferences and controlled experiments.
"""

import json
import uuid
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparisonType(Enum):
    """Types of comparisons in A/B testing."""
    PAIRWISE = "pairwise"
    RANKING = "ranking"
    RATING = "rating"


class PreferenceChoice(Enum):
    """User preference choices."""
    RESPONSE_A = "response_a"
    RESPONSE_B = "response_b"
    NO_PREFERENCE = "no_preference"


@dataclass
class ABTestResponse:
    """Represents a response in an A/B test."""
    id: str
    api_name: str
    model_name: str
    response_text: str
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ABTestTask:
    """Represents an A/B testing task."""
    id: str
    query: str
    reference_text: str
    response_a: ABTestResponse
    response_b: ABTestResponse
    comparison_type: ComparisonType
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ABTestResult:
    """Represents the result of an A/B test."""
    id: str
    task_id: str
    evaluator_id: str
    preference: PreferenceChoice
    confidence: int  # 1-5 scale
    reasoning: Optional[str] = None
    response_time: Optional[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ABTestExperiment:
    """Represents an A/B testing experiment."""
    id: str
    name: str
    description: str
    models_under_test: List[Tuple[str, str]]  # (api_name, model_name) pairs
    total_tasks: int
    completed_tasks: int = 0
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ABTestDatabase:
    """Database manager for A/B testing data."""
    
    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the A/B testing database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    models_under_test TEXT,
                    total_tasks INTEGER,
                    completed_tasks INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            
            # Create tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_tasks (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    query TEXT NOT NULL,
                    reference_text TEXT,
                    response_a_id TEXT NOT NULL,
                    response_b_id TEXT NOT NULL,
                    comparison_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES ab_experiments (id)
                )
            """)
            
            # Create responses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_responses (
                    id TEXT PRIMARY KEY,
                    api_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_results (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    reasoning TEXT,
                    response_time REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES ab_tasks (id)
                )
            """)
            
            conn.commit()
    
    def save_experiment(self, experiment: ABTestExperiment) -> bool:
        """Save an A/B test experiment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ab_experiments 
                    (id, name, description, models_under_test, total_tasks, completed_tasks, created_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment.id, experiment.name, experiment.description,
                    json.dumps(experiment.models_under_test), experiment.total_tasks,
                    experiment.completed_tasks, experiment.created_at.isoformat(),
                    experiment.completed_at.isoformat() if experiment.completed_at else None
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving experiment: {e}")
            return False
    
    def save_response(self, response: ABTestResponse) -> bool:
        """Save an A/B test response."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ab_responses 
                    (id, api_name, model_name, response_text, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    response.id, response.api_name, response.model_name,
                    response.response_text, json.dumps(response.metadata),
                    response.created_at.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving response: {e}")
            return False
    
    def save_task(self, task: ABTestTask, experiment_id: str) -> bool:
        """Save an A/B test task."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ab_tasks 
                    (id, experiment_id, query, reference_text, response_a_id, response_b_id, comparison_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id, experiment_id, task.query, task.reference_text,
                    task.response_a.id, task.response_b.id,
                    task.comparison_type.value, task.created_at.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving task: {e}")
            return False
    
    def save_result(self, result: ABTestResult) -> bool:
        """Save an A/B test result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ab_results 
                    (id, task_id, evaluator_id, preference, confidence, reasoning, response_time, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.id, result.task_id, result.evaluator_id,
                    result.preference.value, result.confidence,
                    result.reasoning, result.response_time,
                    result.created_at.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False
    
    def get_pending_tasks(self, experiment_id: str, evaluator_id: str, limit: int = 10) -> List[ABTestTask]:
        """Get pending A/B test tasks for an experiment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get tasks that haven't been completed by this evaluator
                cursor.execute("""
                    SELECT t.*, ra.api_name as a_api, ra.model_name as a_model, ra.response_text as a_text, ra.metadata as a_meta,
                           rb.api_name as b_api, rb.model_name as b_model, rb.response_text as b_text, rb.metadata as b_meta
                    FROM ab_tasks t
                    JOIN ab_responses ra ON t.response_a_id = ra.id
                    JOIN ab_responses rb ON t.response_b_id = rb.id
                    LEFT JOIN ab_results r ON t.id = r.task_id AND r.evaluator_id = ?
                    WHERE t.experiment_id = ? AND r.id IS NULL
                    LIMIT ?
                """, (evaluator_id, experiment_id, limit))
                
                tasks = []
                for row in cursor.fetchall():
                    # Create response objects
                    response_a = ABTestResponse(
                        id=row[4],
                        api_name=row[8],
                        model_name=row[9],
                        response_text=row[10],
                        metadata=json.loads(row[11])
                    )
                    
                    response_b = ABTestResponse(
                        id=row[5],
                        api_name=row[12],
                        model_name=row[13],
                        response_text=row[14],
                        metadata=json.loads(row[15])
                    )
                    
                    task = ABTestTask(
                        id=row[0],
                        query=row[2],
                        reference_text=row[3],
                        response_a=response_a,
                        response_b=response_b,
                        comparison_type=ComparisonType(row[6]),
                        created_at=datetime.fromisoformat(row[7])
                    )
                    tasks.append(task)
                
                return tasks
        except Exception as e:
            logger.error(f"Error getting pending tasks: {e}")
            return []
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for an A/B test experiment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all results for the experiment
                cursor.execute("""
                    SELECT r.*, t.query, ra.api_name as a_api, ra.model_name as a_model,
                           rb.api_name as b_api, rb.model_name as b_model
                    FROM ab_results r
                    JOIN ab_tasks t ON r.task_id = t.id
                    JOIN ab_responses ra ON t.response_a_id = ra.id
                    JOIN ab_responses rb ON t.response_b_id = rb.id
                    WHERE t.experiment_id = ?
                """, (experiment_id,))
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'result_id': row[0],
                        'task_id': row[1],
                        'evaluator_id': row[2],
                        'preference': row[3],
                        'confidence': row[4],
                        'reasoning': row[5],
                        'response_time': row[6],
                        'created_at': row[7],
                        'query': row[8],
                        'model_a': f"{row[9]} ({row[10]})",
                        'model_b': f"{row[11]} ({row[12]})"
                    }
                    results.append(result)
                
                return {'results': results}
        except Exception as e:
            logger.error(f"Error getting experiment results: {e}")
            return {'results': []}


class ABTestManager:
    """Main class for managing A/B testing experiments."""
    
    def __init__(self, db_path: str = "ab_testing.db"):
        self.db = ABTestDatabase(db_path)
    
    def create_experiment(self, name: str, description: str, 
                         models_under_test: List[Tuple[str, str]]) -> str:
        """Create a new A/B testing experiment."""
        experiment_id = str(uuid.uuid4())
        experiment = ABTestExperiment(
            id=experiment_id,
            name=name,
            description=description,
            models_under_test=models_under_test,
            total_tasks=0
        )
        
        if self.db.save_experiment(experiment):
            logger.info(f"Created A/B test experiment: {experiment_id}")
            return experiment_id
        else:
            logger.error("Failed to create experiment")
            return None
    
    def add_comparison_tasks(self, experiment_id: str, 
                           benchmark_results: List[Dict[str, Any]],
                           comparison_type: ComparisonType = ComparisonType.PAIRWISE) -> List[str]:
        """
        Add comparison tasks to an experiment from benchmark results.
        
        Args:
            experiment_id: ID of the experiment
            benchmark_results: List of benchmark results with responses from different models
            comparison_type: Type of comparison to perform
            
        Returns:
            List of created task IDs
        """
        task_ids = []
        
        # Group results by query
        query_groups = defaultdict(list)
        for result in benchmark_results:
            query = result.get('query', '')
            query_groups[query].append(result)
        
        # Create comparison tasks for each query
        for query, results in query_groups.items():
            if len(results) < 2:
                continue
            
            # Create all possible pairwise comparisons
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    result_a = results[i]
                    result_b = results[j]
                    
                    # Create response objects
                    response_a = ABTestResponse(
                        id=str(uuid.uuid4()),
                        api_name=result_a.get('api_name', ''),
                        model_name=result_a.get('model_name', ''),
                        response_text=result_a.get('response', ''),
                        metadata=result_a.get('metadata', {})
                    )
                    
                    response_b = ABTestResponse(
                        id=str(uuid.uuid4()),
                        api_name=result_b.get('api_name', ''),
                        model_name=result_b.get('model_name', ''),
                        response_text=result_b.get('response', ''),
                        metadata=result_b.get('metadata', {})
                    )
                    
                    # Save responses
                    self.db.save_response(response_a)
                    self.db.save_response(response_b)
                    
                    # Randomize order to avoid bias
                    if random.random() < 0.5:
                        response_a, response_b = response_b, response_a
                    
                    # Create task
                    task = ABTestTask(
                        id=str(uuid.uuid4()),
                        query=query,
                        reference_text=result_a.get('reference', ''),
                        response_a=response_a,
                        response_b=response_b,
                        comparison_type=comparison_type
                    )
                    
                    if self.db.save_task(task, experiment_id):
                        task_ids.append(task.id)
        
        # Update experiment task count
        self._update_experiment_task_count(experiment_id, len(task_ids))
        
        logger.info(f"Added {len(task_ids)} comparison tasks to experiment {experiment_id}")
        return task_ids
    
    def submit_comparison_result(self, task_id: str, evaluator_id: str,
                               preference: PreferenceChoice, confidence: int,
                               reasoning: str = None, response_time: float = None) -> bool:
        """Submit a comparison result."""
        result = ABTestResult(
            id=str(uuid.uuid4()),
            task_id=task_id,
            evaluator_id=evaluator_id,
            preference=preference,
            confidence=confidence,
            reasoning=reasoning,
            response_time=response_time
        )
        
        if self.db.save_result(result):
            logger.info(f"Submitted comparison result for task: {task_id}")
            return True
        else:
            logger.error(f"Failed to submit result for task: {task_id}")
            return False
    
    def get_experiment_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """Get statistics for an A/B test experiment."""
        results = self.db.get_experiment_results(experiment_id)
        
        if not results['results']:
            return {'total_comparisons': 0, 'model_win_rates': {}}
        
        # Calculate win rates
        model_wins = defaultdict(int)
        total_comparisons = len(results['results'])
        
        for result in results['results']:
            preference = result['preference']
            if preference == PreferenceChoice.RESPONSE_A.value:
                model_wins[result['model_a']] += 1
            elif preference == PreferenceChoice.RESPONSE_B.value:
                model_wins[result['model_b']] += 1
        
        # Calculate win rates
        win_rates = {}
        for model, wins in model_wins.items():
            win_rates[model] = wins / total_comparisons
        
        # Calculate confidence statistics
        confidence_scores = [r['confidence'] for r in results['results']]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_comparisons': total_comparisons,
            'model_win_rates': win_rates,
            'average_confidence': avg_confidence,
            'preference_distribution': {
                'response_a': sum(1 for r in results['results'] if r['preference'] == PreferenceChoice.RESPONSE_A.value),
                'response_b': sum(1 for r in results['results'] if r['preference'] == PreferenceChoice.RESPONSE_B.value),
                'no_preference': sum(1 for r in results['results'] if r['preference'] == PreferenceChoice.NO_PREFERENCE.value)
            }
        }
    
    def _update_experiment_task_count(self, experiment_id: str, task_count: int):
        """Update the task count for an experiment."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ab_experiments 
                    SET total_tasks = total_tasks + ?
                    WHERE id = ?
                """, (task_count, experiment_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating experiment task count: {e}")
    
    def export_experiment_results(self, experiment_id: str, output_path: str) -> bool:
        """Export experiment results to JSON file."""
        try:
            results = self.db.get_experiment_results(experiment_id)
            statistics = self.get_experiment_statistics(experiment_id)
            
            export_data = {
                'experiment_id': experiment_id,
                'statistics': statistics,
                'raw_results': results['results'],
                'exported_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported experiment results to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting experiment results: {e}")
            return False


class ABTestInterface:
    """Command-line interface for A/B testing."""
    
    def __init__(self, manager: ABTestManager):
        self.manager = manager
        from utils import (
            print_header, print_section, print_success, print_warning, print_error,
            get_user_input, get_integer_input, get_confirmation, console
        )
        self.print_header = print_header
        self.print_section = print_section
        self.print_success = print_success
        self.print_warning = print_warning
        self.print_error = print_error
        self.get_user_input = get_user_input
        self.get_integer_input = get_integer_input
        self.get_confirmation = get_confirmation
        self.console = console
    
    def run_comparison_session(self, experiment_id: str, evaluator_id: str):
        """Run an interactive comparison session."""
        self.print_header("🆚 A/B Testing Session", f"Experiment: {experiment_id} | Evaluator: {evaluator_id}")
        
        comparison_count = 0
        while True:
            # Get next task
            tasks = self.manager.db.get_pending_tasks(experiment_id, evaluator_id, limit=1)
            if not tasks:
                self.print_warning("No more comparison tasks available.")
                break
            
            task = tasks[0]
            comparison_count += 1
            
            # Display comparison
            self._display_comparison(task, comparison_count)
            
            # Collect preference
            preference, confidence, reasoning = self._collect_preference()
            
            # Submit result
            if self.manager.submit_comparison_result(
                task.id, evaluator_id, preference, confidence, reasoning
            ):
                self.print_success("✅ Comparison result submitted successfully!")
            else:
                self.print_error("❌ Failed to submit comparison result.")
            
            # Ask if user wants to continue
            if not self.get_confirmation("Continue with next comparison?", default=True):
                break
        
        self.print_header("🏁 Comparison Session Complete", f"Comparisons completed: {comparison_count}")
    
    def _display_comparison(self, task: ABTestTask, comparison_number: int):
        """Display a comparison task."""
        from rich.panel import Panel
        from rich.columns import Columns
        
        self.print_section(f"📊 Comparison {comparison_number}")
        
        # Query
        query_panel = Panel(task.query, title="❓ Query", border_style="blue")
        self.console.print(query_panel)
        
        # Responses side by side
        response_a_panel = Panel(
            task.response_a.response_text,
            title=f"🅰️ Response A ({task.response_a.api_name} - {task.response_a.model_name})",
            border_style="green"
        )
        
        response_b_panel = Panel(
            task.response_b.response_text,
            title=f"🅱️ Response B ({task.response_b.api_name} - {task.response_b.model_name})",
            border_style="red"
        )
        
        # Display responses in columns
        columns = Columns([response_a_panel, response_b_panel], equal=True)
        self.console.print(columns)
        
        # Reference answer if available
        if task.reference_text:
            ref_panel = Panel(task.reference_text, title="✅ Reference Answer", border_style="cyan")
            self.console.print(ref_panel)
    
    def _collect_preference(self) -> Tuple[PreferenceChoice, int, str]:
        """Collect user preference."""
        from rich.panel import Panel
        
        self.print_section("🤔 Which response do you prefer?")
        
        # Show options
        choice_info = """
        🅰️ Response A
        🅱️ Response B  
        ⚖️ No preference
        """
        self.console.print(Panel(choice_info, title="📊 Preference Options", border_style="cyan"))
        
        # Get preference
        choice = self.get_integer_input("Your choice (1-3)", minimum=1, maximum=3)
        if choice == 1:
            preference = PreferenceChoice.RESPONSE_A
        elif choice == 2:
            preference = PreferenceChoice.RESPONSE_B
        elif choice == 3:
            preference = PreferenceChoice.NO_PREFERENCE
        else:
            preference = PreferenceChoice.NO_PREFERENCE  # Default
        
        # Get confidence
        confidence = self.get_integer_input("🎯 Confidence level (1-5)", minimum=1, maximum=5)
        if confidence is None:
            confidence = 3  # Default to medium confidence
        
        # Get reasoning
        reasoning = self.get_user_input("💭 Reasoning (optional, press Enter to skip)")
        
        return preference, confidence, reasoning if reasoning else None