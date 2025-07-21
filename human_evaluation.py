"""
Human Evaluation System

This module implements a comprehensive human evaluation system for assessing
LLM response quality through manual review and rating.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationCriteria(Enum):
    """Criteria for evaluating LLM responses."""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    CREATIVITY = "creativity"
    HELPFULNESS = "helpfulness"


@dataclass
class EvaluationTask:
    """Represents a single evaluation task."""
    id: str
    query: str
    reference_text: str
    candidate_text: str
    api_name: str
    model_name: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    evaluator_id: Optional[str] = None


@dataclass
class EvaluationRating:
    """Represents a human evaluation rating."""
    id: str
    task_id: str
    evaluator_id: str
    criterion: EvaluationCriteria
    score: int  # 1-5 scale
    comment: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class EvaluationSession:
    """Represents an evaluation session."""
    id: str
    evaluator_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0


class EvaluationDatabase:
    """Database manager for human evaluation data."""
    
    def __init__(self, db_path: str = "evaluation.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the evaluation database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create evaluation_tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_tasks (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    reference_text TEXT NOT NULL,
                    candidate_text TEXT NOT NULL,
                    api_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    evaluator_id TEXT
                )
            """)
            
            # Create evaluation_ratings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_ratings (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL,
                    criterion TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES evaluation_tasks (id)
                )
            """)
            
            # Create evaluation_sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_sessions (
                    id TEXT PRIMARY KEY,
                    evaluator_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_tasks INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
    
    def save_task(self, task: EvaluationTask) -> bool:
        """Save an evaluation task to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO evaluation_tasks 
                    (id, query, reference_text, candidate_text, api_name, model_name, 
                     created_at, completed_at, evaluator_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id, task.query, task.reference_text, task.candidate_text,
                    task.api_name, task.model_name, task.created_at.isoformat(),
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.evaluator_id
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving task: {e}")
            return False
    
    def save_rating(self, rating: EvaluationRating) -> bool:
        """Save an evaluation rating to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO evaluation_ratings 
                    (id, task_id, evaluator_id, criterion, score, comment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    rating.id, rating.task_id, rating.evaluator_id,
                    rating.criterion.value, rating.score, rating.comment,
                    rating.created_at.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving rating: {e}")
            return False
    
    def get_pending_tasks(self, evaluator_id: str, limit: int = 10) -> List[EvaluationTask]:
        """Get pending evaluation tasks for an evaluator."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM evaluation_tasks 
                    WHERE (evaluator_id IS NULL OR evaluator_id = ?) 
                    AND completed_at IS NULL
                    LIMIT ?
                """, (evaluator_id, limit))
                
                tasks = []
                for row in cursor.fetchall():
                    task = EvaluationTask(
                        id=row[0],
                        query=row[1],
                        reference_text=row[2],
                        candidate_text=row[3],
                        api_name=row[4],
                        model_name=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        evaluator_id=row[8]
                    )
                    tasks.append(task)
                return tasks
        except Exception as e:
            logger.error(f"Error getting pending tasks: {e}")
            return []
    
    def get_evaluation_results(self, task_id: str) -> List[EvaluationRating]:
        """Get all evaluation results for a task."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM evaluation_ratings WHERE task_id = ?
                """, (task_id,))
                
                ratings = []
                for row in cursor.fetchall():
                    rating = EvaluationRating(
                        id=row[0],
                        task_id=row[1],
                        evaluator_id=row[2],
                        criterion=EvaluationCriteria(row[3]),
                        score=row[4],
                        comment=row[5],
                        created_at=datetime.fromisoformat(row[6])
                    )
                    ratings.append(rating)
                return ratings
        except Exception as e:
            logger.error(f"Error getting evaluation results: {e}")
            return []


class HumanEvaluator:
    """Main class for conducting human evaluation."""
    
    def __init__(self, db_path: str = "evaluation.db"):
        self.db = EvaluationDatabase(db_path)
        self.current_session = None
        
    def create_evaluation_tasks(self, benchmark_results: List[Dict[str, Any]]) -> List[str]:
        """
        Create evaluation tasks from benchmark results.
        
        Args:
            benchmark_results: List of benchmark result dictionaries
            
        Returns:
            List of created task IDs
        """
        task_ids = []
        
        for result in benchmark_results:
            task_id = str(uuid.uuid4())
            task = EvaluationTask(
                id=task_id,
                query=result.get('query', ''),
                reference_text=result.get('reference', ''),
                candidate_text=result.get('response', ''),
                api_name=result.get('api_name', ''),
                model_name=result.get('model_name', ''),
                created_at=datetime.now()
            )
            
            if self.db.save_task(task):
                task_ids.append(task_id)
                logger.info(f"Created evaluation task: {task_id}")
            else:
                logger.error(f"Failed to create task for {result.get('api_name', 'unknown')}")
        
        return task_ids
    
    def start_evaluation_session(self, evaluator_id: str) -> str:
        """Start a new evaluation session."""
        session_id = str(uuid.uuid4())
        self.current_session = EvaluationSession(
            id=session_id,
            evaluator_id=evaluator_id,
            started_at=datetime.now()
        )
        
        logger.info(f"Started evaluation session: {session_id}")
        return session_id
    
    def get_next_task(self, evaluator_id: str) -> Optional[EvaluationTask]:
        """Get the next task for evaluation."""
        tasks = self.db.get_pending_tasks(evaluator_id, limit=1)
        return tasks[0] if tasks else None
    
    def submit_evaluation(self, task_id: str, evaluator_id: str, 
                         ratings: Dict[EvaluationCriteria, Tuple[int, str]]) -> bool:
        """
        Submit evaluation ratings for a task.
        
        Args:
            task_id: ID of the evaluation task
            evaluator_id: ID of the evaluator
            ratings: Dictionary mapping criteria to (score, comment) tuples
            
        Returns:
            True if submission was successful
        """
        try:
            # Save all ratings
            for criterion, (score, comment) in ratings.items():
                rating = EvaluationRating(
                    id=str(uuid.uuid4()),
                    task_id=task_id,
                    evaluator_id=evaluator_id,
                    criterion=criterion,
                    score=score,
                    comment=comment
                )
                
                if not self.db.save_rating(rating):
                    logger.error(f"Failed to save rating for criterion: {criterion}")
                    return False
            
            # Mark task as completed
            self._mark_task_completed(task_id, evaluator_id)
            
            logger.info(f"Submitted evaluation for task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting evaluation: {e}")
            return False
    
    def _mark_task_completed(self, task_id: str, evaluator_id: str):
        """Mark a task as completed."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE evaluation_tasks 
                    SET completed_at = ?, evaluator_id = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), evaluator_id, task_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error marking task as completed: {e}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluations."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total tasks
                cursor.execute("SELECT COUNT(*) FROM evaluation_tasks")
                total_tasks = cursor.fetchone()[0]
                
                # Get completed tasks
                cursor.execute("SELECT COUNT(*) FROM evaluation_tasks WHERE completed_at IS NOT NULL")
                completed_tasks = cursor.fetchone()[0]
                
                # Get average scores by criterion
                cursor.execute("""
                    SELECT criterion, AVG(score) as avg_score, COUNT(*) as count
                    FROM evaluation_ratings 
                    GROUP BY criterion
                """)
                
                criterion_stats = {}
                for row in cursor.fetchall():
                    criterion_stats[row[0]] = {
                        'average_score': row[1],
                        'count': row[2]
                    }
                
                return {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
                    'criterion_statistics': criterion_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting evaluation statistics: {e}")
            return {}
    
    def export_evaluation_results(self, output_path: str) -> bool:
        """Export evaluation results to JSON file."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all completed tasks with their ratings
                cursor.execute("""
                    SELECT t.*, r.criterion, r.score, r.comment
                    FROM evaluation_tasks t
                    LEFT JOIN evaluation_ratings r ON t.id = r.task_id
                    WHERE t.completed_at IS NOT NULL
                    ORDER BY t.id, r.criterion
                """)
                
                results = {}
                for row in cursor.fetchall():
                    task_id = row[0]
                    if task_id not in results:
                        results[task_id] = {
                            'task_id': task_id,
                            'query': row[1],
                            'reference_text': row[2],
                            'candidate_text': row[3],
                            'api_name': row[4],
                            'model_name': row[5],
                            'created_at': row[6],
                            'completed_at': row[7],
                            'evaluator_id': row[8],
                            'ratings': {}
                        }
                    
                    if row[9]:  # If there's a rating
                        results[task_id]['ratings'][row[9]] = {
                            'score': row[10],
                            'comment': row[11]
                        }
                
                # Write to JSON file
                with open(output_path, 'w') as f:
                    json.dump(list(results.values()), f, indent=2, default=str)
                
                logger.info(f"Exported evaluation results to: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting evaluation results: {e}")
            return False


class EvaluationInterface:
    """Command-line interface for human evaluation."""
    
    def __init__(self, evaluator: HumanEvaluator):
        self.evaluator = evaluator
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
    
    def run_evaluation_session(self, evaluator_id: str):
        """Run an interactive evaluation session."""
        self.print_header("🔍 Human Evaluation Session", f"Evaluator ID: {evaluator_id}")
        
        session_id = self.evaluator.start_evaluation_session(evaluator_id)
        self.print_success(f"Session ID: {session_id}")
        
        task_count = 0
        while True:
            # Get next task
            task = self.evaluator.get_next_task(evaluator_id)
            if not task:
                self.print_warning("No more tasks available for evaluation.")
                break
            
            task_count += 1
            
            # Display task
            self._display_task(task, task_count)
            
            # Collect ratings
            ratings = self._collect_ratings()
            
            # Submit evaluation
            if self.evaluator.submit_evaluation(task.id, evaluator_id, ratings):
                self.print_success("✅ Evaluation submitted successfully!")
            else:
                self.print_error("❌ Failed to submit evaluation.")
            
            # Ask if user wants to continue
            if not self.get_confirmation("Continue with next task?", default=True):
                break
        
        self.print_header("🏁 Evaluation Session Complete", f"Tasks completed: {task_count}")
    
    def _display_task(self, task: EvaluationTask, task_number: int):
        """Display an evaluation task."""
        from rich.panel import Panel
        from rich.text import Text
        
        self.print_section(f"📋 Task {task_number}")
        
        # Task info
        task_info = f"🆔 Task ID: {task.id}\n🔧 API: {task.api_name}\n🤖 Model: {task.model_name}"
        self.console.print(Panel(task_info, title="Task Information", border_style="cyan"))
        
        # Query
        query_panel = Panel(task.query, title="❓ Query", border_style="blue")
        self.console.print(query_panel)
        
        # Reference answer
        ref_panel = Panel(task.reference_text, title="✅ Reference Answer", border_style="green")
        self.console.print(ref_panel)
        
        # Model response
        response_panel = Panel(task.candidate_text, title="🤖 Model Response", border_style="yellow")
        self.console.print(response_panel)
    
    def _collect_ratings(self) -> Dict[EvaluationCriteria, Tuple[int, str]]:
        """Collect ratings from the evaluator."""
        from rich.panel import Panel
        from rich.table import Table
        
        ratings = {}
        
        # Show rating scale
        scale_info = """
        📊 Rating Scale:
        • 1 = Very Poor
        • 2 = Poor  
        • 3 = Fair
        • 4 = Good
        • 5 = Excellent
        """
        self.console.print(Panel(scale_info, title="📏 Rating Guidelines", border_style="cyan"))
        
        self.print_section("📋 Please rate the response on the following criteria")
        
        for criterion in EvaluationCriteria:
            self.console.print(f"\n[bold cyan]🎯 {criterion.value.title()}:[/bold cyan]")
            
            # Get score
            score = self.get_integer_input(f"Score (1-5)", minimum=1, maximum=5)
            if score is None:
                score = 3  # Default to fair if cancelled
            
            # Get comment
            comment = self.get_user_input("💭 Comment (optional, press Enter to skip)")
            
            ratings[criterion] = (score, comment if comment else None)
        
        return ratings