#!/usr/bin/env python3
"""
Health check script for enhanced LLM benchmarking tool.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Health check for the benchmarking system."""
    
    def __init__(self):
        self.checks = []
        self.status = "healthy"
        
    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all dependencies are available."""
        dependencies = {
            'nltk': False,
            'sentence_transformers': False,
            'sklearn': False,
            'numpy': False,
            'colorama': False,
            'sqlite3': True  # Built-in
        }
        
        try:
            import nltk
            dependencies['nltk'] = True
        except ImportError:
            pass
            
        try:
            import sentence_transformers
            dependencies['sentence_transformers'] = True
        except ImportError:
            pass
            
        try:
            import sklearn
            dependencies['sklearn'] = True
        except ImportError:
            pass
            
        try:
            import numpy
            dependencies['numpy'] = True
        except ImportError:
            pass
            
        try:
            import colorama
            dependencies['colorama'] = True
        except ImportError:
            pass
        
        all_available = all(dependencies.values())
        
        return {
            'name': 'dependencies',
            'status': 'pass' if all_available else 'fail',
            'details': dependencies,
            'message': 'All dependencies available' if all_available else 'Some dependencies missing'
        }
    
    def check_databases(self) -> Dict[str, Any]:
        """Check database connectivity."""
        databases = {
            'evaluation.db': False,
            'ab_testing.db': False
        }
        
        for db_name in databases.keys():
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
                databases[db_name] = True
            except Exception as e:
                logger.warning(f"Database check failed for {db_name}: {e}")
        
        return {
            'name': 'databases',
            'status': 'pass' if any(databases.values()) else 'warn',
            'details': databases,
            'message': 'Database connectivity checked'
        }
    
    def check_nltk_data(self) -> Dict[str, Any]:
        """Check if NLTK data is available."""
        try:
            import nltk
            
            required_data = ['punkt', 'stopwords']
            available_data = {}
            
            for data_name in required_data:
                try:
                    nltk.data.find(f'tokenizers/{data_name}')
                    available_data[data_name] = True
                except LookupError:
                    try:
                        nltk.data.find(f'corpora/{data_name}')
                        available_data[data_name] = True
                    except LookupError:
                        available_data[data_name] = False
            
            all_available = all(available_data.values())
            
            return {
                'name': 'nltk_data',
                'status': 'pass' if all_available else 'fail',
                'details': available_data,
                'message': 'NLTK data available' if all_available else 'NLTK data missing'
            }
        except ImportError:
            return {
                'name': 'nltk_data',
                'status': 'fail',
                'details': {},
                'message': 'NLTK not installed'
            }
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file system permissions."""
        permissions = {
            'read_config': os.access('.env.sample', os.R_OK),
            'write_results': True,  # Will be tested below
            'write_logs': True      # Will be tested below
        }
        
        # Test write permissions
        test_dirs = ['results', 'logs']
        for test_dir in test_dirs:
            try:
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
                test_file = os.path.join(test_dir, 'test_write.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                permissions[f'write_{test_dir}'] = True
            except Exception:
                permissions[f'write_{test_dir}'] = False
        
        all_permissions = all(permissions.values())
        
        return {
            'name': 'file_permissions',
            'status': 'pass' if all_permissions else 'fail',
            'details': permissions,
            'message': 'File permissions OK' if all_permissions else 'File permission issues'
        }
    
    def check_model_loading(self) -> Dict[str, Any]:
        """Check if models can be loaded (basic test)."""
        try:
            from sentence_transformers import SentenceTransformer
            # Try to load a small model for testing
            model = SentenceTransformer('all-MiniLM-L6-v2')
            # Simple test encoding
            test_embedding = model.encode(["test sentence"])
            
            return {
                'name': 'model_loading',
                'status': 'pass',
                'details': {'model': 'all-MiniLM-L6-v2', 'embedding_shape': test_embedding.shape},
                'message': 'Model loading successful'
            }
        except Exception as e:
            return {
                'name': 'model_loading',
                'status': 'fail',
                'details': {'error': str(e)},
                'message': f'Model loading failed: {e}'
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        checks = [
            self.check_dependencies(),
            self.check_databases(),
            self.check_nltk_data(),
            self.check_file_permissions(),
            self.check_model_loading()
        ]
        
        # Determine overall status
        failed_checks = [check for check in checks if check['status'] == 'fail']
        warn_checks = [check for check in checks if check['status'] == 'warn']
        
        if failed_checks:
            overall_status = 'unhealthy'
        elif warn_checks:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'checks': checks,
            'summary': {
                'total_checks': len(checks),
                'passed': len([c for c in checks if c['status'] == 'pass']),
                'failed': len(failed_checks),
                'warnings': len(warn_checks)
            }
        }

def main():
    """Main health check function."""
    checker = HealthChecker()
    results = checker.run_all_checks()
    
    print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    if results['status'] == 'unhealthy':
        sys.exit(1)
    elif results['status'] == 'degraded':
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()