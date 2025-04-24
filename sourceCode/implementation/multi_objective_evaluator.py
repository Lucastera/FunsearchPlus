import ast
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional

class MultiObjectiveEvaluator:
    """Evaluator that scores code on multiple dimensions."""
    
    def __init__(self):
        self.metrics = {
            "performance": self.evaluate_performance,
            "simplicity": self.evaluate_simplicity,
            "interpretability": self.evaluate_interpretability,
            "novelty": self.evaluate_novelty
        }
    
    def evaluate_performance(self, score: Optional[float], **kwargs) -> float:
        """Assess performance using original score."""
        return score if score is not None else float('-inf')
    
    def evaluate_simplicity(self, function_code: str, **kwargs) -> float:
        """Assess code simplicity based on length and complexity."""
        code_lines = [line for line in function_code.split('\n') if line.strip()]
        line_count = len(code_lines)
        
        complexity = sum(1 for line in code_lines if any(kw in line for kw in 
                                                        ['if ', 'else:', 'elif ', 'for ', 'while ']))
        
        return -0.7 * line_count - 0.3 * complexity
    
    def evaluate_interpretability(self, function_code: str, **kwargs) -> float:
        """Assess code interpretability based on comments and naming."""
        code_lines = function_code.split('\n')
        
        comments = sum(1 for line in code_lines if '#' in line)
        comment_ratio = comments / max(1, len(code_lines))
        
        var_names = []
        try:
            tree = ast.parse(function_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    var_names.append(node.id)
        except:
            pass
            
        avg_name_length = sum(len(name) for name in var_names) / max(1, len(var_names))
        name_score = -abs(avg_name_length - 8)
        
        return 5 * comment_ratio + name_score
    
    def evaluate_novelty(self, function_code: str, evaluated_functions: List[str], **kwargs) -> float:
        """Assess code novelty compared to existing solutions."""
        if not evaluated_functions:
            return 1.0
            
        avg_diff = sum(1 - SequenceMatcher(None, function_code, ef).ratio() 
                      for ef in evaluated_functions) / len(evaluated_functions)
        return avg_diff
    
    def compute_multi_objective_score(self, 
                                     function: Any, 
                                     scores_per_test: Optional[Dict], 
                                     evaluated_functions: List[str],
                                     weights: Optional[Dict[str, float]] = None) -> Dict:
        """Compute composite score based on multiple objectives."""
        if weights is None:
            weights = {"performance": 0.5, "simplicity": 0.2, "interpretability": 0.15, "novelty": 0.15}
            
        from implementation.funsearch import _reduce_score
        performance_score = _reduce_score(scores_per_test) if scores_per_test else float('-inf')
        function_code = str(function)
        
        scores = {
            "performance": self.evaluate_performance(performance_score),
            "simplicity": self.evaluate_simplicity(function_code),
            "interpretability": self.evaluate_interpretability(function_code),
            "novelty": self.evaluate_novelty(function_code, evaluated_functions)
        }
        
        total_score = sum(weights[metric] * scores[metric] for metric in weights)
        
        return {
            "total": total_score,
            "scores": scores
        }