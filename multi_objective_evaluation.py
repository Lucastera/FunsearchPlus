"""
Bin Packing Algorithm Evaluator
-------------------------------
This module provides a standardized way to evaluate different bin packing algorithms
using the exact same evaluation criteria as used in FunSearch.
"""

import numpy as np
import json
from typing import Tuple, List, Dict, Any, Callable
from sourceCode import bin_packing_utils


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray, priority_func: Callable
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority_func(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


class BinPackingEvaluator:
    """
    Evaluator for comparing different bin packing algorithms
    using the same scoring mechanism as in FunSearch
    """
    
    def __init__(self, instances: Dict[str, Dict]):
        """
        Initialize the evaluator with bin packing instances
        
        Args:
            instances: Dictionary of bin packing instances
        """
        self.instances = instances
    
    def evaluate(self, priority_func: Callable) -> float:
        """
        Evaluate heuristic function on a set of online binpacking instances.
        Uses the exact same evaluation logic as in FunSearch.
        
        Args:
            priority_func: The priority function to evaluate
            
        Returns:
            Score (negative of average number of bins used)
        """
        # List storing number of bins used for each instance.
        num_bins = []
        # Perform online binpacking for each instance.
        for name in self.instances:
            instance = self.instances[name]
            capacity = instance['capacity']
            items = instance['items']
            # Create num_items bins so there will always be space for all items,
            # regardless of packing order. Array has shape (num_items,).
            bins = np.array([capacity for _ in range(instance['num_items'])])
            # Pack items into bins and return remaining capacity in bins_packed, which
            # has shape (num_items,).
            _, bins_packed = online_binpack(items, bins, priority_func)
            # If remaining capacity in a bin is equal to initial capacity, then it is
            # unused. Count number of used bins.
            num_bins.append((bins_packed != capacity).sum())
        # Score of heuristic function is negative of average number of bins used
        # across instances (as we want to minimize number of bins).
        return -np.mean(num_bins)

    def evaluate_all_datasets(self, priority_func: Callable) -> Dict[str, float]:
        """
        Evaluate priority function on all instances separately and return individual scores
        
        Args:
            priority_func: The priority function to evaluate
            
        Returns:
            Dictionary mapping instance name to score
        """
        results = {}
        for name in self.instances:
            instance = {name: self.instances[name]}
            evaluator = BinPackingEvaluator(instance)
            results[name] = evaluator.evaluate(priority_func)
        return results

    def compare_algorithms(self, algorithms: Dict[str, Callable]) -> Dict[str, float]:
        """
        Compare multiple bin packing algorithms
        
        Args:
            algorithms: Dictionary mapping algorithm names to priority functions
            
        Returns:
            Dictionary mapping algorithm names to scores
        """
        results = {}
        for name, func in algorithms.items():
            score = self.evaluate(func)
            results[name] = score
            print(f"Algorithm: {name}, Score: {score:.4f}")
        return results


# Implementation of common bin packing algorithms
def best_fit_priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Best Fit strategy - choose the bin that will have the smallest remaining space
    
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
        
    Returns:
        Priority score for each bin
    """
    remaining_space = bins - item
    # In case of ties, choose the bin with smaller index
    return -remaining_space + np.linspace(0, 0.0001, len(bins))


def first_fit_priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    First Fit strategy - choose the first bin that can accommodate the item
    
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
        
    Returns:
        Priority score for each bin
    """
    return -np.arange(len(bins)) * 0.0001


# Example usage
if __name__ == "__main__":
    # Choose dataset(s) to evaluate
    OR3_dataset = bin_packing_utils.datasets['OR3']
    Weibull_dataset = bin_packing_utils.datasets['Weibull 5k']
    
    # Define algorithms to compare
    algorithms = {
        "Best Fit": best_fit_priority,
        "First Fit": first_fit_priority,
    }
    
    print("\n----- Evaluating on OR3 dataset -----")
    or3_evaluator = BinPackingEvaluator(OR3_dataset)
    or3_results = or3_evaluator.compare_algorithms(algorithms)
    
    print("\n----- Evaluating on Weibull 5k dataset -----")
    weibull_evaluator = BinPackingEvaluator(Weibull_dataset)
    weibull_results = weibull_evaluator.compare_algorithms(algorithms)
    
    print("\n----- Summary -----")
    print("OR3 dataset:")
    for algo, score in or3_results.items():
        print(f"{algo}: Score = {score:.4f}")
    
    print("\nWeibull 5k dataset:")
    for algo, score in weibull_results.items():
        print(f"{algo}: Score = {score:.4f}")