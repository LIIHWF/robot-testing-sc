#!/usr/bin/env python3
"""
Greedy t-way combinatorial testing selection algorithm.

This script implements a greedy algorithm that selects configurations from
a candidate pool to achieve t-way coverage with minimal number of test cases.

Key optimization: Uses inverted index (tuple -> configs) for fast lookup.
Forbidden tuples are implicitly defined as tuples not appearing in any candidate.

The greedy algorithm works as follows:
1. Extract all t-way tuples from candidate configurations (defines coverable set)
2. Build inverted index: tuple -> set of config indices that cover it
3. Use priority-based selection: pick config covering most uncovered tuples
4. Update incrementally until all tuples are covered
"""

import json
import sys
import argparse
from itertools import combinations, product
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple, FrozenSet, Optional
from pathlib import Path
import time
import heapq


def load_candidate_configs(json_file: str) -> List[Dict[str, Any]]:
    """Load candidate configurations from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    return configs


def extract_parameters_and_domains(configs: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Extract all parameters and their possible values (domains) from configurations.
    
    Returns:
        Dictionary mapping parameter names to sets of possible values
    """
    param_domains: Dict[str, Set[str]] = defaultdict(set)
    
    for config in configs:
        for param_name, param_value in config.items():
            param_domains[param_name].add(str(param_value))
    
    return param_domains


def config_to_tuple_key(config: Dict[str, Any]) -> FrozenSet[Tuple[str, str]]:
    """Convert a configuration to a hashable key for deduplication."""
    return frozenset((k, str(v)) for k, v in config.items())


def get_tway_tuples_from_config(
    config: Dict[str, Any], 
    param_names: List[str], 
    t: int
) -> Set[FrozenSet[Tuple[str, str]]]:
    """
    Extract all t-way tuples from a configuration.
    
    Args:
        config: Configuration dictionary
        param_names: List of parameter names
        t: The interaction strength (t-way)
    
    Returns:
        Set of t-way tuples (each tuple is a frozenset of (param, value) pairs)
    """
    tuples = set()
    
    # Generate all combinations of t parameters
    for param_combo in combinations(param_names, t):
        # Create a tuple with the values for this parameter combination
        tuple_items = []
        for param in param_combo:
            if param in config:
                tuple_items.append((param, str(config[param])))
        
        if len(tuple_items) == t:
            tuples.add(frozenset(tuple_items))
    
    return tuples


def compute_theoretical_tuples_count(param_domains: Dict[str, Set[str]], t: int) -> int:
    """
    Compute the number of theoretically possible t-way tuples without generating them.
    This is used for statistics only.
    """
    param_names = list(param_domains.keys())
    total = 0
    
    for param_combo in combinations(param_names, t):
        combo_size = 1
        for param in param_combo:
            combo_size *= len(param_domains[param])
        total += combo_size
    
    return total


def generate_all_tway_tuples(
    param_domains: Dict[str, Set[str]], 
    t: int
) -> Set[FrozenSet[Tuple[str, str]]]:
    """
    Generate all possible t-way tuples from parameter domains.
    Only called when we need to compute forbidden tuples explicitly.
    """
    all_tuples = set()
    param_names = list(param_domains.keys())
    
    for param_combo in combinations(param_names, t):
        domains = [[(param, val) for val in param_domains[param]] for param in param_combo]
        for value_combo in product(*domains):
            all_tuples.add(frozenset(value_combo))
    
    return all_tuples


class GreedyTWaySelector:
    """
    Greedy t-way combinatorial testing selector with inverted index optimization.
    
    Uses forbidden tuples concept: tuples not in any candidate are implicitly forbidden.
    """
    
    def __init__(self, configs: List[Dict[str, Any]], t: int, verbose: bool = True):
        self.original_configs = configs
        self.t = t
        self.verbose = verbose
        
        # Will be populated during initialization
        self.param_domains: Dict[str, Set[str]] = {}
        self.param_names: List[str] = []
        self.unique_configs: List[Dict[str, Any]] = []
        
        # Core data structures for greedy selection
        self.config_tuples: List[Set[FrozenSet[Tuple[str, str]]]] = []
        self.tuple_to_configs: Dict[FrozenSet[Tuple[str, str]], Set[int]] = defaultdict(set)
        self.coverable_tuples: Set[FrozenSet[Tuple[str, str]]] = set()
        
        # Statistics
        self.theoretical_tuple_count = 0
        self.forbidden_tuple_count = 0
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def initialize(self):
        """Initialize data structures from candidate configurations."""
        start_time = time.time()
        
        # Extract parameter domains
        self.param_domains = extract_parameters_and_domains(self.original_configs)
        self.param_names = sorted(self.param_domains.keys())
        
        self._log(f"Parameters: {len(self.param_names)}")
        self._log(f"Candidate configurations: {len(self.original_configs)}")
        
        # Validate t
        if self.t > len(self.param_names):
            raise ValueError(f"t ({self.t}) cannot be greater than number of parameters ({len(self.param_names)})")
        
        # Deduplicate configurations
        seen_configs: Set[FrozenSet[Tuple[str, str]]] = set()
        for config in self.original_configs:
            config_key = config_to_tuple_key(config)
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                self.unique_configs.append(config)
        
        self._log(f"Unique configurations: {len(self.unique_configs)}")
        
        # Compute theoretical tuple count (for statistics)
        self.theoretical_tuple_count = compute_theoretical_tuples_count(self.param_domains, self.t)
        self._log(f"\nTheoretical {self.t}-way tuples: {self.theoretical_tuple_count}")
        
        # Extract t-way tuples from candidates and build inverted index
        self._log(f"Building inverted index from candidate configurations...")
        
        for idx, config in enumerate(self.unique_configs):
            tuples = get_tway_tuples_from_config(config, self.param_names, self.t)
            self.config_tuples.append(tuples)
            
            # Update inverted index: tuple -> configs that contain it
            for tup in tuples:
                self.tuple_to_configs[tup].add(idx)
            
            # Track all coverable tuples
            self.coverable_tuples.update(tuples)
            
            # Progress indicator for large datasets
            if self.verbose and (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1}/{len(self.unique_configs)} configurations...")
        
        # Forbidden tuples = theoretical - coverable
        self.forbidden_tuple_count = self.theoretical_tuple_count - len(self.coverable_tuples)
        
        self._log(f"Coverable {self.t}-way tuples: {len(self.coverable_tuples)}")
        self._log(f"Forbidden {self.t}-way tuples (not in any candidate): {self.forbidden_tuple_count}")
        
        elapsed = time.time() - start_time
        self._log(f"Initialization completed in {elapsed:.2f} seconds")
    
    def select_greedy(self, post_optimize: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run greedy selection algorithm with inverted index optimization.
        
        Args:
            post_optimize: Whether to run post-optimization to remove redundant configs
        
        Returns:
            Tuple of (selected configurations, statistics dictionary)
        """
        start_time = time.time()
        
        # Track uncovered tuples and their coverage count per config
        uncovered_tuples = self.coverable_tuples.copy()
        
        # For each config, track how many uncovered tuples it covers
        # Using a heap for efficient max selection
        config_coverage_count = []
        for idx, tuples in enumerate(self.config_tuples):
            # Initially, all tuples in config are uncovered
            count = len(tuples & uncovered_tuples)
            # Use negative count for max-heap behavior with heapq (min-heap)
            heapq.heappush(config_coverage_count, (-count, idx))
        
        # Track which configs are already selected
        selected_indices: Set[int] = set()
        selected_configs: List[Dict[str, Any]] = []
        
        self._log(f"\nStarting greedy selection for {self.t}-way coverage...")
        
        iteration = 0
        while uncovered_tuples:
            iteration += 1
            
            # Find the configuration with maximum uncovered tuple coverage
            # Pop candidates until we find one that's not selected and has valid count
            best_idx = -1
            best_count = 0
            
            while config_coverage_count:
                neg_count, idx = heapq.heappop(config_coverage_count)
                
                if idx in selected_indices:
                    continue
                
                # Recompute actual coverage (may have decreased)
                actual_covered = self.config_tuples[idx] & uncovered_tuples
                actual_count = len(actual_covered)
                
                if actual_count == 0:
                    # This config covers nothing new, skip
                    continue
                
                if actual_count == -neg_count:
                    # Count is accurate, this is our best choice
                    best_idx = idx
                    best_count = actual_count
                    break
                else:
                    # Count changed, re-insert with updated count
                    heapq.heappush(config_coverage_count, (-actual_count, idx))
            
            # If no configuration covers any new tuples, we're done
            if best_idx == -1 or best_count == 0:
                break
            
            # Select this configuration
            selected_indices.add(best_idx)
            selected_configs.append(self.unique_configs[best_idx])
            
            # Update uncovered tuples
            covered_by_best = self.config_tuples[best_idx] & uncovered_tuples
            uncovered_tuples -= covered_by_best
            
            # Log progress
            if self.verbose and (iteration <= 10 or iteration % 50 == 0 or not uncovered_tuples):
                coverage_pct = (len(self.coverable_tuples) - len(uncovered_tuples)) / len(self.coverable_tuples) * 100
                print(f"  Iteration {iteration}: selected config #{best_idx}, "
                      f"covered {best_count} new tuples, "
                      f"remaining: {len(uncovered_tuples)}, "
                      f"coverage: {coverage_pct:.2f}%")
        
        greedy_time = time.time() - start_time
        
        # Post-optimization: try to remove redundant configurations
        removed_count = 0
        if post_optimize and len(selected_indices) > 1:
            self._log(f"\nRunning post-optimization to remove redundant configs...")
            selected_indices, removed_count = self._post_optimize(selected_indices)
            selected_configs = [self.unique_configs[idx] for idx in sorted(selected_indices)]
            self._log(f"  Removed {removed_count} redundant configurations")
        
        elapsed_time = time.time() - start_time
        
        # Calculate final coverage
        covered_tuples_count = len(self.coverable_tuples) - len(uncovered_tuples)
        coverage_ratio = covered_tuples_count / len(self.coverable_tuples) if self.coverable_tuples else 1.0
        
        # Build statistics
        stats = {
            "t": self.t,
            "total_parameters": len(self.param_names),
            "parameter_names": self.param_names,
            "candidate_count": len(self.original_configs),
            "unique_candidate_count": len(self.unique_configs),
            "theoretical_tuples": self.theoretical_tuple_count,
            "coverable_tuples": len(self.coverable_tuples),
            "forbidden_tuples": self.forbidden_tuple_count,
            "covered_tuples": covered_tuples_count,
            "post_optimization_removed": removed_count,
            "uncovered_tuples": len(uncovered_tuples),
            "coverage_ratio": coverage_ratio,
            "selected_count": len(selected_configs),
            "reduction_ratio": 1 - len(selected_configs) / len(self.unique_configs) if self.unique_configs else 0,
            "iterations": iteration,
            "greedy_time_seconds": greedy_time,
            "elapsed_time_seconds": elapsed_time
        }
        
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Greedy Selection Complete!")
            print(f"{'=' * 60}")
            print(f"  Interaction strength (t): {self.t}")
            print(f"  Total parameters: {len(self.param_names)}")
            print(f"  Candidate configurations: {len(self.unique_configs)}")
            print(f"  Selected configurations: {len(selected_configs)}")
            print(f"  Reduction: {stats['reduction_ratio']*100:.2f}%")
            print(f"  Coverage: {coverage_ratio*100:.2f}% of coverable tuples")
            print(f"  Forbidden tuples: {self.forbidden_tuple_count} (implicitly excluded)")
            if removed_count > 0:
                print(f"  Post-optimization removed: {removed_count} redundant configs")
            print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        
        return selected_configs, stats
    
    def _post_optimize(self, selected_indices: Set[int]) -> Tuple[Set[int], int]:
        """
        Post-optimization: remove redundant configurations.
        
        A configuration is redundant if all its tuples are covered by other
        selected configurations.
        
        Args:
            selected_indices: Set of selected configuration indices
        
        Returns:
            Tuple of (optimized indices set, number of configs removed)
        """
        # Sort by coverage count (ascending) - try to remove low-coverage ones first
        sorted_indices = sorted(
            selected_indices,
            key=lambda idx: len(self.config_tuples[idx])
        )
        
        optimized = set(selected_indices)
        removed = 0
        
        for idx in sorted_indices:
            if idx not in optimized:
                continue
            
            # Check if this config's tuples are all covered by others
            my_tuples = self.config_tuples[idx]
            others = optimized - {idx}
            
            # Get all tuples covered by other selected configs
            covered_by_others = set()
            for other_idx in others:
                covered_by_others.update(self.config_tuples[other_idx])
            
            # Check if all my tuples are covered
            if my_tuples.issubset(covered_by_others):
                optimized.remove(idx)
                removed += 1
        
        return optimized, removed
    
    def get_forbidden_tuples(self) -> Set[FrozenSet[Tuple[str, str]]]:
        """
        Explicitly compute and return forbidden tuples.
        These are t-way tuples that don't appear in any candidate configuration.
        
        Warning: This can be memory-intensive for large parameter spaces.
        """
        self._log("Computing forbidden tuples explicitly...")
        all_theoretical = generate_all_tway_tuples(self.param_domains, self.t)
        forbidden = all_theoretical - self.coverable_tuples
        self._log(f"Found {len(forbidden)} forbidden tuples")
        return forbidden
    
    def export_forbidden_tuples(self, output_file: str):
        """Export forbidden tuples to a JSON file."""
        forbidden = self.get_forbidden_tuples()
        
        # Convert to serializable format
        forbidden_list = []
        for tup in forbidden:
            forbidden_list.append({param: val for param, val in tup})
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(forbidden_list, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(forbidden_list)} forbidden tuples to {output_file}")


def greedy_tway_selection(
    configs: List[Dict[str, Any]], 
    t: int,
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Use greedy algorithm to select minimum configurations for t-way coverage.
    
    This is a convenience function that wraps GreedyTWaySelector.
    
    Args:
        configs: List of candidate configurations
        t: The interaction strength (t-way coverage)
        verbose: Whether to print progress information
    
    Returns:
        Tuple of (selected configurations, statistics dictionary)
    """
    selector = GreedyTWaySelector(configs, t, verbose)
    selector.initialize()
    return selector.select_greedy()


def save_results(
    selected_configs: List[Dict[str, Any]], 
    stats: Dict[str, Any],
    output_json: str,
    output_stats: str = None
) -> None:
    """Save selected configurations and statistics to files."""
    # Save selected configurations
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(selected_configs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(selected_configs)} selected configurations to {output_json}")
    
    # Save statistics
    if output_stats:
        stats_copy = {k: v for k, v in stats.items()}
        with open(output_stats, 'w', encoding='utf-8') as f:
            json.dump(stats_copy, f, indent=2, ensure_ascii=False)
        print(f"Saved statistics to {output_stats}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Greedy t-way combinatorial testing selection algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2-way coverage (pairwise testing)
  python greedy_tway_selection.py candidates.json -t 2 -o selected_2way.json
  
  # 3-way coverage
  python greedy_tway_selection.py candidates.json -t 3 -o selected_3way.json
  
  # Save statistics and forbidden tuples
  python greedy_tway_selection.py candidates.json -t 2 -o selected.json --stats stats.json --forbidden forbidden.json
"""
    )
    
    parser.add_argument(
        "input_json", 
        type=str, 
        help="Path to JSON file containing candidate configurations"
    )
    parser.add_argument(
        "-t", "--t-way",
        type=int,
        default=2,
        help="Interaction strength for t-way coverage (default: 2 for pairwise)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file for selected configurations (default: input_tway.json)"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Output JSON file for statistics (optional)"
    )
    parser.add_argument(
        "--forbidden",
        type=str,
        default=None,
        help="Output JSON file for forbidden tuples (optional, can be memory-intensive)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip post-optimization step"
    )
    
    args = parser.parse_args()
    
    # Determine output file name
    if args.output is None:
        input_path = Path(args.input_json)
        args.output = str(input_path.parent / f"{input_path.stem}_{args.t_way}way.json")
    
    # Load candidate configurations
    print(f"Loading candidate configurations from {args.input_json}...")
    configs = load_candidate_configs(args.input_json)
    print(f"Loaded {len(configs)} candidate configurations")
    
    # Create selector and initialize
    selector = GreedyTWaySelector(configs, args.t_way, verbose=not args.quiet)
    selector.initialize()
    
    # Run greedy selection
    selected_configs, stats = selector.select_greedy(post_optimize=not args.no_optimize)
    
    # Save results
    save_results(selected_configs, stats, args.output, args.stats)
    
    # Export forbidden tuples if requested
    if args.forbidden:
        selector.export_forbidden_tuples(args.forbidden)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
