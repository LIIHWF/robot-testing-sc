#!/usr/bin/env python3
"""
WP Analysis Tool for Enumerated Tasks

This tool reads tasks from an enumeration file, computes the Weakest Precondition (WP)
for each task using the Tabletop domain WP interface, and records the results.

Usage:
    python wp_analysis.py <enum_file> [--output <output_file>] [--limit <n>]
    
Example:
    python wp_analysis.py ct_model/enum_test.txt --output cache/wp_analysis_result.txt
"""

import os
import sys
import argparse
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

# Add py_lib/wp_z3 to path to import wp_z3 modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'py_lib', 'wp_z3'))

from wp_z3_interface import TabletopWP, Z3_AVAILABLE
from z3_utils import Formula, FormulaType


@dataclass
class TaskAnalysisResult:
    """Result of analyzing a single task"""
    task_string: str
    rule_sequence: str
    prolog_program: List[str]
    wp_formula: Optional[str]
    is_satisfiable: Optional[bool]
    is_false: bool  # WP = false (never executable)
    error: Optional[str]
    

def parse_task_string(task_str: str) -> List[str]:
    """
    Parse a task string from enum file to Prolog program format.
    
    Examples:
        "put bread plate" -> ["put(bread, plate)"]
        "put bread plate ; open microwave" -> ["put(bread, plate)", "open(microwave)"]
        "if loc bread plate then open microwave else close drawer" 
            -> ["ndet([?(loc(bread, plate)), open(microwave)], [?(neg(loc(bread, plate))), close(drawer)])"]
    """
    # Split by semicolon for sequences
    parts = [p.strip() for p in task_str.split(';')]
    
    program = []
    for part in parts:
        tokens = part.split()
        
        if not tokens:
            continue
            
        if tokens[0] == 'if':
            # Parse if-then-else
            # Format: if <cond> then <action1> else <action2>
            # or: if <neg> <cond> then <action1> else <action2>
            try:
                then_idx = tokens.index('then')
                else_idx = tokens.index('else')
                
                cond_tokens = tokens[1:then_idx]
                then_tokens = tokens[then_idx+1:else_idx]
                else_tokens = tokens[else_idx+1:]
                
                cond = parse_condition(cond_tokens)
                then_action = parse_action(then_tokens)
                else_action = parse_action(else_tokens)
                
                # Golog semantics: ndet([?(cond), action1], [?(neg(cond)), action2])
                if_stmt = f"ndet([?({cond}), {then_action}], [?(neg({cond})), {else_action}])"
                program.append(if_stmt)
            except (ValueError, IndexError) as e:
                # Fallback: just join tokens
                program.append(part)
        else:
            # Regular action
            action = parse_action(tokens)
            program.append(action)
    
    return program


def parse_action(tokens: List[str]) -> str:
    """Parse action tokens to Prolog format"""
    if not tokens:
        return ""
    
    action_type = tokens[0]
    
    if action_type == 'put':
        if len(tokens) >= 3:
            return f"put({tokens[1]}, {tokens[2]})"
        return f"put({', '.join(tokens[1:])})"
    elif action_type == 'open':
        if len(tokens) >= 2:
            return f"open({tokens[1]})"
        return "open(unknown)"
    elif action_type == 'close':
        if len(tokens) >= 2:
            return f"close({tokens[1]})"
        return "close(unknown)"
    elif action_type == 'turn_on':
        if len(tokens) >= 2:
            return f"turn_on({tokens[1]})"
        return "turn_on(unknown)"
    else:
        # Unknown action type, try to make it a functor
        if len(tokens) == 1:
            return tokens[0]
        return f"{tokens[0]}({', '.join(tokens[1:])})"


def parse_condition(tokens: List[str]) -> str:
    """Parse condition tokens to Prolog format"""
    if not tokens:
        return "true"
    
    # Check for negation
    if tokens[0] == 'neg':
        inner = parse_condition(tokens[1:])
        return f"neg({inner})"
    
    cond_type = tokens[0]
    
    if cond_type == 'loc':
        if len(tokens) >= 3:
            return f"loc({tokens[1]}, {tokens[2]})"
        return f"loc({', '.join(tokens[1:])})"
    elif cond_type == 'door_open':
        if len(tokens) >= 2:
            return f"door_open({tokens[1]})"
        return "door_open(unknown)"
    elif cond_type == 'running':
        if len(tokens) >= 2:
            return f"running({tokens[1]})"
        return "running(unknown)"
    else:
        # Unknown condition type
        if len(tokens) == 1:
            return tokens[0]
        return f"{tokens[0]}({', '.join(tokens[1:])})"


def load_tasks(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Load tasks from enumeration file.
    
    Returns:
        List of (task_string, rule_sequence, rule_counts) tuples
    """
    tasks = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse line: task_string | rule_sequence | rule_counts
            parts = line.split('|')
            if len(parts) >= 3:
                task_string = parts[0].strip()
                rule_sequence = parts[1].strip()
                rule_counts = parts[2].strip()
                tasks.append((task_string, rule_sequence, rule_counts))
            elif len(parts) == 1:
                # Just the task string
                tasks.append((parts[0].strip(), "", ""))
    
    return tasks


def is_formula_false(formula: Formula) -> bool:
    """Check if formula is the constant false"""
    return formula.type == FormulaType.FALSE


def analyze_task_worker(args: Tuple[str, str, str]) -> TaskAnalysisResult:
    """
    Worker function for parallel processing.
    Creates its own WP interface instance.
    
    Args:
        args: Tuple of (domain_file, task_string, rule_sequence)
    
    Returns:
        TaskAnalysisResult
    """
    # Ensure imports are available in subprocess
    # This is needed because subprocesses may not inherit the module-level imports
    import sys
    import os
    
    # Add py_lib/wp_z3 to path if not already present (for subprocess safety)
    wp_z3_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        'py_lib', 'wp_z3'
    )
    wp_z3_path = os.path.abspath(wp_z3_path)
    if wp_z3_path not in sys.path:
        sys.path.insert(0, wp_z3_path)
    
    from wp_z3_interface import TabletopWP
    
    domain_file, task_string, rule_sequence = args
    wp_interface = TabletopWP(domain_file)
    return analyze_task(wp_interface, task_string, rule_sequence)


def analyze_task(wp_interface: TabletopWP, task_string: str, rule_sequence: str) -> TaskAnalysisResult:
    """
    Analyze a single task using WP computation.
    """
    try:
        # Convert task string to Prolog program
        program = parse_task_string(task_string)
        
        # Compute WP with postcondition 'true' (checking executability)
        try:
            wp_formula = wp_interface.compute_wp(program, 'true')
            wp_str = str(wp_formula)
            
            # Check if WP is false
            is_false = is_formula_false(wp_formula)
            
            # Check satisfiability (only if not obviously false)
            is_sat = None
            if not is_false and Z3_AVAILABLE:
                try:
                    is_sat = wp_interface.is_satisfiable(wp_formula)
                except Exception as e:
                    is_sat = None
            
            return TaskAnalysisResult(
                task_string=task_string,
                rule_sequence=rule_sequence,
                prolog_program=program,
                wp_formula=wp_str,
                is_satisfiable=is_sat,
                is_false=is_false,
                error=None
            )
        except Exception as e:
            return TaskAnalysisResult(
                task_string=task_string,
                rule_sequence=rule_sequence,
                prolog_program=program,
                wp_formula=None,
                is_satisfiable=None,
                is_false=False,
                error=str(e)
            )
            
    except Exception as e:
        return TaskAnalysisResult(
            task_string=task_string,
            rule_sequence=rule_sequence,
            prolog_program=[],
            wp_formula=None,
            is_satisfiable=None,
            is_false=False,
            error=f"Parse error: {str(e)}"
        )


def analyze_all_tasks(
    enum_file: str, 
    output_file: str,
    limit: Optional[int] = None,
    verbose: bool = True,
    num_workers: Optional[int] = None
):
    """
    Analyze all tasks from an enumeration file.
    
    Args:
        enum_file: Path to enumeration file
        output_file: Path to output results file
        limit: Maximum number of tasks to analyze (None = all)
        verbose: Print progress information
        num_workers: Number of parallel workers (None = CPU count)
    """
    # Load tasks
    if verbose:
        print(f"Loading tasks from {enum_file}...")
    tasks = load_tasks(enum_file)
    total_tasks = len(tasks)
    
    if limit:
        tasks = tasks[:limit]
    
    if verbose:
        print(f"Loaded {total_tasks} tasks, analyzing {len(tasks)}")
    
    # Initialize WP interface
    domain_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        'prolog_lib', 'domains', 'main.pl'
    )
    
    if verbose:
        print(f"Initializing WP interface with domain: {domain_file}")
        if num_workers:
            print(f"Using {num_workers} parallel workers")
        else:
            import multiprocessing
            print(f"Using {multiprocessing.cpu_count()} parallel workers (default)")
    
    # Prepare task arguments for parallel processing
    # Each worker needs domain_file, task_string, rule_sequence
    task_args = [(domain_file, task_str, rule_seq) for task_str, rule_seq, _ in tasks]
    
    # Initialize stats
    stats = {
        'total': len(tasks),
        'valid': 0,
        'invalid_false': 0,
        'invalid_unsat': 0,
        'errors': 0,
    }
    
    start_time = time.time()
    results: List[TaskAnalysisResult] = [None] * len(tasks)  # Pre-allocate to maintain order
    
    # Parallel processing
    if num_workers == 1:
        # Sequential processing (for debugging)
        task_iterator = tqdm(enumerate(task_args), total=len(task_args), desc="Analyzing tasks", unit="task", disable=not verbose) if verbose else enumerate(task_args)
        
        for i, args in task_iterator:
            result = analyze_task_worker(args)
            results[i] = result
            
            # Update stats
            if result.error:
                stats['errors'] += 1
            elif result.is_false:
                stats['invalid_false'] += 1
            elif result.is_satisfiable is False:
                stats['invalid_unsat'] += 1
            else:
                stats['valid'] += 1
            
            # Update progress bar description with stats
            if verbose and hasattr(task_iterator, 'set_postfix'):
                task_iterator.set_postfix({
                    'valid': stats['valid'],
                    'false': stats['invalid_false'],
                    'unsat': stats['invalid_unsat'],
                    'errors': stats['errors']
                })
    else:
        # Parallel processing with ProcessPoolExecutor
        import multiprocessing
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(analyze_task_worker, args): i 
                for i, args in enumerate(task_args)
            }
            
            # Process completed tasks with progress bar
            if verbose:
                pbar = tqdm(total=len(tasks), desc="Analyzing tasks", unit="task")
            else:
                pbar = None
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    
                    # Update stats
                    if result.error:
                        stats['errors'] += 1
                    elif result.is_false:
                        stats['invalid_false'] += 1
                    elif result.is_satisfiable is False:
                        stats['invalid_unsat'] += 1
                    else:
                        stats['valid'] += 1
                    
                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'valid': stats['valid'],
                            'false': stats['invalid_false'],
                            'unsat': stats['invalid_unsat'],
                            'errors': stats['errors']
                        })
                except Exception as e:
                    # Handle errors in worker
                    error_result = TaskAnalysisResult(
                        task_string=task_args[index][1],
                        rule_sequence=task_args[index][2],
                        prolog_program=[],
                        wp_formula=None,
                        is_satisfiable=None,
                        is_false=False,
                        error=f"Worker error: {str(e)}"
                    )
                    results[index] = error_result
                    stats['errors'] += 1
                    if pbar:
                        pbar.update(1)
            
            if pbar:
                pbar.close()
    
    elapsed = time.time() - start_time
    
    # Write results
    if verbose:
        print(f"\nWriting results to {output_file}...")
    
    write_results(results, stats, output_file, elapsed)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Analysis Complete")
        print(f"{'='*60}")
        print(f"Total tasks: {stats['total']}")
        print(f"Valid (SAT): {stats['valid']}")
        print(f"Invalid (WP=false): {stats['invalid_false']}")
        print(f"Invalid (UNSAT): {stats['invalid_unsat']}")
        print(f"Errors: {stats['errors']}")
        print(f"Time: {elapsed:.2f}s ({len(tasks)/elapsed:.1f} tasks/s)")
    
    return results, stats


def write_results(results: List[TaskAnalysisResult], stats: Dict, output_file: str, elapsed: float):
    """Write analysis results to file"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("# WP Analysis Results\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total tasks: {stats['total']}\n")
        f.write(f"# Valid (SAT): {stats['valid']}\n")
        f.write(f"# Invalid (WP=false): {stats['invalid_false']}\n")
        f.write(f"# Invalid (UNSAT): {stats['invalid_unsat']}\n")
        f.write(f"# Errors: {stats['errors']}\n")
        f.write(f"# Time: {elapsed:.2f}s\n")
        f.write("#\n")
        f.write("# Format: status | task_string | rule_sequence | wp_formula\n")
        f.write("#   status: VALID, FALSE (WP=false), UNSAT, ERROR\n")
        f.write("#\n")
        f.write("\n")
        
        # Results
        for r in results:
            if r.error:
                status = "ERROR"
                wp = f"Error: {r.error}"
            elif r.is_false:
                status = "FALSE"
                wp = "false"
            elif r.is_satisfiable is False:
                status = "UNSAT"
                wp = r.wp_formula or "unknown"
            else:
                status = "VALID"
                wp = r.wp_formula or "unknown"
            
            f.write(f"{status} | {r.task_string} | {r.rule_sequence} | {wp}\n")
        
        f.write("\n")
        f.write("# End of results\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze WP for enumerated tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wp_analysis.py ct_model/enum_test.txt
  python wp_analysis.py ct_model/enum_test.txt --output cache/wp_results.txt
  python wp_analysis.py ct_model/enum_test.txt --limit 100 --output cache/wp_sample.txt
        """
    )
    parser.add_argument('enum_file', type=str, help='Path to enumeration file')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output file path (default: cache/wp_analysis_result.txt)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Maximum number of tasks to analyze')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count, use 1 for sequential)')
    
    args = parser.parse_args()
    
    # Default output file
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'cache', 'wp_analysis_result.txt'
        )
    
    # Run analysis
    analyze_all_tasks(
        args.enum_file,
        args.output,
        limit=args.limit,
        verbose=not args.quiet,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()

