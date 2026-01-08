#!/usr/bin/env python3
"""
Verify CT Configurations Script

This script verifies configurations in simp_ct_configurations.json by:
1. Parsing task expressions
2. Computing weakest preconditions (WP) for task expressions
3. Validating if initial conditions satisfy the preconditions

Usage:
    python verify_ct_config.py cache/simp_ct_configurations.json
    python verify_ct_config.py cache/simp_ct_configurations.json --output results.json
"""

import argparse
import json
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add py_lib/wp_z3 to path to import wp_z3 modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "py_lib" / "wp_z3"))

from wp_z3_interface import TabletopWP
from z3_utils import Formula, FormulaType, PrologParser, Z3Converter, SATChecker, Z3_AVAILABLE
from z3 import And, Or, Not, Implies, Solver, sat, unsat, BoolVal

if not Z3_AVAILABLE:
    print("Error: Z3 not available. Install with: pip install z3-solver")
    sys.exit(1)


# =============================================================================
# TASK EXPRESSION PARSING
# =============================================================================

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
        return f"{cond_type}({', '.join(tokens[1:])})"


def parse_task_expression(task_str: str) -> Tuple[List[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse a task expression string.
    
    Returns:
        Tuple of (action_sequence, condition, then_action, else_action)
        - For simple sequences: (actions, None, None, None)
        - For conditionals: ([], condition, then_action, else_action)
        - For mixed: (actions_before, condition, then_action, else_action)
    """
    task_str = task_str.strip()
    
    # Split by semicolon for sequences
    parts = [p.strip() for p in task_str.split(';')]
    
    action_sequence = []
    condition = None
    then_action = None
    else_action = None
    
    for part in parts:
        tokens = part.split()
        
        if not tokens:
            continue
        
        if tokens[0] == 'if':
            # Parse if-then-else
            try:
                then_idx = tokens.index('then')
                else_idx = tokens.index('else')
                
                cond_tokens = tokens[1:then_idx]
                then_tokens = tokens[then_idx+1:else_idx]
                else_tokens = tokens[else_idx+1:]
                
                condition = parse_condition(cond_tokens)
                then_action = parse_action(then_tokens)
                else_action = parse_action(else_tokens)
            except (ValueError, IndexError) as e:
                # Fallback: treat as regular action
                action_sequence.append(parse_action(tokens))
        else:
            # Regular action
            action_sequence.append(parse_action(tokens))
    
    return (action_sequence, condition, then_action, else_action)


# =============================================================================
# WP COMPUTATION FOR CONDITIONAL PROGRAMS
# =============================================================================

def compute_wp_conditional(
    wp_interface: TabletopWP,
    condition: str,
    then_action: str,
    else_action: str,
    postcondition: str = "true"
) -> Formula:
    """
    Compute WP for conditional program: if cond then action1 else action2
    
    WP(if cond then a1 else a2, Q) = (cond ∧ WP(a1, Q)) ∨ (¬cond ∧ WP(a2, Q))
    """
    # Compute WP for both branches
    wp_then = wp_interface.compute_wp([then_action], postcondition)
    wp_else = wp_interface.compute_wp([else_action], postcondition)
    
    # Parse condition
    cond_formula = PrologParser.parse(condition)
    
    # Build: (cond ∧ WP_then) ∨ (¬cond ∧ WP_else)
    cond_and_wp_then = Formula(FormulaType.AND, [cond_formula, wp_then])
    neg_cond = Formula(FormulaType.NEG, [cond_formula])
    neg_cond_and_wp_else = Formula(FormulaType.AND, [neg_cond, wp_else])
    
    combined_wp = Formula(FormulaType.OR, [cond_and_wp_then, neg_cond_and_wp_else])
    
    return combined_wp


def compute_wp_for_task(
    wp_interface: TabletopWP,
    task_expression: str,
    postcondition: str = "true"
) -> Formula:
    """
    Compute WP for a task expression (handles sequences and conditionals)
    
    For sequences with conditionals: WP(seq; if-then-else, Q) = WP(seq, WP(if-then-else, Q))
    """
    action_sequence, condition, then_action, else_action = parse_task_expression(task_expression)
    
    if condition and then_action and else_action:
        # Conditional program
        # Compute WP for conditional: WP(if cond then a1 else a2, Q)
        wp_cond = compute_wp_conditional(wp_interface, condition, then_action, else_action, postcondition)
        
        if action_sequence:
            # There are actions before the conditional
            # WP(seq; if-then-else, Q) = WP(seq, WP(if-then-else, Q))
            # For now, use a conservative approach: AND of sequence WP and conditional WP
            # Note: This is an approximation. The correct approach would require
            # converting the conditional WP back to Prolog and using it as postcondition
            wp_seq = wp_interface.compute_wp(action_sequence, postcondition)
            return Formula(FormulaType.AND, [wp_seq, wp_cond])
        else:
            # Pure conditional
            return wp_cond
    else:
        # Simple action sequence
        if not action_sequence:
            # Empty program
            return Formula(FormulaType.TRUE, None)
        return wp_interface.compute_wp(action_sequence, postcondition)


# =============================================================================
# INITIAL CONDITIONS VALIDATION
# =============================================================================

def initial_conditions_to_z3(
    initial_conditions: Dict[str, bool],
    z3_converter: Z3Converter
) -> 'BoolRef':
    """
    Convert initial conditions dictionary to Z3 formula.
    
    Args:
        initial_conditions: Dict mapping atom strings to boolean values
        z3_converter: Z3Converter instance for atom conversion
    
    Returns:
        Z3 formula representing the conjunction of all initial conditions
    """
    conditions = []
    
    for atom_str, value in initial_conditions.items():
        atom_formula = PrologParser.parse(atom_str)
        z3_atom = z3_converter.to_z3(atom_formula)
        
        if value:
            conditions.append(z3_atom)
        else:
            conditions.append(Not(z3_atom))
    
    if not conditions:
        return BoolVal(True)
    
    return And(*conditions) if len(conditions) > 1 else conditions[0]


def collect_atoms_from_formula(formula: Formula) -> set:
    """
    Recursively collect all atom strings from a Formula.
    """
    atoms = set()
    
    if formula.type == FormulaType.ATOM:
        atoms.add(str(formula.content))
    elif formula.type in (FormulaType.TRUE, FormulaType.FALSE):
        pass  # No atoms
    elif formula.type == FormulaType.INEQ:
        pass  # Inequality doesn't contain atoms we need to track
    elif isinstance(formula.content, list):
        for child in formula.content:
            atoms.update(collect_atoms_from_formula(child))
    
    return atoms


def check_satisfies(
    wp_formula: Formula,
    initial_conditions: Dict[str, bool],
    wp_interface: TabletopWP
) -> Tuple[bool, Optional[str]]:
    """
    Check if initial conditions satisfy the WP formula.
    
    We check if initial_conditions ∧ WP is satisfiable under Closed World Assumption (CWA).
    For atoms in WP that are not specified in initial_conditions, we assume they are false.
    
    Note: This is a necessary condition. For a complete check, we would verify
    that initial_conditions → WP is valid, but checking satisfiability of
    initial_conditions ∧ WP is sufficient to detect violations.
    
    Returns:
        Tuple of (satisfies, error_message)
    """
    try:
        # Convert WP to Z3 (this will populate the converter's atom cache)
        z3_wp = wp_interface.to_z3(wp_formula)
        
        # Collect all atoms from WP formula
        wp_atoms = collect_atoms_from_formula(wp_formula)
        
        # Normalize initial condition keys (remove spaces for comparison)
        ic_normalized = {k.replace(" ", ""): v for k, v in initial_conditions.items()}
        
        # Apply Closed World Assumption: atoms in WP but not in initial_conditions are false
        extended_conditions = dict(initial_conditions)
        for atom_str in wp_atoms:
            atom_normalized = atom_str.replace(" ", "")
            if atom_normalized not in ic_normalized:
                # CWA: assume unspecified atoms are false
                extended_conditions[atom_str] = False
        
        # Convert extended initial conditions to Z3 (using the same converter)
        z3_initial = initial_conditions_to_z3(extended_conditions, wp_interface.z3_converter)
        
        # Check if initial_conditions ∧ WP is satisfiable
        # If unsatisfiable, then IC violates WP (IC → ¬WP is valid)
        solver = Solver()
        solver.add(z3_initial)
        solver.add(z3_wp)
        
        result = solver.check()
        
        if result == sat:
            # Satisfiable: initial conditions are consistent with the precondition
            return (True, None)
        elif result == unsat:
            # Unsatisfiable: initial conditions violate the precondition
            # This is not an error, just a normal result - return None for error
            return (False, None)
        else:
            # Unknown (shouldn't happen with Z3)
            return (False, "Solver returned unknown result")
    
    except Exception as e:
        import traceback
        return (False, f"Error during validation: {str(e)}\n{traceback.format_exc()}")


# =============================================================================
# MAIN VERIFICATION LOGIC
# =============================================================================

def verify_configuration_worker(args: Tuple[Dict[str, Any], str, str, bool]) -> Dict[str, Any]:
    """
    Worker function for parallel processing.
    Creates its own WP interface instance.
    
    Args:
        args: Tuple of (config, domain_file, postcondition, verbose)
    
    Returns:
        Dictionary with verification results
    """
    # Ensure imports are available in subprocess
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
    
    # Import necessary modules
    from wp_z3_interface import TabletopWP
    
    # Re-import current module to access verify_configuration function
    # This is needed because subprocesses may not have access to the module's functions
    import importlib.util
    module_file = os.path.abspath(__file__)
    spec = importlib.util.spec_from_file_location("verify_ct_config_worker", module_file)
    verify_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_module)
    
    config, domain_file, postcondition, verbose = args
    wp_interface = TabletopWP(domain_file)
    return verify_module.verify_configuration(config, wp_interface, postcondition, verbose)


def verify_configuration(
    config: Dict[str, Any],
    wp_interface: TabletopWP,
    postcondition: str = "true",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Verify a single configuration.
    
    Args:
        config: Configuration dictionary
        wp_interface: TabletopWP interface instance
        postcondition: Postcondition formula (default: "true")
        verbose: If True, print detailed processing information
    
    Returns:
        Dictionary with verification results
    """
    config_num = config.get("configuration_number", "unknown")
    task_expression = config.get("task_expression", "")
    initial_conditions = config.get("initial_conditions", {})
    rule_sequence = config.get("rule_sequence", "")
    raw_parameters = config.get("raw_parameters", {})
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing Configuration #{config_num}")
        print(f"{'='*80}")
        print(f"Task Expression: {task_expression}")
        if rule_sequence:
            print(f"Rule Sequence: {rule_sequence}")
        print(f"\nInitial Conditions:")
        for key, value in sorted(initial_conditions.items()):
            print(f"  {key}: {value}")
        if raw_parameters:
            print(f"\nRaw Parameters:")
            for key, value in sorted(raw_parameters.items()):
                print(f"  {key}: {value}")
        print(f"\nPostcondition: {postcondition}")
        print()
    
    result = {
        "configuration_number": config_num,
        "task_expression": task_expression,
        "valid": False,
        "error": None,
        "wp_formula": None,
        "satisfies_precondition": False,
        "rule_sequence": rule_sequence,
        "initial_conditions": initial_conditions,
        "raw_parameters": raw_parameters
    }
    
    if not task_expression:
        result["error"] = "No task expression found"
        if verbose:
            print("ERROR: No task expression found")
        return result
    
    if not initial_conditions:
        result["error"] = "No initial conditions found"
        if verbose:
            print("ERROR: No initial conditions found")
        return result
    
    try:
        if verbose:
            print("Step 1: Parsing task expression...")
            action_sequence, condition, then_action, else_action = parse_task_expression(task_expression)
            if condition:
                print(f"  Detected conditional: if {condition} then {then_action} else {else_action}")
            if action_sequence:
                print(f"  Action sequence: {action_sequence}")
            print()
        
        # Compute WP
        if verbose:
            print("Step 2: Computing weakest precondition (WP)...")
        wp_formula = compute_wp_for_task(wp_interface, task_expression, postcondition)
        result["wp_formula"] = str(wp_formula)
        
        if verbose:
            print(f"  WP Formula: {wp_formula}")
            print()
        
        # Check if initial conditions satisfy WP
        if verbose:
            print("Step 3: Checking if initial conditions satisfy WP...")
            print("  Converting initial conditions to Z3 formula...")
        
        satisfies, error = check_satisfies(wp_formula, initial_conditions, wp_interface)
        result["satisfies_precondition"] = satisfies
        
        if verbose:
            if satisfies:
                print("  ✓ Initial conditions satisfy the precondition")
            else:
                print(f"  ✗ Initial conditions do NOT satisfy the precondition")
                if error:
                    print(f"  Error: {error}")
            print()
        
        if error:
            result["error"] = error
        else:
            # valid is True only if satisfies is True and no error
            result["valid"] = satisfies
        
        if verbose:
            print(f"Result: {'VALID' if result['valid'] else 'INVALID'}")
            if result["error"]:
                print(f"Error: {result['error']}")
            print(f"{'='*80}\n")
    
    except Exception as e:
        result["error"] = f"Exception during verification: {str(e)}"
        import traceback
        result["traceback"] = traceback.format_exc()
        if verbose:
            print(f"EXCEPTION: {str(e)}")
            print(traceback.format_exc())
            print(f"{'='*80}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify CT configurations by checking preconditions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to simp_ct_configurations.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON). If not specified, prints to stdout."
    )
    parser.add_argument(
        "--postcondition", "-p",
        type=str,
        default="true",
        help="Postcondition formula (default: 'true')"
    )
    parser.add_argument(
        "--domain-file",
        type=str,
        default=None,
        help="Path to main.pl file (default: prolog_lib/domains/main.pl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of configurations to verify (for testing)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count). Use 1 for sequential processing."
    )
    parser.add_argument(
        "--config-number", "-n",
        type=int,
        default=None,
        help="Process only a specific configuration by its number (1-indexed). Enables verbose output automatically."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed processing information for each configuration"
    )
    
    args = parser.parse_args()
    
    # If config-number is specified, enable verbose mode automatically
    if args.config_number is not None:
        args.verbose = True
    
    # Resolve paths
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        # First try relative to current working directory
        cwd_path = Path.cwd() / input_path
        if cwd_path.exists():
            input_path = cwd_path
        else:
            # Fallback to relative to script directory
            input_path = Path(__file__).parent.parent / input_path
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Initialize WP interface
    if args.domain_file:
        domain_file = Path(args.domain_file)
        if not domain_file.is_absolute():
            domain_file = Path(__file__).parent.parent.parent / domain_file
    else:
        domain_file = Path(__file__).parent.parent.parent / "prolog_lib" / "domains" / "main.pl"
    
    if not domain_file.exists():
        print(f"Error: Domain file not found: {domain_file}")
        sys.exit(1)
    
    print(f"Loading configurations from: {input_path}")
    print(f"Using domain file: {domain_file}")
    print(f"Postcondition: {args.postcondition}")
    print("=" * 60)
    
    # Load configurations
    with open(input_path, 'r', encoding='utf-8') as f:
        configurations = json.load(f)
    
    if not isinstance(configurations, list):
        print("Error: Expected JSON array of configurations")
        sys.exit(1)
    
    # Filter by configuration number if specified
    if args.config_number is not None:
        config_num = args.config_number
        matching_configs = [c for c in configurations if c.get("configuration_number") == config_num]
        if not matching_configs:
            print(f"Error: Configuration #{config_num} not found in file")
            print(f"Available configuration numbers: {sorted(set(c.get('configuration_number', 'unknown') for c in configurations))[:20]}...")
            sys.exit(1)
        configurations = matching_configs
        print(f"Processing configuration #{config_num} only")
    
    # Limit if specified
    if args.limit:
        configurations = configurations[:args.limit]
    
    print(f"Verifying {len(configurations)} configuration(s)...")
    
    # Determine number of workers
    if args.workers is None:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = args.workers
    
    if num_workers == 1:
        print("Using sequential processing (1 worker)")
    else:
        print(f"Using parallel processing with {num_workers} workers")
    print()
    
    # Prepare arguments for workers
    worker_args = [(config, str(domain_file), args.postcondition, args.verbose) for config in configurations]
    
    # Verify configurations
    results = []
    valid_count = 0
    invalid_count = 0
    error_count = 0
    
    # If verbose mode or single config, use sequential processing
    if args.verbose or args.config_number is not None or num_workers == 1:
        # Sequential processing (for debugging or single-threaded execution)
        try:
            wp_interface = TabletopWP(str(domain_file))
        except Exception as e:
            print(f"Error initializing WP interface: {e}")
            sys.exit(1)
        
        for i, config in enumerate(configurations):
            if not args.verbose and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(configurations)} configurations...")
            
            result = verify_configuration(config, wp_interface, args.postcondition, args.verbose)
            results.append(result)
            
            if result["error"]:
                error_count += 1
            elif result["satisfies_precondition"]:
                valid_count += 1
            else:
                invalid_count += 1
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(verify_configuration_worker, args): i 
                for i, args in enumerate(worker_args)
            }
            
            # Process completed tasks
            completed = 0
            results = [None] * len(configurations)  # Pre-allocate to maintain order
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    
                    if result["error"]:
                        error_count += 1
                    elif result["satisfies_precondition"]:
                        valid_count += 1
                    else:
                        invalid_count += 1
                    
                    completed += 1
                    if completed % 100 == 0:
                        print(f"Processed {completed}/{len(configurations)} configurations...")
                except Exception as e:
                    # Handle errors in worker
                    config_num = configurations[index].get("configuration_number", "unknown")
                    results[index] = {
                        "configuration_number": config_num,
                        "task_expression": configurations[index].get("task_expression", ""),
                        "valid": False,
                        "error": f"Worker exception: {str(e)}",
                        "wp_formula": None,
                        "satisfies_precondition": False
                    }
                    error_count += 1
                    completed += 1
    
    # Summary
    print()
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total configurations: {len(configurations)}")
    print(f"Valid (satisfies precondition): {valid_count}")
    print(f"Invalid (does not satisfy precondition): {invalid_count}")
    print(f"Errors: {error_count}")
    
    # Output results
    output_data = {
        "summary": {
            "total": len(configurations),
            "valid": valid_count,
            "invalid": invalid_count,
            "errors": error_count
        },
        "results": results
    }
    
    if args.output:
        output_path = Path(args.output)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Verification results saved to: {output_path}")
    else:
        # Print to stdout
        print(json.dumps(outpust_data, indent=2))


if __name__ == "__main__":
    main()

