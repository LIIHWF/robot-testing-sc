#!/usr/bin/env python3
"""
ACTS Combinatorial Testing Model to ASP (Answer Set Programming) Converter
Enumerates all feasible configurations using the Clingo solver

Usage:
    python py_tools/enumeration/acts_to_asp_enumerator.py model.txt [-n MAX] [-o output.json] [--asp-only]
"""

import re
import subprocess
import sys
import json
import argparse
from collections import OrderedDict


def parse_acts_model(content):
    """Parse ACTS model file and extract parameters and constraints"""
    lines = content.strip().split('\n')
    
    parameters = OrderedDict()
    constraints = []
    system_name = None
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].lower()
            continue
        
        if current_section == 'system':
            # Parse system name
            if line.startswith('Name:'):
                system_name = line[5:].strip()
        
        elif current_section == 'parameter':
            # Parse parameter line: ParamName (enum) : value1,value2,...
            match = re.match(r'(\S+)\s*\(enum\)\s*:\s*(.+)', line)
            if match:
                param_name = match.group(1)
                values = [v.strip() for v in match.group(2).split(',')]
                parameters[param_name] = values
        
        elif current_section == 'constraint':
            if line:
                constraints.append(line)
    
    return system_name, parameters, constraints


def sanitize_name(name):
    """Convert a name to a valid ASP atom name (lowercase, no special chars)"""
    # Replace special characters and convert to lowercase
    result = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()
    # Ensure it starts with a letter
    if result and result[0].isdigit():
        result = 'v_' + result
    return result


def is_simple_equality_constraint(constraint):
    """Check if constraint is a simple equality (Param="value" without ||)"""
    if '||' in constraint:
        return False
    return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*="[^"]*"$', constraint.strip()) is not None


def convert_to_asp(parameters, constraints, verbose=False, skip_simple_eq=False, show_contradictions=False):
    """Convert ACTS model to ASP program"""
    asp_lines = []
    
    # Header comment
    asp_lines.append("% ACTS Model converted to ASP")
    asp_lines.append("% Each answer set represents a valid configuration")
    asp_lines.append("")
    
    # Define parameter domains
    asp_lines.append("% Parameter definitions")
    param_map = {}  # Map original param names to sanitized names
    value_map = {}  # Map original values to sanitized values for each param
    
    for param, values in parameters.items():
        sanitized_param = sanitize_name(param)
        param_map[param] = sanitized_param
        
        value_map[param] = {}
        for val in values:
            sanitized_val = sanitize_name(val)
            value_map[param][val] = sanitized_val
            # Define each possible value for the parameter
            asp_lines.append(f"param_value({sanitized_param}, {sanitized_val}).")
    
    asp_lines.append("")
    
    # Generate choice rules: exactly one value per parameter
    asp_lines.append("% Each parameter must have exactly one value")
    for param in parameters:
        sanitized_param = param_map[param]
        asp_lines.append(f"1 {{ assign({sanitized_param}, V) : param_value({sanitized_param}, V) }} 1.")
    
    asp_lines.append("")
    
    # Detect contradictory simple equality constraints
    if show_contradictions or skip_simple_eq:
        eq_constraints = {}  # param -> list of values
        for constraint in constraints:
            if is_simple_equality_constraint(constraint):
                match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', constraint.strip())
                if match:
                    param, value = match.groups()
                    if param not in eq_constraints:
                        eq_constraints[param] = []
                    eq_constraints[param].append(value)
        
        # Report contradictions
        contradictions = {p: v for p, v in eq_constraints.items() if len(v) > 1}
        if contradictions and show_contradictions:
            print("Detected contradictory simple equality constraints:")
            for param, values in contradictions.items():
                print(f"  {param} must equal ALL of: {values}")
    
    # Convert constraints
    asp_lines.append("% Constraints from ACTS model")
    
    converted_count = 0
    skipped_count = 0
    skipped_eq_count = 0
    
    for idx, constraint in enumerate(constraints):
        # Skip simple equality constraints if requested
        if skip_simple_eq and is_simple_equality_constraint(constraint):
            skipped_eq_count += 1
            continue
            
        asp_constraint = convert_constraint(constraint, param_map, value_map)
        if asp_constraint:
            asp_lines.append(f"% C{idx+1}: {constraint[:80]}{'...' if len(constraint) > 80 else ''}")
            asp_lines.append(asp_constraint)
            converted_count += 1
        else:
            if verbose:
                print(f"Warning: Could not convert constraint: {constraint}")
            skipped_count += 1
    
    asp_lines.append("")
    
    # Show directive to output assignments
    asp_lines.append("% Output")
    asp_lines.append("#show assign/2.")
    
    if verbose:
        print(f"Converted {converted_count} constraints, skipped {skipped_count}")
        if skip_simple_eq:
            print(f"Skipped {skipped_eq_count} simple equality constraints")
    
    return '\n'.join(asp_lines), param_map, value_map


def parse_literal(lit, param_map, value_map):
    """
    Parse a single literal (Param="value" or Param!="value")
    Also handles spaced format: Param = "value" or Param != "value"
    Returns (is_equality, sanitized_param, sanitized_value) or None
    """
    lit = lit.strip()
    
    # Match Param!="value" or Param != "value" (inequality)
    neq_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*!=\s*"([^"]*)"', lit)
    if neq_match:
        param, value = neq_match.groups()
        if param in param_map and value in value_map.get(param, {}):
            return (False, param_map[param], value_map[param][value])
        return None  # param or value not found
    
    # Match Param="value" or Param = "value" (equality)
    eq_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"]*)"', lit)
    if eq_match:
        param, value = eq_match.groups()
        if param in param_map and value in value_map.get(param, {}):
            return (True, param_map[param], value_map[param][value])
        return None  # param or value not found
    
    return None  # No match at all


def convert_constraint(constraint, param_map, value_map):
    """
    Convert a single ACTS constraint to ASP rule.
    
    ACTS constraints can be:
    1. Disjunctions: L1 || L2 || ... || Ln
    2. Implications: (A && B && ...) => C  which becomes !A || !B || ... || C
    3. Negations: !(A && B && ...) which becomes !A || !B || ...
    
    Each literal is either Param="value" or Param!="value"
    
    The constraint means: at least one of the literals must be true.
    
    In ASP, we express this as an integrity constraint:
    :- not L1, not L2, ..., not Ln
    (It's a violation if all literals are false)
    """
    constraint = constraint.strip()
    
    # Handle implication: (condition) => consequence
    impl_match = re.match(r'^\((.+)\)\s*=>\s*(.+)$', constraint)
    if impl_match:
        antecedent, consequent = impl_match.groups()
        # (A && B) => C becomes !A || !B || C
        # Parse antecedent as conjunction
        ant_parts = re.split(r'\s*&&\s*', antecedent)
        # Negate each part of antecedent and combine with consequent
        disjuncts = []
        for part in ant_parts:
            part = part.strip()
            # Negate: Param="value" becomes Param!="value"
            if '!=' not in part and '=' in part:
                negated = part.replace('=', '!=', 1)
                disjuncts.append(negated)
            elif '!=' in part:
                negated = part.replace('!=', '=', 1)
                disjuncts.append(negated)
        # Add consequent
        disjuncts.append(consequent.strip())
        constraint = ' || '.join(disjuncts)
    
    # Handle negation of conjunction: !(A && B && ...)
    neg_match = re.match(r'^!\((.+)\)$', constraint)
    if neg_match:
        inner = neg_match.group(1)
        # !(A && B) becomes !A || !B
        parts = re.split(r'\s*&&\s*', inner)
        disjuncts = []
        for part in parts:
            part = part.strip()
            if '!=' not in part and '=' in part:
                negated = part.replace('=', '!=', 1)
                disjuncts.append(negated)
            elif '!=' in part:
                negated = part.replace('!=', '=', 1)
                disjuncts.append(negated)
        constraint = ' || '.join(disjuncts)
    
    # Parse literals (split by ||)
    literals = re.split(r'\s*\|\|\s*', constraint)
    
    body_parts = []
    
    for lit in literals:
        # Remove parentheses
        lit = lit.strip().strip('()')
        parsed = parse_literal(lit, param_map, value_map)
        if parsed is None:
            continue
        
        is_eq, sanitized_param, sanitized_val = parsed
        
        if is_eq:
            # Original: Param="value" (this should be true)
            # Negation for body: not assign(param, value)
            body_parts.append(f"not assign({sanitized_param}, {sanitized_val})")
        else:
            # Original: Param!="value" (this should be true, i.e., param != value)
            # Negation for body: assign(param, value)
            body_parts.append(f"assign({sanitized_param}, {sanitized_val})")
    
    if not body_parts:
        return None
    
    # The integrity constraint: violation if all literals in disjunction are false
    return f":- {', '.join(body_parts)}."


def run_clingo(asp_program, max_solutions=0, timeout=None):
    """Run clingo using Python API and return all answer sets
    
    Ensures deterministic results by:
    1. Setting a fixed random seed (--seed=42)
    2. Sorting answer sets lexicographically for consistent ordering
    """
    try:
        import clingo
        import time
        
        start_time = time.time()
        answer_sets = []
        
        # Create clingo control object with appropriate arguments
        args = []
        if max_solutions > 0:
            args.append(f'-n{max_solutions}')
        else:
            args.append('-n0')  # 0 means all solutions
        
        # Set fixed random seed for deterministic behavior
        # Using a non-zero seed value (42) to explicitly set a fixed seed
        # This ensures reproducible results across multiple runs
        args.append('--seed=42')
        
        ctl = clingo.Control(args)
        
        # Add the program
        ctl.add("base", [], asp_program)
        
        # Ground the program
        ctl.ground([("base", [])])
        
        # Solve and collect all models
        def on_model(model):
            # Collect the shown atoms
            atoms = [str(atom) for atom in model.symbols(shown=True)]
            answer_sets.append(atoms)
        
        # Solve - returns SolveResult directly
        solve_result = ctl.solve(on_model=on_model)
        
        # Sort answer sets lexicographically for deterministic ordering
        # Each answer set is represented as a list of atom strings
        answer_sets.sort()
        
        end_time = time.time()
        
        # Build output info similar to JSON output
        result_str = 'UNKNOWN'
        if solve_result.satisfiable:
            result_str = 'SATISFIABLE'
        elif solve_result.unsatisfiable:
            result_str = 'UNSATISFIABLE'
        
        output = {
            'Result': result_str,
            'Models': {'Number': len(answer_sets)},
            'Time': {'Total': end_time - start_time}
        }
        
        return answer_sets, output
        
    except ImportError:
        print("Error: clingo Python package not found. Please install it:")
        print("  pip install clingo")
        sys.exit(1)
    except Exception as e:
        print(f"Error during solving: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def decode_answer_set(answer_set, param_map, value_map, parameters):
    """Decode answer set atoms back to original parameter names and values"""
    # Reverse the mappings
    reverse_param = {v: k for k, v in param_map.items()}
    reverse_value = {}
    for param, vals in value_map.items():
        reverse_value[param] = {v: k for k, v in vals.items()}
    
    temp_config = {}
    
    for atom in answer_set:
        # Parse assign(param, value)
        match = re.match(r'assign\((\w+),(\w+)\)', atom)
        if match:
            sanitized_param, sanitized_val = match.groups()
            if sanitized_param in reverse_param:
                original_param = reverse_param[sanitized_param]
                if original_param in reverse_value and sanitized_val in reverse_value[original_param]:
                    original_val = reverse_value[original_param][sanitized_val]
                    temp_config[original_param] = original_val
    
    # Return config in original parameter order
    config = OrderedDict()
    for param in parameters:
        if param in temp_config:
            config[param] = temp_config[param]
    
    return config


def config_sort_key(config):
    """Generate a sort key for a configuration to ensure deterministic ordering.
    
    Returns a tuple of (param_name, param_value) pairs sorted by parameter name.
    This ensures configurations are sorted consistently.
    """
    return tuple(sorted((param, str(val)) for param, val in config.items()))


def format_config_csv(config, parameters):
    """Format a configuration as CSV row"""
    values = []
    for param in parameters:
        values.append(config.get(param, ""))
    return ','.join(values)


def main():
    parser = argparse.ArgumentParser(
        description='Enumerate all feasible configurations for an ACTS combinatorial testing model using ASP (Answer Set Programming)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s model.txt                    # Enumerate all configurations
  %(prog)s model.txt -n 10              # Get first 10 configurations  
  %(prog)s model.txt -o configs.json    # Save to JSON file
  %(prog)s model.txt --asp-only         # Only generate ASP program
  %(prog)s model.txt --csv              # Output as CSV
'''
    )
    parser.add_argument('model_file', help='Path to the ACTS model file')
    parser.add_argument('-n', '--max-solutions', type=int, default=0,
                        help='Maximum number of solutions (0 = all, default: 0)')
    parser.add_argument('-o', '--output', help='Output file for configurations')
    parser.add_argument('--csv', action='store_true',
                        help='Output in CSV format')
    parser.add_argument('--asp-only', action='store_true',
                        help='Only generate ASP program, do not solve')
    parser.add_argument('--asp-file', help='Save generated ASP program to file')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds for solver')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--count-only', action='store_true',
                        help='Only count solutions, do not output them')
    parser.add_argument('--skip-simple-eq', action='store_true',
                        help='Skip simple equality constraints (Param="value" without ||)')
    parser.add_argument('--show-contradictions', action='store_true',
                        help='Show detected contradictory constraints')
    
    args = parser.parse_args()
    
    # Read model file
    print(f"Reading model from {args.model_file}...")
    with open(args.model_file, 'r') as f:
        content = f.read()
    
    # Parse model
    system_name, parameters, constraints = parse_acts_model(content)
    
    print(f"System: {system_name or 'unnamed'}")
    print(f"Parameters: {len(parameters)}")
    if args.verbose:
        for p, vals in parameters.items():
            print(f"  {p}: {len(vals)} values ({', '.join(vals[:5])}{'...' if len(vals) > 5 else ''})")
    print(f"Constraints: {len(constraints)}")
    
    # Calculate theoretical maximum (without constraints)
    max_configs = 1
    for vals in parameters.values():
        max_configs *= len(vals)
    print(f"Theoretical maximum configurations (no constraints): {max_configs:,}")
    print()
    
    # Convert to ASP
    print("Converting to ASP program...")
    asp_program, param_map, value_map = convert_to_asp(
        parameters, constraints, 
        verbose=args.verbose,
        skip_simple_eq=args.skip_simple_eq,
        show_contradictions=args.show_contradictions
    )
    
    # Save ASP program if requested
    if args.asp_file:
        with open(args.asp_file, 'w') as f:
            f.write(asp_program)
        print(f"ASP program saved to {args.asp_file}")
    
    if args.asp_only:
        print("\n" + "="*60)
        print(asp_program)
        return
    
    # Run solver
    print("Running clingo solver...")
    if args.max_solutions == 0:
        print("(Searching for ALL solutions, this may take a while...)")
    else:
        print(f"(Searching for up to {args.max_solutions} solutions)")
    
    answer_sets, clingo_output = run_clingo(asp_program, args.max_solutions, args.timeout)
    
    # Get statistics
    solving_info = clingo_output.get('Result', 'UNKNOWN')
    models_info = clingo_output.get('Models', {})
    num_models = models_info.get('Number', 0)
    time_info = clingo_output.get('Time', {})
    total_time = time_info.get('Total', 0)
    
    print()
    print(f"Result: {solving_info}")
    if args.max_solutions > 0 and num_models >= args.max_solutions:
        print(f"Generated {num_models:,} configuration(s) (limit reached, more may exist)")
    else:
        print(f"Total feasible configurations: {num_models:,}")
    print(f"Solving time: {total_time:.3f}s")
    
    if args.count_only:
        return
    
    if answer_sets:
        # Decode configurations
        configurations = []
        for ans in answer_sets:
            config = decode_answer_set(ans, param_map, value_map, parameters)
            configurations.append(config)
        
        # Sort configurations for deterministic ordering
        # This ensures consistent output even if answer set order varies
        configurations.sort(key=config_sort_key)
        
        # Output configurations
        if args.output:
            if args.csv:
                # CSV output
                with open(args.output, 'w') as f:
                    # Header
                    f.write(','.join(parameters.keys()) + '\n')
                    # Rows
                    for config in configurations:
                        f.write(format_config_csv(config, parameters) + '\n')
                print(f"Configurations saved to {args.output} (CSV format)")
            else:
                # JSON output
                with open(args.output, 'w') as f:
                    json.dump(configurations, f, indent=2)
                print(f"Configurations saved to {args.output} (JSON format)")
        else:
            if len(configurations) <= 20 or args.verbose:
                print("\nConfigurations:")
                for i, config in enumerate(configurations, 1):
                    print(f"\n--- Configuration {i} ---")
                    for param, val in config.items():
                        print(f"  {param} = {val}")
            else:
                print(f"\n(Showing first 5 of {len(configurations)} configurations)")
                for i, config in enumerate(configurations[:5], 1):
                    print(f"\n--- Configuration {i} ---")
                    for param, val in config.items():
                        print(f"  {param} = {val}")
                print("\n... use -o to save all configurations to a file")
    else:
        if solving_info == 'UNSATISFIABLE':
            print("\nNo feasible configurations exist! The constraints are contradictory.")
        else:
            print("\nNo configurations found.")


if __name__ == '__main__':
    main()

