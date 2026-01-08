#!/usr/bin/env python3
"""
Tabletop Domain WP Z3 Tool

A command-line tool to compute Weakest Precondition (WP) for the Tabletop domain
and check satisfiability using Z3 solver.

Usage:
    python wp_z3_example.py "['put(bread, plate)']"
    python wp_z3_example.py "['put(bread, plate)', 'put(plate, microwave)']"
    python wp_z3_example.py "['put(bread, plate)']" --postcondition "loc(bread, plate)"
"""

import argparse
import ast
from wp_z3_interface import TabletopWP


def compute_wp(program_str: str, postcondition: str = "true"):
    """
    Compute WP from command line arguments
    
    Args:
        program_str: Program as Python list string, e.g., "['put(bread, plate)']"
        postcondition: Postcondition formula, default is "true"
    """
    # Parse program string to Python list
    try:
        program = ast.literal_eval(program_str)
        if not isinstance(program, list):
            program = [program]
    except (ValueError, SyntaxError):
        # If parsing fails, treat as single action string
        program = [program_str]
    
    interface = TabletopWP()
    
    try:
        print(f"Program: {program}")
        print(f"Postcondition: {postcondition}")
        print("=" * 60)
        
        # Compute WP
        wp = interface.compute_wp(program, postcondition)
        
        print("\nWP Formula Structure:")
        print(interface.print_formula(wp))
        
        # Convert to Z3
        z3_formula = interface.to_z3(wp)
        print("\nZ3 Formula:")
        print(z3_formula)
        
        # Check satisfiability
        print("\nChecking satisfiability...")
        result = interface.get_model(wp)
        
        if result:
            print("\n✓ Satisfiable! Model:")
            for atom, value in sorted(result.items()):
                print(f"  {atom} = {value}")
        else:
            print("\n✗ Unsatisfiable")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Tabletop Domain WP Z3 Tool - Compute Weakest Precondition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wp_z3_example.py "['put(bread, plate)']"
  python wp_z3_example.py "['put(bread, plate)', 'put(plate, microwave)']"
  python wp_z3_example.py "['close(microwave)', 'turn_on(microwave)']" --postcondition "running(microwave)"
        """
    )
    parser.add_argument(
        "program", 
        type=str, 
        help="Prolog program as Python list string, e.g., \"['put(bread, plate)']\""
    )
    parser.add_argument(
        "--postcondition", "-p",
        type=str,
        default="true",
        help="Postcondition formula (default: 'true')"
    )
    
    args = parser.parse_args()
    compute_wp(args.program, args.postcondition)


if __name__ == "__main__":
    main()

