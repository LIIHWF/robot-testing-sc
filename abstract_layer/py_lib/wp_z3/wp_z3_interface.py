#!/usr/bin/env python3
"""
Tabletop Domain WP-Z3 Interface

This module provides the interface between Prolog WP computation and Z3 
for the Tabletop domain. It leverages the generic z3_utils module.

Usage:
    from wp_z3_interface import TabletopWP
    
    wp_interface = TabletopWP()
    
    # Compute WP and get Formula
    wp_formula = wp_interface.compute_wp(['put(bread, plate)'], 'loc(bread, plate)')
    
    # Convert to Z3 and check satisfiability
    z3_formula = wp_interface.to_z3(wp_formula)
    is_sat = wp_interface.is_satisfiable(wp_formula)
"""

import subprocess
import tempfile
import os
from typing import Any, Dict, List, Optional, Union

from z3_utils import (
    Formula, FormulaType, PrologParser, Z3Converter, CNFConverter, SATChecker,
    check_z3_available, Z3_AVAILABLE
)

if Z3_AVAILABLE:
    from z3 import BoolRef


# =============================================================================
# PROLOG INTERFACE
# =============================================================================

class PrologInterface:
    """Interface for running Prolog queries"""
    
    def __init__(self, domain_file: str):
        """
        Initialize Prolog interface
        
        Args:
            domain_file: Path to the domain Prolog file (e.g., main.pl)
        """
        self.domain_file = os.path.abspath(domain_file)
        self.domain_dir = os.path.dirname(self.domain_file)
        self.prolog_path = self._find_prolog()
    
    def _find_prolog(self) -> str:
        """Find SWI-Prolog executable"""
        for cmd in ['swipl', 'swi-prolog']:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        raise RuntimeError(
            "SWI-Prolog not found. Install from: https://www.swi-prolog.org/download/stable"
        )
    
    def query_wp(self, program: List[str], postcondition: str) -> str:
        """
        Query Prolog for WP computation
        
        Args:
            program: List of actions as strings, e.g., ['put(bread, plate)']
            postcondition: Postcondition as string, e.g., 'loc(bread, plate)'
            
        Returns:
            WP formula as Prolog term string
        """
        # Format program as Prolog list
        prog_str = "[" + ", ".join(program) + "]"
        
        # Create temporary Prolog script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
            script = f""":- use_module(library(lists)).
:- ['{self.domain_file}'].

main :-
    wp({prog_str}, {postcondition}, WP),
    write(WP),
    nl,
    halt.

main :- 
    write('WP computation failed'),
    nl,
    halt(1).
"""
            f.write(script)
            script_file = f.name
        
        try:
            result = subprocess.run(
                [self.prolog_path, '-g', 'main', '-t', 'halt', script_file],
                cwd=self.domain_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Check for Prolog errors and warnings first
            combined_output = result.stdout + result.stderr
            errors, warnings = self._check_prolog_errors(combined_output)
            
            # Report warnings
            for warning in warnings:
                print(f"[Prolog Warning] {warning}")
            
            # Raise on errors
            if errors:
                error_details = "\n  ".join(errors)
                raise RuntimeError(f"Prolog errors detected:\n  {error_details}")
            
            # Check return code
            if result.returncode != 0:
                raise RuntimeError(
                    f"Prolog exited with code {result.returncode}.\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
            
            # Parse output
            output = self._parse_output(combined_output)
            
            if not output:
                error_msg = result.stderr or "No output"
                raise RuntimeError(f"Empty WP result. Error: {error_msg}")
            
            return output
            
        finally:
            if os.path.exists(script_file):
                os.unlink(script_file)
    
    def _check_prolog_errors(self, output: str) -> tuple:
        """
        Check Prolog output for errors and warnings.
        
        Returns:
            Tuple of (errors, warnings) where each is a list of strings.
        """
        errors = []
        warnings = []
        
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Collect errors
            if line.startswith('ERROR:') or 'ERROR' in line:
                errors.append(line)
            elif 'Syntax error' in line or 'syntax error' in line:
                errors.append(line)
            elif 'Unknown procedure' in line:
                errors.append(line)
            elif 'WP computation failed' in line:
                errors.append(line)
            # Collect warnings
            elif line.startswith('Warning:'):
                warnings.append(line)
        
        return errors, warnings
    
    def _parse_output(self, output: str) -> str:
        """Extract WP formula from Prolog output"""
        result_lines = []
        
        for line in output.split('\n'):
            line = line.strip()
            # Skip empty, warnings, comments, and error messages
            if not line:
                continue
            if line.startswith('%') or line.startswith('Warning:') or line.startswith('ERROR:'):
                continue
            # Skip lines with error messages
            if 'Unknown procedure' in line or 'ERROR' in line:
                continue
            # Check if line looks like a valid formula (starts with known constructs)
            valid_starts = ['and(', 'or(', 'neg(', 'impl(', 'loc(', 
                           'door_open(', 'running(', 'empty(', 'in(',
                           'true', 'false']
            if any(line.startswith(s) for s in valid_starts):
                result_lines.append(line)
            # Also handle lines that are pure atoms like door_open(microwave)
            elif '(' in line and ')' in line and not any(c in line for c in [':', '-', 'ERROR']):
                result_lines.append(line)
        
        return ' '.join(result_lines).strip()


# =============================================================================
# TABLETOP WP INTERFACE
# =============================================================================

class TabletopWP:
    """
    Main interface for Tabletop domain WP computation and Z3 conversion
    
    This class combines:
    - Prolog WP computation
    - Formula parsing
    - Z3 conversion
    - Satisfiability checking
    """
    
    def __init__(self, domain_file: Optional[str] = None):
        """
        Initialize TabletopWP interface
        
        Args:
            domain_file: Path to main.pl. If None, uses default location.
        """
        if domain_file is None:
            # Default to prolog_lib/domains/main.pl
            domain_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'prolog_lib', 'domains', 'main.pl'
            )
        
        self.prolog = PrologInterface(domain_file)
        self.z3_converter = Z3Converter() if Z3_AVAILABLE else None
        self.cnf_converter = CNFConverter() if Z3_AVAILABLE else None
        self.sat_checker = SATChecker() if Z3_AVAILABLE else None
    
    # -------------------------------------------------------------------------
    # WP COMPUTATION
    # -------------------------------------------------------------------------
    
    def compute_wp(self, program: List[str], postcondition: str) -> Formula:
        """
        Compute weakest precondition via Prolog
        
        Args:
            program: List of actions, e.g., ['put(bread, plate)', 'close(microwave)']
            postcondition: Postcondition formula, e.g., 'loc(bread, plate)'
            
        Returns:
            Formula object representing the WP
        """
        wp_str = self.prolog.query_wp(program, postcondition)
        return PrologParser.parse(wp_str)
    
    def compute_wp_string(self, program: List[str], postcondition: str) -> str:
        """
        Compute WP and return as string (useful for debugging)
        """
        return self.prolog.query_wp(program, postcondition)
    
    # -------------------------------------------------------------------------
    # Z3 CONVERSION
    # -------------------------------------------------------------------------
    
    def to_z3(self, formula: Formula) -> 'BoolRef':
        """
        Convert Formula to Z3 BoolRef
        
        Args:
            formula: Formula object
            
        Returns:
            Z3 BoolRef expression
        """
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available. Install with: pip install z3-solver")
        return self.z3_converter.to_z3(formula)
    
    def wp_to_z3(self, program: List[str], postcondition: str) -> 'BoolRef':
        """
        Compute WP and convert directly to Z3
        
        Args:
            program: List of actions
            postcondition: Postcondition formula
            
        Returns:
            Z3 BoolRef expression
        """
        formula = self.compute_wp(program, postcondition)
        return self.to_z3(formula)
    
    # -------------------------------------------------------------------------
    # SATISFIABILITY CHECKING
    # -------------------------------------------------------------------------
    
    def is_satisfiable(self, formula: Formula) -> bool:
        """Check if formula is satisfiable"""
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available")
        z3_fml = self.to_z3(formula)
        return self.sat_checker.check_sat(z3_fml)
    
    def get_model(self, formula: Formula) -> Optional[Dict[str, bool]]:
        """
        Get satisfying assignment if formula is SAT
        
        Returns:
            Dictionary mapping atom names to boolean values, or None if UNSAT
        """
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available")
        z3_fml = self.to_z3(formula)
        atoms = self.z3_converter.get_atoms()
        return self.sat_checker.get_model(z3_fml, atoms)
    
    def solve_wp(self, program: List[str], postcondition: str) -> Optional[Dict[str, bool]]:
        """
        Compute WP and find satisfying initial state
        
        Args:
            program: List of actions
            postcondition: Postcondition formula
            
        Returns:
            Dictionary of atom values if satisfiable, None otherwise
        """
        formula = self.compute_wp(program, postcondition)
        return self.get_model(formula)
    
    # -------------------------------------------------------------------------
    # CNF CONVERSION
    # -------------------------------------------------------------------------
    
    def to_cnf(self, formula: Formula) -> List['BoolRef']:
        """
        Convert formula to CNF (Conjunctive Normal Form)
        
        Args:
            formula: Formula object
            
        Returns:
            List of CNF clauses (disjunctions)
        """
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available")
        z3_fml = self.to_z3(formula)
        return self.cnf_converter.simplify_to_cnf(z3_fml)
    
    def wp_to_cnf(self, program: List[str], postcondition: str) -> List['BoolRef']:
        """
        Compute WP and convert to CNF
        """
        formula = self.compute_wp(program, postcondition)
        return self.to_cnf(formula)
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def get_atoms(self) -> Dict[str, 'BoolRef']:
        """Get all Z3 atoms created so far"""
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available")
        return self.z3_converter.get_atoms()
    
    def print_formula(self, formula: Formula) -> str:
        """Pretty print a formula"""
        return formula.pretty_print()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_interface(domain_file: Optional[str] = None) -> TabletopWP:
    """Create a TabletopWP interface with optional custom domain file"""
    return TabletopWP(domain_file)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the WP-Z3 interface"""
    if not Z3_AVAILABLE:
        print("Z3 not available. Install with: pip install z3-solver")
        return
    
    print("=" * 60)
    print("Tabletop Domain WP-Z3 Interface Demo")
    print("=" * 60)
    
    interface = TabletopWP()
    
    # Example 1: Simple put action
    print("\n--- Example 1: WP([put(bread, plate)], loc(bread, plate)) ---")
    try:
        formula = interface.compute_wp(['put(bread, plate)'], 'loc(bread, plate)')
        print(f"WP Formula:\n{interface.print_formula(formula)}")
        
        z3_formula = interface.to_z3(formula)
        print(f"\nZ3 Formula: {z3_formula}")
        
        is_sat = interface.is_satisfiable(formula)
        print(f"\nSatisfiable: {is_sat}")
        
        if is_sat:
            model = interface.get_model(formula)
            print("Model:")
            for atom, value in model.items():
                print(f"  {atom} = {value}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Sequence with close and turn_on
    print("\n--- Example 2: WP([close(microwave), turn_on(microwave)], running(microwave)) ---")
    try:
        formula = interface.compute_wp(
            ['close(microwave)', 'turn_on(microwave)'], 
            'running(microwave)'
        )
        print(f"WP Formula (truncated): {str(formula)[:200]}...")
        
        is_sat = interface.is_satisfiable(formula)
        print(f"\nSatisfiable: {is_sat}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Multi-step sequence
    print("\n--- Example 3: WP([put(bread, plate), put(plate, microwave)], loc(plate, microwave)) ---")
    try:
        formula = interface.compute_wp(
            ['put(bread, plate)', 'put(plate, microwave)'], 
            'loc(plate, microwave)'
        )
        z3_formula = interface.to_z3(formula)
        print(f"Z3 Formula: {z3_formula}")
        
        is_sat = interface.is_satisfiable(formula)
        print(f"Satisfiable: {is_sat}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    demo()

