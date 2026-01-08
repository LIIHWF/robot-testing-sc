#!/usr/bin/env python3
"""
Z3 Utility Functions for Prolog Formula Handling

This module provides core Z3 functionality separated from domain-specific logic:
1. Prolog formula parsing
2. Formula data structures
3. Z3 conversion utilities
4. CNF simplification

Usage:
    from z3_utils import Formula, FormulaType, Z3Converter, PrologParser
"""

import re
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

try:
    from z3 import (
        Bool, BoolRef, BoolVal, And, Or, Not, Implies, Solver, sat, unsat,
        is_bool, is_app, is_true,
        Z3_OP_AND, Z3_OP_OR, Z3_OP_IMPLIES, Z3_OP_EQ, 
        Z3_OP_TRUE, Z3_OP_FALSE, Z3_OP_XOR, Z3_OP_NOT
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# =============================================================================
# FORMULA DATA STRUCTURES
# =============================================================================

class FormulaType(Enum):
    """Types of formula nodes"""
    AND = "and"
    OR = "or"
    NEG = "neg"
    IMPL = "impl"
    ATOM = "atom"
    TRUE = "true"
    FALSE = "false"
    INEQ = "ineq"  # Inequality: O1 != O2


@dataclass
class Formula:
    """Represents a logical formula"""
    type: FormulaType
    content: Any  # Can be string (atom), tuple (ineq), or list (children)
    
    def __repr__(self):
        if self.type == FormulaType.ATOM:
            return f"Atom({self.content})"
        elif self.type == FormulaType.TRUE:
            return "True"
        elif self.type == FormulaType.FALSE:
            return "False"
        elif self.type == FormulaType.INEQ:
            return f"Ineq({self.content[0]} != {self.content[1]})"
        else:
            children = ", ".join(str(c) for c in self.content)
            return f"{self.type.value.upper()}({children})"
    
    def pretty_print(self, indent: int = 0) -> str:
        """Pretty print the formula with indentation"""
        prefix = "  " * indent
        if self.type == FormulaType.ATOM:
            return f"{prefix}{self.content}"
        elif self.type == FormulaType.TRUE:
            return f"{prefix}True"
        elif self.type == FormulaType.FALSE:
            return f"{prefix}False"
        elif self.type == FormulaType.INEQ:
            return f"{prefix}{self.content[0]} ≠ {self.content[1]}"
        elif self.type == FormulaType.NEG:
            return f"{prefix}NOT\n" + self.content[0].pretty_print(indent + 1)
        elif self.type in [FormulaType.AND, FormulaType.OR, FormulaType.IMPL]:
            op = self.type.value.upper()
            result = f"{prefix}{op}\n"
            for child in self.content:
                result += child.pretty_print(indent + 1) + "\n"
            return result.rstrip()
        else:
            return f"{prefix}{self}"


# =============================================================================
# PROLOG PARSER
# =============================================================================

class PrologParser:
    """Parser for Prolog term strings to Formula objects"""
    
    @staticmethod
    def parse(term_str: str) -> Formula:
        """
        Parse Prolog term string into Formula object
        
        Handles:
        - and(P, Q), or(P, Q), neg(P), impl(P, Q)
        - Fluents like loc(O1, O2), door_open(O), running(O)
        - true, false
        - O1 \\= O2 (inequality)
        """
        term_str = term_str.strip().rstrip('.').strip()
        
        # Handle true/false
        if term_str == "true":
            return Formula(FormulaType.TRUE, None)
        if term_str == "false":
            return Formula(FormulaType.FALSE, None)
        
        # Handle negation
        if term_str.startswith("neg("):
            inner = PrologParser._extract_inner(term_str, "neg")
            return Formula(FormulaType.NEG, [PrologParser.parse(inner)])
        
        # Handle and
        if term_str.startswith("and("):
            inner = PrologParser._extract_inner(term_str, "and")
            children = PrologParser._parse_comma_separated(inner)
            return Formula(FormulaType.AND, children)
        
        # Handle or
        if term_str.startswith("or("):
            inner = PrologParser._extract_inner(term_str, "or")
            children = PrologParser._parse_comma_separated(inner)
            return Formula(FormulaType.OR, children)
        
        # Handle impl
        if term_str.startswith("impl("):
            inner = PrologParser._extract_inner(term_str, "impl")
            children = PrologParser._parse_comma_separated(inner)
            if len(children) == 2:
                return Formula(FormulaType.IMPL, children)
        
        # Handle inequality: O1 \= O2
        if r"\=" in term_str or "\\=" in term_str:
            match = re.match(r'(.+?)\s*\\=\s*(.+)', term_str)
            if match:
                left = match.group(1).strip()
                right = match.group(2).strip()
                return Formula(FormulaType.INEQ, (left, right))
        
        # Handle atomic formulas (fluents): pred(arg1, arg2, ...)
        if "(" in term_str and term_str.endswith(")"):
            return Formula(FormulaType.ATOM, term_str)
        
        # Simple atom
        return Formula(FormulaType.ATOM, term_str)
    
    @staticmethod
    def _extract_inner(term: str, prefix: str) -> str:
        """Extract content inside prefix(...)"""
        start = len(prefix) + 1
        if term.endswith(")"):
            return term[start:-1]
        return term[start:]
    
    @staticmethod
    def _parse_comma_separated(s: str) -> List[Formula]:
        """Parse comma-separated terms, handling nested parentheses"""
        result = []
        depth = 0
        current = []
        
        for char in s:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                if current:
                    term_str = ''.join(current).strip()
                    if term_str:
                        result.append(PrologParser.parse(term_str))
                    current = []
            else:
                current.append(char)
        
        # Add remaining
        if current:
            term_str = ''.join(current).strip()
            if term_str:
                result.append(PrologParser.parse(term_str))
        
        return result if result else [Formula(FormulaType.TRUE, None)]


# =============================================================================
# Z3 CONVERTER
# =============================================================================

class Z3Converter:
    """Converts Formula objects to Z3 expressions"""
    
    def __init__(self):
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available. Install with: pip install z3-solver")
        self.atom_cache: Dict[str, BoolRef] = {}
    
    def to_z3(self, formula: Formula) -> Union[BoolRef, bool]:
        """Convert Formula to Z3 BoolRef"""
        if formula.type == FormulaType.TRUE:
            return BoolVal(True)
        elif formula.type == FormulaType.FALSE:
            return BoolVal(False)
        elif formula.type == FormulaType.ATOM:
            return self._atom_to_z3(formula.content)
        elif formula.type == FormulaType.NEG:
            return Not(self.to_z3(formula.content[0]))
        elif formula.type == FormulaType.AND:
            children_z3 = [self.to_z3(c) for c in formula.content]
            return And(*children_z3) if len(children_z3) > 1 else (children_z3[0] if children_z3 else BoolVal(True))
        elif formula.type == FormulaType.OR:
            children_z3 = [self.to_z3(c) for c in formula.content]
            return Or(*children_z3) if len(children_z3) > 1 else (children_z3[0] if children_z3 else BoolVal(False))
        elif formula.type == FormulaType.IMPL:
            if len(formula.content) != 2:
                raise ValueError(f"Implies requires 2 children, got {len(formula.content)}")
            return Implies(self.to_z3(formula.content[0]), self.to_z3(formula.content[1]))
        elif formula.type == FormulaType.INEQ:
            left, right = formula.content
            atom_name = f"ineq_{left}_{right}"
            return self._atom_to_z3(atom_name)
        else:
            raise ValueError(f"Unknown formula type: {formula.type}")
    
    def _atom_to_z3(self, atom_str: str) -> BoolRef:
        """Convert atom string to Z3 Bool variable (with caching)"""
        # Normalize: loc(o1, o2) -> loc_o1_o2
        normalized = atom_str.replace("(", "_").replace(")", "").replace(",", "_").replace(" ", "")
        
        if normalized not in self.atom_cache:
            self.atom_cache[normalized] = Bool(normalized)
        
        return self.atom_cache[normalized]
    
    def get_atoms(self) -> Dict[str, BoolRef]:
        """Get all cached atoms"""
        return self.atom_cache.copy()


# =============================================================================
# CNF UTILITIES
# =============================================================================

class CNFConverter:
    """Converts Z3 formulas to CNF using model enumeration"""
    
    def __init__(self):
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available. Install with: pip install z3-solver")
    
    def is_atom(self, t: BoolRef) -> bool:
        """Check if a Z3 term is an atomic formula"""
        if not is_bool(t):
            return False
        if not is_app(t):
            return False
        
        k = t.decl().kind()
        non_atomic = {Z3_OP_AND, Z3_OP_OR, Z3_OP_IMPLIES, Z3_OP_TRUE, Z3_OP_FALSE, Z3_OP_XOR, Z3_OP_NOT}
        if k in non_atomic:
            return False
        if k == Z3_OP_EQ and t.arg(0).is_bool():
            return False
        return True
    
    def extract_atoms(self, fml: BoolRef) -> Set[BoolRef]:
        """Extract all atomic formulas from a Z3 formula"""
        visited = set()
        atoms = set()
        
        def recurse(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            
            if self.is_atom(t):
                atoms.add(t)
            
            for child in t.children():
                recurse(child)
        
        recurse(fml)
        return atoms
    
    def to_cnf(self, fml: BoolRef):
        """
        Extract CNF representation using model enumeration
        
        Yields CNF clauses (disjunctions)
        """
        atoms = self.extract_atoms(fml)
        snot = Solver()
        snot.add(Not(fml))
        
        while snot.check() == sat:
            m = snot.model()
            # Build blocking clause: at least one literal must change
            clause_lits = []
            for a in atoms:
                eval_result = m.eval(a)
                if is_true(eval_result):
                    clause_lits.append(Not(a))
                else:
                    clause_lits.append(a)
            
            if not clause_lits:
                yield BoolVal(False)
                break
            
            clause = Or(clause_lits) if len(clause_lits) > 1 else clause_lits[0]
            yield clause
            snot.add(clause)
    
    def simplify_to_cnf(self, fml: BoolRef) -> List[BoolRef]:
        """Convert formula to CNF and return as list of clauses"""
        return list(self.to_cnf(fml))


# =============================================================================
# SATISFIABILITY CHECKER
# =============================================================================

class SATChecker:
    """Z3-based satisfiability checking"""
    
    def __init__(self):
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available. Install with: pip install z3-solver")
    
    def check_sat(self, fml: BoolRef) -> bool:
        """Check if formula is satisfiable"""
        s = Solver()
        s.add(fml)
        return s.check() == sat
    
    def get_model(self, fml: BoolRef, atoms: Dict[str, BoolRef]) -> Optional[Dict[str, bool]]:
        """Get satisfying model if formula is SAT"""
        s = Solver()
        s.add(fml)
        
        if s.check() == sat:
            m = s.model()
            result = {}
            for name, var in atoms.items():
                val = m.eval(var)
                if val is not None:
                    result[name] = is_true(val)
            return result
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_prolog(term_str: str) -> Formula:
    """Convenience function: parse Prolog term to Formula"""
    return PrologParser.parse(term_str)


def formula_to_z3(formula: Formula, converter: Optional[Z3Converter] = None) -> BoolRef:
    """Convenience function: convert Formula to Z3"""
    if converter is None:
        converter = Z3Converter()
    return converter.to_z3(formula)


def check_z3_available() -> bool:
    """Check if Z3 is available"""
    return Z3_AVAILABLE




