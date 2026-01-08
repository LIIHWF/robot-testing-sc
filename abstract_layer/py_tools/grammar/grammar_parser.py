#!/usr/bin/env python3
"""
Grammar Parser for Task Enumeration

Parses grammar files with the following format:
    # Comments start with #
    A ::= action
    A ::= action ; A
    A ::= if cond then A else A
    
    action ::= pick obj
    action ::= put obj obj
    
    {max_step: 1}
    obj ::= bread
    
    {max_step: 3}
    obj ::= plate
    obj ::= microwave

The {max_step: N} directive specifies that subsequent rules can be applied at most N times.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum


class SymbolType(Enum):
    """Type of grammar symbol"""
    TERMINAL = "terminal"
    NONTERMINAL = "nonterminal"


@dataclass
class Symbol:
    """A grammar symbol (terminal or nonterminal)"""
    name: str
    type: SymbolType
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return hash((self.name, self.type))
    
    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name and self.type == other.type
        return False


@dataclass
class Production:
    """A grammar production rule"""
    lhs: str  # Left-hand side (nonterminal name)
    rhs: List[Symbol]  # Right-hand side (sequence of symbols)
    max_step: int = -1  # Maximum times this rule can be applied (-1 = unlimited)
    rule_id: str = ""  # Unique identifier for this rule
    
    def __repr__(self):
        rhs_str = " ".join(str(s) for s in self.rhs)
        limit_str = f" [max:{self.max_step}]" if self.max_step > 0 else ""
        return f"{self.lhs} ::= {rhs_str}{limit_str}"


@dataclass
class Grammar:
    """A context-free grammar"""
    productions: List[Production] = field(default_factory=list)
    nonterminals: Set[str] = field(default_factory=set)
    terminals: Set[str] = field(default_factory=set)
    start_symbol: str = ""
    
    # Map from (nonterminal, rule_id) to Production
    rule_map: Dict[Tuple[str, str], Production] = field(default_factory=dict)
    
    # Map from nonterminal to list of productions
    productions_for: Dict[str, List[Production]] = field(default_factory=dict)
    
    def add_production(self, prod: Production):
        """Add a production to the grammar"""
        self.productions.append(prod)
        self.nonterminals.add(prod.lhs)
        
        # Update productions_for
        if prod.lhs not in self.productions_for:
            self.productions_for[prod.lhs] = []
        self.productions_for[prod.lhs].append(prod)
        
        # Update rule_map
        self.rule_map[(prod.lhs, prod.rule_id)] = prod
        
        # Update terminals and nonterminals
        for sym in prod.rhs:
            if sym.type == SymbolType.TERMINAL:
                self.terminals.add(sym.name)
    
    def get_productions(self, nonterminal: str) -> List[Production]:
        """Get all productions for a nonterminal"""
        return self.productions_for.get(nonterminal, [])
    
    def finalize(self):
        """
        Finalize grammar: determine which symbols are nonterminals vs terminals
        based on whether they appear on LHS of any production
        """
        # All LHS symbols are nonterminals
        lhs_symbols = {p.lhs for p in self.productions}
        
        # Update symbol types in all productions
        for prod in self.productions:
            for i, sym in enumerate(prod.rhs):
                if sym.name in lhs_symbols:
                    prod.rhs[i] = Symbol(sym.name, SymbolType.NONTERMINAL)
                    self.nonterminals.add(sym.name)
                else:
                    prod.rhs[i] = Symbol(sym.name, SymbolType.TERMINAL)
                    self.terminals.add(sym.name)
        
        # Set start symbol as the first nonterminal that appears
        if self.productions and not self.start_symbol:
            self.start_symbol = self.productions[0].lhs


class GrammarParser:
    """Parser for grammar files"""
    
    def __init__(self):
        self.current_max_step = -1  # -1 means unlimited
        self.rule_counter: Dict[str, int] = {}  # Counter for generating rule IDs
    
    def parse_file(self, filepath: str) -> Grammar:
        """Parse a grammar file and return a Grammar object"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse(content)
    
    def parse(self, content: str) -> Grammar:
        """Parse grammar content string"""
        grammar = Grammar()
        self.current_max_step = -1
        self.rule_counter = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for max_step directive
            if line.startswith('{') and line.endswith('}'):
                self._parse_directive(line)
                continue
            
            # Parse production rule
            if '::=' in line:
                prod = self._parse_production(line)
                if prod:
                    grammar.add_production(prod)
        
        grammar.finalize()
        return grammar
    
    def _parse_directive(self, line: str):
        """Parse a directive like {max_step: 3}"""
        # Remove braces
        content = line[1:-1].strip()
        
        # Parse key: value
        if ':' in content:
            key, value = content.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'max_step':
                self.current_max_step = int(value)
    
    def _parse_production(self, line: str) -> Optional[Production]:
        """Parse a production rule like 'A ::= action ; A'"""
        parts = line.split('::=', 1)
        if len(parts) != 2:
            return None
        
        lhs = parts[0].strip()
        rhs_str = parts[1].strip()
        
        # Generate rule ID
        if lhs not in self.rule_counter:
            self.rule_counter[lhs] = 0
        self.rule_counter[lhs] += 1
        rule_id = f"{lhs}_{self.rule_counter[lhs]}"
        
        # Parse RHS symbols
        rhs = []
        tokens = self._tokenize(rhs_str)
        for token in tokens:
            # Initially mark all as nonterminal; will be corrected in finalize()
            rhs.append(Symbol(token, SymbolType.NONTERMINAL))
        
        return Production(
            lhs=lhs,
            rhs=rhs,
            max_step=self.current_max_step,
            rule_id=rule_id
        )
    
    def _tokenize(self, s: str) -> List[str]:
        """Tokenize RHS of a production"""
        # Split by whitespace, keeping keywords and identifiers
        tokens = []
        current = []
        
        for char in s:
            if char.isspace():
                if current:
                    tokens.append(''.join(current))
                    current = []
            else:
                current.append(char)
        
        if current:
            tokens.append(''.join(current))
        
        return tokens


def print_grammar(grammar: Grammar):
    """Pretty print a grammar"""
    print("=" * 60)
    print("Grammar")
    print("=" * 60)
    print(f"Start symbol: {grammar.start_symbol}")
    print(f"Nonterminals: {grammar.nonterminals}")
    print(f"Terminals: {grammar.terminals}")
    print("\nProductions:")
    for prod in grammar.productions:
        print(f"  {prod}")
    print("=" * 60)


# Demo
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "my_ct/grammar.txt"
    
    parser = GrammarParser()
    grammar = parser.parse_file(filepath)
    print_grammar(grammar)

