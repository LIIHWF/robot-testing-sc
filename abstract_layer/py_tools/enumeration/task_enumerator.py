#!/usr/bin/env python3
"""
Task Enumerator for Grammar-based Test Generation

Enumerates all possible tasks by expanding grammar rules up to specified limits.
Tracks rule application counts to respect max_step constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator, Set
from copy import deepcopy

import sys
import os
# Add py_lib to path for imports
lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'py_lib')
sys.path.insert(0, lib_path)
# Import using core as package name (since py_lib is added to path)
from core.grammar_parser import Grammar, Production, Symbol, SymbolType, GrammarParser


@dataclass
class RuleApplication:
    """Records a single rule application"""
    step: int  # Step number (1-based)
    rule_id: str  # Rule identifier (e.g., "A_1", "obj_2")
    nonterminal: str  # The nonterminal being expanded
    
    def __repr__(self):
        return f"Step{self.step}={self.rule_id}"


@dataclass
class DerivationState:
    """
    State during derivation process.
    Tracks the current sentential form and rule application history.
    """
    # Current sentential form (list of symbols)
    form: List[Symbol]
    
    # Rule application history
    history: List[RuleApplication] = field(default_factory=list)
    
    # Count of how many times each rule has been applied
    rule_counts: Dict[str, int] = field(default_factory=dict)
    
    # Current step number
    current_step: int = 0
    
    def is_terminal(self) -> bool:
        """Check if the current form contains only terminals"""
        return all(s.type == SymbolType.TERMINAL for s in self.form)
    
    def get_leftmost_nonterminal(self) -> Optional[Tuple[int, Symbol]]:
        """Get the index and symbol of the leftmost nonterminal"""
        for i, sym in enumerate(self.form):
            if sym.type == SymbolType.NONTERMINAL:
                return (i, sym)
        return None
    
    def get_rule_count(self, rule_id: str) -> int:
        """Get the count of times a rule has been applied"""
        return self.rule_counts.get(rule_id, 0)
    
    def can_apply_rule(self, prod: Production) -> bool:
        """Check if a production rule can be applied (respects max_step)"""
        if prod.max_step <= 0:  # No limit
            return True
        return self.get_rule_count(prod.rule_id) < prod.max_step
    
    def apply_rule(self, index: int, prod: Production) -> 'DerivationState':
        """
        Apply a production rule at the given index.
        Returns a new DerivationState.
        """
        # Create new form by replacing symbol at index with RHS
        new_form = self.form[:index] + prod.rhs + self.form[index + 1:]
        
        # Create new state
        new_state = DerivationState(
            form=new_form,
            history=self.history.copy(),
            rule_counts=self.rule_counts.copy(),
            current_step=self.current_step + 1
        )
        
        # Record rule application
        new_state.history.append(RuleApplication(
            step=new_state.current_step,
            rule_id=prod.rule_id,
            nonterminal=prod.lhs
        ))
        
        # Update rule count
        new_state.rule_counts[prod.rule_id] = new_state.get_rule_count(prod.rule_id) + 1
        
        return new_state
    
    def to_string(self) -> str:
        """Convert current form to string representation"""
        return " ".join(s.name for s in self.form)
    
    def __repr__(self):
        form_str = self.to_string()
        history_str = ", ".join(str(h) for h in self.history)
        return f"[{form_str}] <- [{history_str}]"


@dataclass
class EnumeratedTask:
    """A fully expanded task (no nonterminals)"""
    # The terminal string representation
    task_string: str
    
    # Rule application sequence
    rule_sequence: List[RuleApplication]
    
    # Rule counts used in this derivation
    rule_counts: Dict[str, int]
    
    def __repr__(self):
        return f"Task({self.task_string})"
    
    def get_step_rules(self) -> Dict[int, str]:
        """Get mapping from step number to rule_id"""
        return {app.step: app.rule_id for app in self.rule_sequence}


class TaskEnumerator:
    """
    Enumerates all possible tasks by expanding grammar rules.
    Uses leftmost derivation with rule application limits.
    """
    
    def __init__(self, grammar: Grammar, max_derivation_depth: int = 100):
        """
        Initialize enumerator.
        
        Args:
            grammar: The grammar to enumerate from
            max_derivation_depth: Maximum derivation steps (safety limit)
        """
        self.grammar = grammar
        self.max_derivation_depth = max_derivation_depth
    
    def enumerate(self) -> Generator[EnumeratedTask, None, None]:
        """
        Enumerate all possible tasks.
        
        Yields:
            EnumeratedTask objects for each valid derivation
        """
        # Start with the start symbol
        start_sym = Symbol(self.grammar.start_symbol, SymbolType.NONTERMINAL)
        initial_state = DerivationState(form=[start_sym])
        
        # Use a stack for depth-first enumeration
        stack = [initial_state]
        
        while stack:
            state = stack.pop()
            
            # Check if we've reached a terminal form
            if state.is_terminal():
                yield EnumeratedTask(
                    task_string=state.to_string(),
                    rule_sequence=state.history,
                    rule_counts=state.rule_counts
                )
                continue
            
            # Check depth limit
            if state.current_step >= self.max_derivation_depth:
                continue
            
            # Find leftmost nonterminal
            nt_info = state.get_leftmost_nonterminal()
            if nt_info is None:
                continue
            
            index, nonterminal = nt_info
            
            # Get all applicable productions
            productions = self.grammar.get_productions(nonterminal.name)
            
            # Try each production (in reverse order so stack pops in forward order)
            for prod in reversed(productions):
                if state.can_apply_rule(prod):
                    new_state = state.apply_rule(index, prod)
                    stack.append(new_state)
    
    def enumerate_all(self) -> List[EnumeratedTask]:
        """Enumerate all tasks and return as list"""
        return list(self.enumerate())
    
    def get_statistics(self, tasks: List[EnumeratedTask]) -> Dict:
        """Get statistics about enumerated tasks"""
        if not tasks:
            return {"count": 0}
        
        max_steps = max(len(t.rule_sequence) for t in tasks)
        min_steps = min(len(t.rule_sequence) for t in tasks)
        
        # Count by derivation length
        by_length = {}
        for t in tasks:
            length = len(t.rule_sequence)
            by_length[length] = by_length.get(length, 0) + 1
        
        # Count unique rule combinations
        rule_combos = set()
        for t in tasks:
            combo = tuple(sorted(t.rule_counts.items()))
            rule_combos.add(combo)
        
        return {
            "count": len(tasks),
            "max_steps": max_steps,
            "min_steps": min_steps,
            "by_length": by_length,
            "unique_rule_combos": len(rule_combos)
        }


def print_tasks(tasks: List[EnumeratedTask], max_print: int = 50):
    """Pretty print enumerated tasks"""
    print(f"\n{'=' * 70}")
    print(f"Enumerated Tasks (total: {len(tasks)})")
    print('=' * 70)
    
    for i, task in enumerate(tasks[:max_print]):
        steps = ", ".join(str(app) for app in task.rule_sequence)
        print(f"{i+1:3d}. {task.task_string}")
        print(f"     Rules: [{steps}]")
        print(f"     Counts: {dict(task.rule_counts)}")
        print()
    
    if len(tasks) > max_print:
        print(f"... and {len(tasks) - max_print} more tasks")


def export_rule_sequences(tasks: List[EnumeratedTask], filepath: str):
    """
    Export rule application sequences to a file.
    Format: Each line is a task with its rule sequence.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Task Enumeration Results\n")
        f.write(f"# Total tasks: {len(tasks)}\n")
        f.write("#\n")
        f.write("# Format: task_string | rule_sequence | rule_counts\n")
        f.write("#\n")
        
        for task in tasks:
            rules = ";".join(f"{app.step}:{app.rule_id}" for app in task.rule_sequence)
            counts = ",".join(f"{k}={v}" for k, v in sorted(task.rule_counts.items()))
            f.write(f"{task.task_string} | {rules} | {counts}\n")
    
    print(f"Exported {len(tasks)} tasks to {filepath}")


def build_step_parameter_model(tasks: List[EnumeratedTask], grammar: Grammar) -> dict:
    """
    Build a model showing which rules are available at each step position.
    
    Returns:
        Dictionary with:
        - 'max_steps': Maximum number of steps across all tasks
        - 'step_rules': Dict[step_num -> Set[rule_id]] - available rules at each step
        - 'constraints': List of (condition, forbidden_rule) pairs
    """
    if not tasks:
        return {'max_steps': 0, 'step_rules': {}, 'constraints': []}
    
    max_steps = max(len(t.rule_sequence) for t in tasks)
    
    # Collect all rules used at each step position
    step_rules: Dict[int, Set[str]] = {i: set() for i in range(1, max_steps + 1)}
    
    for task in tasks:
        for app in task.rule_sequence:
            step_rules[app.step].add(app.rule_id)
    
    # Add 'None' option for optional steps (steps that don't always have a rule)
    for step in range(2, max_steps + 1):
        # Check if some tasks end before this step
        tasks_with_step = sum(1 for t in tasks if len(t.rule_sequence) >= step)
        if tasks_with_step < len(tasks):
            step_rules[step].add('None')
    
    return {
        'max_steps': max_steps,
        'step_rules': {k: sorted(v) for k, v in step_rules.items()},
    }


def generate_acts_model(tasks: List[EnumeratedTask], grammar: Grammar, output_path: str):
    """
    Generate an ACTS model from enumerated tasks.
    
    The model captures which rule combinations are valid.
    """
    model = build_step_parameter_model(tasks, grammar)
    
    lines = []
    lines.append("[System]")
    lines.append("Name: GrammarModel")
    
    lines.append("[Parameter]")
    for step in range(1, model['max_steps'] + 1):
        rules = model['step_rules'].get(step, ['None'])
        rules_str = ",".join(rules)
        lines.append(f"Step{step} (enum) : {rules_str}")
    
    lines.append("[Relation]")
    lines.append("[Constraint]")
    
    # Generate constraints: if Step{i} = None, then Step{i+1} = None
    # Only add constraint if BOTH steps have None in their domain
    for step in range(1, model['max_steps']):
        step_rules = model['step_rules'].get(step, [])
        step_plus_1_rules = model['step_rules'].get(step + 1, [])
        if 'None' in step_rules and 'None' in step_plus_1_rules:
            lines.append(f'(Step{step} = "None") => Step{step + 1} = "None"')
    
    # Generate constraints based on actual valid sequences
    # Build a set of valid prefixes at each step
    valid_prefixes: Dict[int, Set[Tuple]] = {i: set() for i in range(1, model['max_steps'] + 1)}
    
    for task in tasks:
        prefix = []
        for app in task.rule_sequence:
            prefix.append(app.rule_id)
            valid_prefixes[app.step].add(tuple(prefix))
        
        # Mark remaining steps as None
        for step in range(len(task.rule_sequence) + 1, model['max_steps'] + 1):
            prefix.append('None')
            valid_prefixes[step].add(tuple(prefix))
    
    # For each step, find invalid transitions and add constraints
    for step in range(1, model['max_steps']):
        # Get all prefixes at this step
        prefixes_at_step = valid_prefixes[step]
        next_step_rules = model['step_rules'].get(step + 1, set())
        
        for prefix in prefixes_at_step:
            # Find which rules are NOT valid after this prefix
            valid_next = set()
            for vp in valid_prefixes[step + 1]:
                if vp[:step] == prefix:
                    valid_next.add(vp[step])
            
            invalid_next = set(next_step_rules) - valid_next
            
            if invalid_next and len(invalid_next) < len(next_step_rules):
                # Build condition
                cond_parts = [f'Step{i+1} = "{prefix[i]}"' for i in range(len(prefix))]
                cond = " && ".join(cond_parts)
                
                for invalid_rule in sorted(invalid_next):
                    lines.append(f'({cond}) => Step{step + 1} != "{invalid_rule}"')
    
    lines.append("[Misc]")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"Generated ACTS model to {output_path}")


def demo():
    """Demo the task enumerator"""
    # Parse grammar
    parser = GrammarParser()
    
    # Example grammar content
    grammar_content = """
# Simple example
A ::= action
A ::= action ; A
A ::= if cond then A else A

action ::= pick obj
action ::= put obj obj
action ::= turnon obj

{max_step: 1}
obj ::= bread

{max_step: 3}
obj ::= plate
obj ::= microwave
"""
    
    grammar = parser.parse(grammar_content)
    
    print("Parsed Grammar:")
    print("-" * 40)
    for prod in grammar.productions:
        print(f"  {prod}")
    print()
    
    # Enumerate tasks
    print("Enumerating tasks...")
    enumerator = TaskEnumerator(grammar, max_derivation_depth=20)
    tasks = enumerator.enumerate_all()
    
    # Print statistics
    stats = enumerator.get_statistics(tasks)
    print(f"\nStatistics:")
    print(f"  Total tasks: {stats['count']}")
    print(f"  Derivation steps: {stats['min_steps']} - {stats['max_steps']}")
    print(f"  By length: {stats['by_length']}")
    print(f"  Unique rule combos: {stats['unique_rule_combos']}")
    
    # Print tasks
    print_tasks(tasks, max_print=30)


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Enumerate tasks from grammar')
    parser.add_argument('grammar_file', nargs='?', default=None, help='Grammar file path')
    parser.add_argument('-d', '--depth', type=int, default=20, help='Max derivation depth')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file for rule sequences')
    parser.add_argument('-m', '--model', type=str, default=None, help='Output ACTS model file')
    parser.add_argument('-p', '--print-max', type=int, default=50, help='Max tasks to print')
    
    args = parser.parse_args()
    
    if args.grammar_file:
        # Parse from file
        grammar_parser = GrammarParser()
        grammar = grammar_parser.parse_file(args.grammar_file)
        
        print("Parsed Grammar:")
        print("-" * 40)
        for prod in grammar.productions:
            print(f"  {prod}")
        print()
        
        # Enumerate tasks
        print(f"Enumerating tasks (max depth: {args.depth})...")
        
        enumerator = TaskEnumerator(grammar, max_derivation_depth=args.depth)
        tasks = enumerator.enumerate_all()
        
        # Print statistics
        stats = enumerator.get_statistics(tasks)
        print(f"\nStatistics:")
        print(f"  Total tasks: {stats['count']}")
        if stats['count'] > 0:
            print(f"  Derivation steps: {stats['min_steps']} - {stats['max_steps']}")
            print(f"  By length: {stats['by_length']}")
        
        # Print tasks
        print_tasks(tasks, max_print=args.print_max)
        
        # Export if requested
        if args.output:
            export_rule_sequences(tasks, args.output)
        
        # Generate ACTS model if requested
        if args.model:
            generate_acts_model(tasks, grammar, args.model)
    else:
        demo()

