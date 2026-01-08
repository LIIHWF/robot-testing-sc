#!/usr/bin/env python3
"""
Translate ACTS test configurations into task expressions using grammar.

This script reads ACTS configuration files and converts each configuration into
a structured format with:
- Task expression (derived from Step parameters using grammar)
- Initial conditions (from environment parameters, if present)
- Error reports (when translation fails)

Supports two types of configuration files:
1. Full configurations: Contains both Step parameters (grammar) and environment parameters
   (e.g., bread_Location, Drawer_Door_State, etc.)
2. Grammar-only configurations: Contains only Step parameters (no environment parameters)

Unlike translate_tests.py, this script:
- Uses meta_model/task_grammar.txt to translate rule sequences instead of task mapping
- Reports errors instead of trying to fix them (no find_best_match fallback)
- Adapts to the non-CSV format of ACTS configuration files
"""

import re
import json
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """A grammar rule"""
    id: str
    lhs: str
    rhs: List[str]


class GrammarParser:
    """Parser for grammar files"""
    
    def parse_file(self, filepath: Path) -> Dict[str, List[Rule]]:
        """Parse a grammar file and return grammar dictionary"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse(content)
    
    def parse(self, content: str) -> Dict[str, List[Rule]]:
        """Parse grammar content string"""
        raw_rules = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip directives like {max_step: 1}
            if line.startswith('{') and line.endswith('}'):
                continue
            
            # Parse production rule
            if '::=' in line:
                lhs, rhs = line.split('::=', 1)
                lhs = lhs.strip()
                rhs_tokens = rhs.strip().split()
                raw_rules.append((lhs, rhs_tokens))
        
        # Build grammar dictionary
        nonterminals = {lhs for lhs, _ in raw_rules}
        grammar: Dict[str, List[Rule]] = {nt: [] for nt in nonterminals}
        
        # Assign rule IDs: Program_1, Program_2, Action_1, etc.
        counter: Dict[str, int] = {}
        for lhs, rhs_tokens in raw_rules:
            counter.setdefault(lhs, 0)
            counter[lhs] += 1
            rule_id = f"{lhs}_{counter[lhs]}"
            grammar[lhs].append(Rule(id=rule_id, lhs=lhs, rhs=rhs_tokens))
        
        return grammar


class SentenceBuilder:
    """Build task expressions from rule sequences using grammar"""
    
    def __init__(self, grammar: Dict[str, List[Rule]], start_symbol: str = "Program"):
        """
        Initialize sentence builder
        
        Args:
            grammar: Grammar dictionary {nonterminal -> [rules]}
            start_symbol: Start symbol (default: "Program")
        """
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.nonterminals = set(grammar.keys())
        
        # Build rule ID to rule mapping
        self.rule_map: Dict[str, Rule] = {}
        for rules in grammar.values():
            for rule in rules:
                self.rule_map[rule.id] = rule
    
    def build_sentence(self, step_sequence: List[str]) -> Tuple[str, Optional[str]]:
        """
        Build task expression from rule sequence
        
        Args:
            step_sequence: List of rule IDs like ["Program_1", "Action_1", "Movable_1", "Container_1"]
        
        Returns:
            Tuple of (task_expression, error_message)
            If successful, error_message is None
            If failed, task_expression is empty string and error_message contains error details
        """
        # Filter out "None" and empty values
        rule_ids = [rid for rid in step_sequence if rid and rid != "None"]
        
        if not rule_ids:
            # Empty sequence - try to use base rule
            if self.start_symbol in self.grammar and self.grammar[self.start_symbol]:
                base_rule = self.grammar[self.start_symbol][0]
                return (" ".join(base_rule.rhs), None)
            return ("", f"Empty rule sequence and no base rule for {self.start_symbol}")
        
        # Start derivation from start symbol
        form = [self.start_symbol]
        
        for rule_id in rule_ids:
            # Check if rule exists
            if rule_id not in self.rule_map:
                return ("", f"Unknown rule ID: {rule_id}")
            
            rule = self.rule_map[rule_id]
            
            # Find first matching nonterminal (leftmost derivation)
            replaced = False
            for i, sym in enumerate(form):
                if sym == rule.lhs:
                    # Replace this nonterminal with rule RHS
                    form = form[:i] + rule.rhs + form[i+1:]
                    replaced = True
                    break
            
            if not replaced:
                rhs_str = " ".join(rule.rhs)
                form_str = " ".join(form)
                return ("", f"Cannot apply rule {rule_id} ({rule.lhs} -> {rhs_str}): "
                           f"no matching nonterminal '{rule.lhs}' in current form: {form_str}")
        
        # Return final sentence
        return (" ".join(form), None)


def parse_configuration(config_lines: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a single configuration from the input file."""
    config = {}
    
    for line in config_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Parse lines like "1 = Step1=Program_1"
        match = re.match(r'(\d+)\s*=\s*(.+)', line)
        if match:
            param_num = match.group(1)
            param_value = match.group(2)
            
            # Parse parameter name and value
            param_match = re.match(r'(\w+)=(.+)', param_value)
            if param_match:
                param_name = param_match.group(1)
                param_val = param_match.group(2)
                
                config[param_name] = param_val
    
    return config if config else None


def build_rule_sequence(config: Dict[str, Any]) -> List[str]:
    """Build rule sequence list from Step parameters."""
    step_params = []
    for i in range(1, 11):  # Step1 to Step10
        step_key = f'Step{i}'
        if step_key in config:
            step_params.append(config[step_key])
        else:
            break
    
    return step_params


def extract_task_expression(config: Dict[str, Any], sentence_builder: SentenceBuilder) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract task expression from Step parameters using grammar.
    
    Returns:
        Tuple of (task_expression, error_message)
        If successful, error_message is None
        If failed, task_expression is None and error_message contains error details
    """
    # Build rule sequence
    rule_seq = build_rule_sequence(config)
    
    if not rule_seq:
        return (None, "No Step parameters found")
    
    # Translate using grammar
    task_expr, error = sentence_builder.build_sentence(rule_seq)
    
    if error:
        return (None, error)
    
    return (task_expr, None)


def extract_initial_conditions(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract initial conditions from environment parameters.
    
    Returns empty dict if no environment parameters are present
    (grammar-only configurations).
    """
    initial_conditions = {}
    
    # Location fluents
    location_params = ['bread_Location', 'fruit_Location', 'vegetable_Location', 'plate_Location']
    for param in location_params:
        if param in config:
            obj_name = param.replace('_Location', '').lower()
            location = config[param]
            initial_conditions[f'loc({obj_name}, {location})'] = True
    
    # Door states
    door_params = ['Drawer_Door_State', 'Cabinet_Door_State', 'Microwave_Door_State']
    for param in door_params:
        if param in config:
            obj_name = param.replace('_Door_State', '').lower()
            state = config[param]
            initial_conditions[f'door_open({obj_name})'] = (state == 'open')
    
    # Running state
    if 'Microwave_Running_State' in config:
        state = config['Microwave_Running_State']
        initial_conditions['running(microwave)'] = (state == 'running')
    
    return initial_conditions


def parse_configurations_file(input_file: Path) -> List[Dict[str, Any]]:
    """Parse the entire configurations file and extract all configurations.
    
    Supports two formats:
    1. JSON array format: [{"Step1": "...", ...}, ...]
    2. Text format: Configuration #1: ... ---
    """
    configurations = []
    
    # Try to parse as JSON first
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If it's a JSON array, use it directly
        if isinstance(data, list):
            for idx, config in enumerate(data, start=1):
                if isinstance(config, dict):
                    config['configuration_number'] = idx
                    configurations.append(config)
            return configurations
        # If it's a single JSON object, wrap it in a list
        elif isinstance(data, dict):
            data['configuration_number'] = 1
            configurations.append(data)
            return configurations
    except (json.JSONDecodeError, ValueError):
        # Not JSON, fall back to text format parsing
        pass
    
    # Parse as text format (original logic)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_config_lines = []
    config_num = None
    
    for line in lines:
        line = line.strip()
        
        # Check for configuration start
        match = re.match(r'Configuration\s*#(\d+):', line)
        if match:
            # Save previous configuration if exists
            if current_config_lines:
                config = parse_configuration(current_config_lines)
                if config:
                    config['configuration_number'] = config_num
                    configurations.append(config)
            
            # Start new configuration
            config_num = int(match.group(1))
            current_config_lines = []
            continue
        
        # Check for configuration end (separator line)
        if line.startswith('---'):
            if current_config_lines:
                config = parse_configuration(current_config_lines)
                if config:
                    config['configuration_number'] = config_num
                    configurations.append(config)
                current_config_lines = []
            continue
        
        # Add line to current configuration
        if current_config_lines is not None:
            current_config_lines.append(line)
    
    # Handle last configuration
    if current_config_lines:
        config = parse_configuration(current_config_lines)
        if config:
            config['configuration_number'] = config_num
            configurations.append(config)
    
    return configurations


def translate_to_configurations(configs: List[Dict[str, Any]], sentence_builder: SentenceBuilder) -> List[Dict[str, Any]]:
    """Translate raw configurations to structured format with tasks and initial conditions."""
    translated = []
    
    for config in configs:
        config_num = config.get('configuration_number', 0)
        
        # Extract task expression (with error reporting)
        task_expression, error = extract_task_expression(config, sentence_builder)
        
        # Extract initial conditions
        initial_conditions = extract_initial_conditions(config)
        
        # Build rule sequence for reference
        rule_sequence = build_rule_sequence(config)
        
        # Create translated configuration
        translated_config = {
            'configuration_number': config_num,
            'task_expression': task_expression,
            'error': error,  # Report error if translation failed
            'rule_sequence': ';'.join([f"{i+1}:{val}" for i, val in enumerate(rule_sequence)]),  # Keep for reference/debugging
            'initial_conditions': initial_conditions,
            'raw_parameters': {k: v for k, v in config.items() if k != 'configuration_number'}
        }
        
        translated.append(translated_config)
    
    return translated


def output_json(configs: List[Dict[str, Any]], output_file: Optional[Path] = None):
    """Output configurations as JSON."""
    output = json.dumps(configs, indent=2, ensure_ascii=False)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✓ Output written to {output_file}")
    else:
        print(output)


def output_prolog(configs: List[Dict[str, Any]], output_file: Optional[Path] = None):
    """Output configurations as Prolog initial state format."""
    lines = []
    lines.append("% Generated test configurations with tasks and initial conditions")
    lines.append("% Errors are reported in comments")
    lines.append("% Grammar-only configurations have no initial conditions")
    lines.append("")
    
    for config in configs:
        config_num = config['configuration_number']
        task_expr = config.get('task_expression')
        error = config.get('error')
        init_conds = config['initial_conditions']
        
        lines.append(f"% Configuration #{config_num}")
        if error:
            lines.append(f"% ERROR: {error}")
            lines.append(f"% Rule sequence: {config.get('rule_sequence', 'N/A')}")
        elif task_expr:
            lines.append(f"% Task expression: {task_expr}")
        else:
            lines.append(f"% Task expression: (not found)")
            lines.append(f"% Rule sequence: {config.get('rule_sequence', 'N/A')}")
        lines.append("")
        
        # Output initial conditions (may be empty for grammar-only configurations)
        if init_conds:
            for fluent, value in init_conds.items():
                if isinstance(value, bool):
                    if value:
                        lines.append(f"initially({fluent}, true).")
                    else:
                        lines.append(f"initially({fluent}, false).")
                else:
                    lines.append(f"initially({fluent}, {value}).")
        else:
            lines.append("% No initial conditions (grammar-only configuration)")
        
        lines.append("")
        lines.append("% ---")
        lines.append("")
    
    output = '\n'.join(lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✓ Output written to {output_file}")
    else:
        print(output)


def print_error_summary(configs: List[Dict[str, Any]]):
    """Print summary of errors encountered."""
    total = len(configs)
    errors = [c for c in configs if c.get('error')]
    successful = total - len(errors)
    
    print(f"\n{'='*60}")
    print("Translation Summary")
    print(f"{'='*60}")
    print(f"Total configurations: {total}")
    print(f"Successful translations: {successful}")
    print(f"Failed translations: {len(errors)}")
    
    if errors:
        print(f"\nError details:")
        for config in errors[:10]:  # Show first 10 errors
            print(f"  Config #{config['configuration_number']}: {config['error']}")
            print(f"    Rule sequence: {config.get('rule_sequence', 'N/A')}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    print(f"{'='*60}\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Translate ACTS test configurations into task expressions using grammar'
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input configurations file (supports both full and grammar-only configurations)'
    )
    parser.add_argument(
        '-g', '--grammar',
        type=Path,
        default=Path('meta_model/task_grammar.txt'),
        help='Grammar file (default: meta_model/task_grammar.txt)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file path'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'prolog'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--start-symbol',
        default='Program',
        help='Start symbol for grammar (default: Program)'
    )
    
    args = parser.parse_args()
    
    # Load grammar
    print(f"Loading grammar from {args.grammar}...")
    grammar_parser = GrammarParser()
    grammar = grammar_parser.parse_file(args.grammar)
    
    # Count rules
    total_rules = sum(len(rules) for rules in grammar.values())
    print(f"  Loaded {len(grammar)} nonterminals with {total_rules} total rules")
    
    # Create sentence builder
    sentence_builder = SentenceBuilder(grammar, args.start_symbol)
    
    # Parse configurations file
    print(f"Reading {args.input_file}...")
    configs = parse_configurations_file(args.input_file)
    print(f"  Found {len(configs)} configurations")
    
    # Translate configurations
    print("Translating configurations...")
    translated = translate_to_configurations(configs, sentence_builder)
    print(f"  Translated {len(translated)} configurations")
    
    # Print error summary
    print_error_summary(translated)
    
    # Output results
    if args.format == 'json':
        output_json(translated, args.output)
    elif args.format == 'prolog':
        output_prolog(translated, args.output)


if __name__ == '__main__':
    main()

