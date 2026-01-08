#!/usr/bin/env python3
"""
Combined Model Generator with Z3 Simplification

Merges task generation CT model with environment CT model,
incorporating WP analysis results as constraints.
"""

import re
import sys
import argparse
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

# Try to import Z3 for constraint simplification
try:
    from z3 import Bool, BoolRef, Solver, simplify, And, Or, Not, Implies, unsat, sat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 not available. Install with: pip install z3-solver", file=sys.stderr)


class Parameter:
    """Represents a parameter in the ACTS model"""
    def __init__(self, name: str, param_type: str, values: List[str]):
        self.name = name
        self.type = param_type  # 'enum', 'int', 'bool'
        self.values = values
    
    def to_acts_format(self) -> str:
        """Convert to ACTS format"""
        if self.type == 'enum':
            # Use comma without space to match ACTS format
            values_str = ','.join(self.values)
            return f"{self.name} (enum): {values_str}"
        else:
            return f"{self.name} ({self.type}): {','.join(map(str, self.values))}"


class Constraint:
    """Represents a constraint in the ACTS model"""
    def __init__(self, expression: str, comment: str = ""):
        self.expression = expression
        self.comment = comment
    
    def to_acts_format(self) -> str:
        """Convert to ACTS format"""
        # ACTS may have issues with comments in constraints, so we omit them
        return self.expression


class SystemModel:
    """Represents a complete ACTS system model"""
    def __init__(self, name: str = "CombinedModel", strength: int = 2):
        self.name = name
        self.strength = strength
        self.parameters: Dict[str, Parameter] = {}
        self.constraints: List[Constraint] = []
    
    def add_parameter(self, param: Parameter):
        """Add a parameter to the model"""
        self.parameters[param.name] = param
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the model"""
        self.constraints.append(constraint)
    
    def to_acts_file(self, output_file: str):
        """Write model to ACTS format file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # System section
            f.write("[System]\n")
            f.write(f"Name: {self.name}\n")
            f.write(f"Strength: {self.strength}\n")
            f.write("\n")
            
            # Parameter section
            f.write("[Parameter]\n")
            for param in self.parameters.values():
                f.write(param.to_acts_format() + "\n")
            f.write("\n")
            
            # Constraint section
            f.write("[Constraint]\n")
            for constraint in self.constraints:
                f.write(constraint.to_acts_format() + "\n")
            f.write("\n")
            
            # Relation section (empty for now)
            f.write("[Relation]\n")
            f.write("\n")
            
            # Misc section (empty)
            f.write("[Misc]\n")


class WPFormulaParser:
    """Parse WP formulas and convert to ACTS constraints"""
    
    # Environment parameter mappings
    ENV_MAPPINGS = {
        'loc': 'Location',  # loc(bread, plate) -> bread_Location = "plate"
        'door_open': 'Door_State',  # door_open(microwave) -> Microwave_Door_State = "open"
        'running': 'Running_State',  # running(microwave) -> Microwave_Running_State = "running"
    }
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize names to PascalCase"""
        # Handle special cases
        if name == 'microwave':
            return 'Microwave'
        if name == 'drawer':
            return 'Drawer'
        if name == 'cabinet':
            return 'Cabinet'
        
        # Capitalize first letter
        return name.capitalize()
    
    def parse_wp_formula(self, formula: str) -> Tuple[str, Set[Tuple[str, str]]]:
        """
        Parse WP formula and convert to ACTS constraint
        
        Returns:
            (constraint_string, set of (param_name, value) tuples for validation)
        """
        # Track parameters referenced for validation (no auto-repair)
        referenced_params = set()
        
        # Recursively parse the formula
        result = self._parse_expression(formula, referenced_params)
        
        return result, referenced_params
    
    def _parse_expression(self, expr: str, refs: Set) -> str:
        """Recursively parse WP expression"""
        expr = expr.strip()
        
        # Handle logical operators
        if expr.startswith('AND('):
            return self._parse_and(expr, refs)
        elif expr.startswith('OR('):
            return self._parse_or(expr, refs)
        elif expr.startswith('IMPL('):
            return self._parse_implication(expr, refs)
        elif expr.startswith('NEG('):
            return self._parse_negation(expr, refs)
        elif expr.startswith('Atom('):
            return self._parse_atom(expr, refs)
        else:
            # Unknown format, return as-is
            return expr
    
    def _parse_and(self, expr: str, refs: Set) -> str:
        """Parse AND expression"""
        # Extract content between AND( and matching )
        content = self._extract_content(expr, 'AND(')
        sub_exprs = self._split_top_level(content)
        
        parsed = [self._parse_expression(e, refs) for e in sub_exprs]
        return f"({' && '.join(parsed)})"
    
    def _parse_or(self, expr: str, refs: Set) -> str:
        """Parse OR expression"""
        content = self._extract_content(expr, 'OR(')
        sub_exprs = self._split_top_level(content)
        
        parsed = [self._parse_expression(e, refs) for e in sub_exprs]
        return f"({' || '.join(parsed)})"
    
    def _parse_implication(self, expr: str, refs: Set) -> str:
        """Parse IMPL(A, B) as ((!A) || B) - convert implication to disjunction"""
        content = self._extract_content(expr, 'IMPL(')
        sub_exprs = self._split_top_level(content)
        
        if len(sub_exprs) != 2:
            return f"({content})"
        
        antecedent = self._parse_expression(sub_exprs[0], refs)
        consequent = self._parse_expression(sub_exprs[1], refs)
        
        # Convert A => B to ((!A) || B)
        # Ensure antecedent is properly parenthesized for negation
        if not (antecedent.startswith('(') and antecedent.endswith(')')):
            # Antecedent needs parentheses for negation
            negated_antecedent = f"!({antecedent})"
        else:
            # Antecedent already has parentheses
            negated_antecedent = f"!{antecedent}"
        
        # Ensure consequent is properly parenthesized if it contains operators
        if '&&' in consequent or '||' in consequent or '=>' in consequent:
            if not (consequent.startswith('(') and consequent.endswith(')')):
                consequent = f"({consequent})"
        
        return f"(({negated_antecedent}) || {consequent})"
    
    def _parse_negation(self, expr: str, refs: Set) -> str:
        """Parse NEG expression"""
        content = self._extract_content(expr, 'NEG(')
        parsed = self._parse_expression(content, refs)
        
        # For boolean parameters, try to use explicit opposite value
        # For enum parameters with known domain, use explicit opposite
        # Otherwise, use !() syntax
        if '=' in parsed and parsed.count('=') == 1:
            parts = parsed.split('=', 1)
            if len(parts) == 2:
                param = parts[0].strip()
                value = parts[1].strip().strip('"')
                
                # Try to find opposite value for known domains
                # For door states: "open" -> "closed"
                if value == "open" and "Door_State" in param:
                    return f'{param}="closed"'
                elif value == "closed" and "Door_State" in param:
                    return f'{param}="open"'
                # For running states: "running" -> "stopped"
                elif value == "running" and "Running_State" in param:
                    return f'{param}="stopped"'
                elif value == "stopped" and "Running_State" in param:
                    return f'{param}="running"'
        
        return f"!({parsed})"
    
    def _parse_atom(self, expr: str, refs: Set) -> str:
        """Parse atomic formula"""
        content = self._extract_content(expr, 'Atom(')
        
        # Parse different atom types
        # loc(bread, plate)
        loc_match = re.match(r'loc\((\w+),\s*(\w+)\)', content)
        if loc_match:
            obj, location = loc_match.groups()
            param_name = f"{obj}_{self.ENV_MAPPINGS['loc']}"
            refs.add((param_name, location))
            return f'{param_name}="{location}"'
        
        # door_open(container)
        door_match = re.match(r'door_open\((\w+)\)', content)
        if door_match:
            container = door_match.group(1)
            param_name = f"{self.normalize_name(container)}_{self.ENV_MAPPINGS['door_open']}"
            refs.add((param_name, "open"))
            return f'{param_name}="open"'
        
        # running(device)
        running_match = re.match(r'running\((\w+)\)', content)
        if running_match:
            device = running_match.group(1)
            param_name = f"{self.normalize_name(device)}_{self.ENV_MAPPINGS['running']}"
            refs.add((param_name, "running"))
            return f'{param_name}="running"'
        
        # Generic atom - convert to boolean
        refs.add((content, "true"))
        return f'{content}="true"'
    
    def _extract_content(self, expr: str, prefix: str) -> str:
        """Extract content between prefix and matching closing paren"""
        if not expr.startswith(prefix):
            return expr
        
        start = len(prefix)
        depth = 1
        i = start
        
        while i < len(expr) and depth > 0:
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
            i += 1
        
        return expr[start:i-1]
    
    def _split_top_level(self, content: str) -> List[str]:
        """Split by comma at top level (not inside parentheses)"""
        parts = []
        current = []
        depth = 0
        
        for char in content:
            if char == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                current.append(char)
        
        if current:
            parts.append(''.join(current).strip())
        
        return parts


class CombinedModelGenerator:
    """Generate combined CT model from task, environment, and WP analysis"""
    
    # Reverse mapping from grammar rule IDs to objects
    # Based on meta_model/task_grammar.txt:
    # Movable ::= bread | fruit | vegetable | plate
    # Container ::= plate | microwave | cabinet | drawer | basket
    # HasDoor ::= microwave | cabinet | drawer
    # HasRunningState ::= microwave
    RULE_TO_OBJECT = {
        'Movable_1': 'bread',
        'Movable_2': 'fruit',
        'Movable_3': 'vegetable',
        'Movable_4': 'plate',
        'Container_1': 'plate',
        'Container_2': 'microwave',
        'Container_3': 'cabinet',
        'Container_4': 'drawer',
        'Container_5': 'basket',
        'HasDoor_1': 'microwave',
        'HasDoor_2': 'cabinet',
        'HasDoor_3': 'drawer',
        'HasRunningState_1': 'microwave',
    }
    
    # Mapping from objects to their environment parameters
    # Objects with Location parameter (movable objects)
    OBJECT_TO_LOCATION_PARAM = {
        'bread': 'bread_Location',
        'fruit': 'fruit_Location',
        'vegetable': 'vegetable_Location',
        'plate': 'plate_Location',
        'basket': 'basket_Location',
    }
    
    # Objects with Door_State parameter
    OBJECT_TO_DOOR_PARAM = {
        'microwave': 'Microwave_Door_State',
        'cabinet': 'Cabinet_Door_State',
        'drawer': 'Drawer_Door_State',
    }
    
    # Objects with Running_State parameter
    OBJECT_TO_RUNNING_PARAM = {
        'microwave': 'Microwave_Running_State',
    }
    
    def __init__(self, name: str = "CombinedModel", strength: int = 2):
        self.model = SystemModel(name, strength)
        self.wp_parser = WPFormulaParser()
    
    def load_task_model(self, task_model_file: str):
        """Load task model parameters and constraints"""
        print(f"Loading task model: {task_model_file}")
        
        with open(task_model_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        in_parameter = False
        in_constraint = False
        param_count = 0
        constraint_count = 0
        
        for line in lines:
            line = line.strip()
            
            if line == "[Parameter]":
                in_parameter = True
                in_constraint = False
                continue
            elif line == "[Constraint]":
                in_parameter = False
                in_constraint = True
                continue
            elif line.startswith("["):
                in_parameter = False
                in_constraint = False
                continue
            
            if in_parameter and line:
                # Parse parameter line: Step1 (enum) : A_1,A_2
                match = re.match(r'(\w+)\s*\((\w+)\)\s*:\s*(.+)', line)
                if match:
                    name = match.group(1)
                    param_type = match.group(2)
                    values = [v.strip() for v in match.group(3).split(',')]
                    
                    param = Parameter(name, param_type, values)
                    self.model.add_parameter(param)
                    param_count += 1
            
            elif in_constraint and line:
                constraint = Constraint(line)
                self.model.add_constraint(constraint)
                constraint_count += 1
        
        print(f"  Loaded {param_count} task parameters")
        print(f"  Loaded {constraint_count} task constraints")
    
    def load_environment_model(self, env_model_file: str):
        """Load environment model parameters and constraints"""
        print(f"Loading environment model: {env_model_file}")
        
        with open(env_model_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        in_parameter = False
        in_constraint = False
        param_count = 0
        constraint_count = 0
        
        for line in lines:
            line = line.strip()
            
            if line == "[Parameter]":
                in_parameter = True
                in_constraint = False
                continue
            elif line == "[Constraint]":
                in_parameter = False
                in_constraint = True
                continue
            elif line.startswith("["):
                in_parameter = False
                in_constraint = False
                continue
            
            if in_parameter and line:
                match = re.match(r'(\w+)\s*\((\w+)\)\s*:\s*(.+)', line)
                if match:
                    name = match.group(1)
                    param_type = match.group(2)
                    values = [v.strip() for v in match.group(3).split(',')]
                    
                    param = Parameter(name, param_type, values)
                    self.model.add_parameter(param)
                    param_count += 1
            
            elif in_constraint and line:
                constraint = Constraint(line)
                self.model.add_constraint(constraint)
                constraint_count += 1
        
        print(f"  Loaded {param_count} environment parameters")
        print(f"  Loaded {constraint_count} environment constraints")
    
    def load_wp_analysis(self, wp_file: str):
        """Load WP analysis and generate mapping constraints"""
        print(f"Loading WP analysis: {wp_file}")
        
        with open(wp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        valid_count = 0
        invalid_count = 0
        constraint_count = 0
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Parse: STATUS | task_string | rule_sequence | wp_formula
            parts = line.split('|')
            if len(parts) != 4:
                continue
            
            status = parts[0].strip()
            task_string = parts[1].strip()
            rule_seq = parts[2].strip()
            wp_formula = parts[3].strip()
            
            # Explicitly forbid FALSE and UNSAT tasks
            if status == "FALSE" or status == "UNSAT":
                invalid_count += 1
                # Parse rule sequence to task parameter constraint
                task_constraint = self._rule_seq_to_constraint(rule_seq)
                
                if task_constraint:
                    # Add constraint: !(task_constraint) (forbids this combination)
                    # Format: !(Step1="X" && Step2="Y" && ...)
                    # This ensures ACTS cannot generate this invalid combination
                    # Remove outer parentheses if present for cleaner negation
                    clean_constraint = task_constraint.strip()
                    if clean_constraint.startswith('(') and clean_constraint.endswith(')'):
                        clean_constraint = clean_constraint[1:-1].strip()
                    forbidden_constraint = f"!({clean_constraint})"
                    self.model.add_constraint(Constraint(forbidden_constraint, f"FORBIDDEN: {task_string}"))
                    constraint_count += 1
                continue
            
            if status == "VALID":
                valid_count += 1
                
                # Parse rule sequence to task parameter constraint
                task_constraint = self._rule_seq_to_constraint(rule_seq)
                
                if task_constraint:
                    # Parse WP formula
                    try:
                        wp_constraint, refs = self.wp_parser.parse_wp_formula(wp_formula)
                        
                        # Track referenced parameters for validation only (no auto-repair)
                        for param_name, value in refs:
                            if param_name not in self.model.parameters:
                                print(f"  Error: Parameter '{param_name}' referenced but not found in model")
                            else:
                                # Parameter exists, check if value is in its domain
                                existing_param = self.model.parameters[param_name]
                                if value not in existing_param.values:
                                    # Value not in domain - report error but don't auto-repair
                                    print(f"  Error: Value '{value}' not in {param_name} domain (values: {existing_param.values})")
                        
                        # Create implication: task => environment condition
                        # Split complex constraints with && in consequent into multiple constraints
                        # ACTS may have issues with complex consequents
                        if ' && ' in wp_constraint and wp_constraint.count(' && ') > 0:
                            # Split the consequent by && and create multiple constraints
                            # Remove outer parentheses if present
                            consequent = wp_constraint.strip()
                            if consequent.startswith('(') and consequent.endswith(')'):
                                consequent = consequent[1:-1]
                            
                            # Split by && at top level (not inside parentheses)
                            # Use a more robust approach: iterate through the string and look for " && " pattern
                            parts = []
                            current = []
                            depth = 0
                            i = 0
                            while i < len(consequent):
                                # Check for " && " pattern at top level
                                if depth == 0 and i + 3 < len(consequent) and consequent[i:i+4] == ' && ':
                                    # Found && separator at top level
                                    if len(current) > 0:
                                        parts.append(''.join(current).strip())
                                        current = []
                                    i += 4  # Skip " && "
                                    continue
                                
                                char = consequent[i]
                                if char == '(':
                                    depth += 1
                                    current.append(char)
                                elif char == ')':
                                    depth -= 1
                                    current.append(char)
                                else:
                                    current.append(char)
                                i += 1
                            
                            if current:
                                parts.append(''.join(current).strip())
                            
                            # Create separate constraints for each part
                            # Convert A => B to ((!A) || B) format
                            for part in parts:
                                if part:
                                    # Negate task_constraint: ensure it's properly parenthesized
                                    if not (task_constraint.startswith('(') and task_constraint.endswith(')')):
                                        negated_task = f"!({task_constraint})"
                                    else:
                                        negated_task = f"!{task_constraint}"
                                    
                                    # Ensure part is properly parenthesized if it contains operators
                                    if '&&' in part or '||' in part or '=>' in part:
                                        if not (part.startswith('(') and part.endswith(')')):
                                            part = f"({part})"
                                    
                                    simple_constraint = f"(({negated_task}) || {part})"
                                    self.model.add_constraint(Constraint(simple_constraint, f"{task_string}"))
                                    constraint_count += 1
                        else:
                            # Convert A => B to ((!A) || B) format
                            # Negate task_constraint: ensure it's properly parenthesized
                            if not (task_constraint.startswith('(') and task_constraint.endswith(')')):
                                negated_task = f"!({task_constraint})"
                            else:
                                negated_task = f"!{task_constraint}"
                            
                            # Ensure wp_constraint is properly parenthesized if it contains operators
                            if '&&' in wp_constraint or '||' in wp_constraint or '=>' in wp_constraint:
                                if not (wp_constraint.startswith('(') and wp_constraint.endswith(')')):
                                    wp_constraint = f"({wp_constraint})"
                            
                            full_constraint = f"(({negated_task}) || {wp_constraint})"
                            self.model.add_constraint(Constraint(full_constraint, f"{task_string}"))
                            constraint_count += 1
                    
                    except Exception as e:
                        print(f"  Warning: Failed to parse WP formula: {e}")
                        continue
        
        print(f"  Processed {valid_count} valid tasks")
        print(f"  Forbidden {invalid_count} invalid tasks (FALSE/UNSAT)")
        print(f"  Generated {constraint_count} WP mapping constraints")
        
        # Add constraints to restrict Step parameter combinations to only valid ones
        self._add_valid_step_combinations_constraints(wp_file)
        
        # Add na constraints based on Step parameter values
        self._add_na_constraints_from_step_values()
    
    def _rule_seq_to_constraint(self, rule_seq: str) -> Optional[str]:
        """Convert rule sequence to task parameter constraint"""
        # Parse: 1:Program_1;2:Action_1;3:Movable_1
        parts = rule_seq.split(';')
        constraints = []
        
        for part in parts:
            match = re.match(r'(\d+):(\w+)', part.strip())
            if match:
                step_num = match.group(1)
                rule_id = match.group(2)
                param_name = f"Step{step_num}"
                
                # Check if this parameter exists
                if param_name in self.model.parameters:
                    constraints.append(f'{param_name}="{rule_id}"')
        
        if constraints:
            return '(' + ' && '.join(constraints) + ')'
        return None
    
    def _add_valid_step_combinations_constraints(self, wp_file: str):
        """Add constraints to restrict Step parameter combinations to only valid ones from WP analysis"""
        # Collect all valid Step1-4 combinations from WP analysis
        valid_first4 = set()
        
        with open(wp_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) != 4:
                    continue
                
                status = parts[0].strip()
                rule_seq = parts[2].strip()
                
                if status == "VALID":
                    # Extract first 4 Step parameters
                    step_parts = rule_seq.split(';')[:4]
                    if len(step_parts) == 4:
                        # Build constraint string for first 4 steps
                        step_constraints = []
                        for part in step_parts:
                            match = re.match(r'(\d+):(\w+)', part.strip())
                            if match:
                                step_num = match.group(1)
                                rule_id = match.group(2)
                                param_name = f"Step{step_num}"
                                if param_name in self.model.parameters:
                                    step_constraints.append(f'{param_name}="{rule_id}"')
                        
                        if len(step_constraints) == 4:
                            first4_key = '(' + ' && '.join(step_constraints) + ')'
                            valid_first4.add(first4_key)
        
        if not valid_first4:
            print("  Warning: No valid Step1-4 combinations found")
            return
        
        print(f"  Found {len(valid_first4)} unique valid Step1-4 combinations")
        
        # Create a large OR constraint: at least one valid combination must be true
        # Format: (combo1) || (combo2) || ... || (comboN)
        # But ACTS may have issues with very large OR constraints
        # So we'll split into chunks if needed
        chunk_size = 100  # Try chunks of 100 OR terms
        valid_list = sorted(list(valid_first4))
        
        constraint_count = 0
        for i in range(0, len(valid_list), chunk_size):
            chunk = valid_list[i:i+chunk_size]
            
            if len(chunk) == 1:
                # Single combination - can't create OR constraint
                # Instead, we'll note it but not add a constraint
                # (The existing WP constraints already handle this)
                continue
            
            # Create OR constraint for this chunk
            # Note: ACTS doesn't directly support "must be one of", but we can try
            # Actually, a better approach is to ensure that if Step1-4 don't match any valid pattern,
            # then some impossible condition must hold (which will make the constraint unsatisfiable)
            
            # For now, we'll create a constraint that lists all valid combinations
            # Format: (combo1) || (combo2) || ... || (comboN)
            or_constraint = ' || '.join(chunk)
            
            # But this alone doesn't restrict invalid combinations
            # We need to add: if NOT (any valid combo), then false
            # Which is: !(combo1 || combo2 || ...) => false
            # Which simplifies to: (combo1 || combo2 || ...) || false
            # Which is just: combo1 || combo2 || ...
            
            # Actually, the real issue is that ACTS will generate all combinations
            # and we can't easily restrict it to only valid ones with constraints alone
            
            # However, we can add a note that these are the valid combinations
            # The actual filtering will need to happen post-generation or through
            # more sophisticated constraint modeling
            
            # For now, let's just document this
            if i == 0:
                print(f"  Note: Valid Step1-4 combinations identified but not fully constrained")
                print(f"  ACTS may still generate invalid combinations")
                print(f"  Post-filtering may be needed to remove invalid test cases")
        
        # Store valid combinations for potential post-processing
        self.valid_step_combinations = valid_first4
    
    def _add_na_constraints_from_step_values(self):
        """
        Add na constraints based on Step parameter values using grammar rule mappings.
        
        Logic:
        - For each environment parameter (e.g., bread_Location, Microwave_Door_State):
          - Find all rule IDs that involve this object (e.g., Movable_1 for bread)
          - If ANY Step parameter has one of these rule IDs, the parameter cannot be "na"
          - If NO Step parameter has any of these rule IDs, the parameter must be "na"
        
        This is implemented as:
        - For each environment parameter P with associated rule IDs {R1, R2, ...}:
          - Constraint: (StepX="R1" || StepY="R2" || ...) <=> P != "na"
        
        Which is equivalent to two constraints:
        - (StepX="R1" || StepY="R2" || ...) => P != "na"  (if any rule used, not na)
        - !(StepX="R1" || StepY="R2" || ...) => P = "na"  (if no rule used, must be na)
        """
        print("\n  Adding na constraints based on Step values...")
        
        # Build reverse mapping: object -> list of rule IDs
        object_to_rules = {}
        for rule_id, obj in self.RULE_TO_OBJECT.items():
            if obj not in object_to_rules:
                object_to_rules[obj] = []
            object_to_rules[obj].append(rule_id)
        
        # Find which Step parameters can have which rule IDs
        step_params = {name: param.values for name, param in self.model.parameters.items() 
                       if name.startswith('Step')}
        
        constraint_count = 0
        
        # Process each type of environment parameter
        param_mappings = [
            (self.OBJECT_TO_LOCATION_PARAM, 'Location'),
            (self.OBJECT_TO_DOOR_PARAM, 'Door_State'),
            (self.OBJECT_TO_RUNNING_PARAM, 'Running_State'),
        ]
        
        for obj_to_param, param_type in param_mappings:
            for obj, param_name in obj_to_param.items():
                if param_name not in self.model.parameters:
                    continue
                
                # Check if "na" is a valid value for this parameter
                param = self.model.parameters[param_name]
                if 'na' not in param.values:
                    continue
                
                # Get all rule IDs that involve this object
                rule_ids = object_to_rules.get(obj, [])
                if not rule_ids:
                    continue
                
                # Build list of (StepN, rule_id) pairs where this rule can appear
                step_rule_pairs = []
                for step_name, step_values in step_params.items():
                    for rule_id in rule_ids:
                        if rule_id in step_values:
                            step_rule_pairs.append((step_name, rule_id))
                
                if not step_rule_pairs:
                    # No Step can have this rule, so parameter should always be na
                    self.model.add_constraint(Constraint(
                        f'{param_name}="na"',
                        f"NA_ALWAYS: {param_name} (no Step uses {obj})"
                    ))
                    constraint_count += 1
                    continue
                
                # Build the OR condition: (Step1="R1" || Step2="R2" || ...)
                or_parts = [f'{step_name}="{rule_id}"' for step_name, rule_id in step_rule_pairs]
                or_condition = ' || '.join(or_parts)
                
                # Constraint 1: If any rule is used, parameter cannot be na
                # (or_condition) => param != "na"
                # Which is: !(or_condition) || !(param = "na")
                constraint1 = f'!({or_condition}) || !({param_name}="na")'
                self.model.add_constraint(Constraint(constraint1, f"NA_FORBIDDEN: {param_name}"))
                constraint_count += 1
                
                # Constraint 2: If no rule is used, parameter must be na
                # !(or_condition) => param = "na"
                # Which is: (or_condition) || (param = "na")
                constraint2 = f'({or_condition}) || ({param_name}="na")'
                self.model.add_constraint(Constraint(constraint2, f"NA_REQUIRED: {param_name}"))
                constraint_count += 1
        
        print(f"  Added {constraint_count} na constraints")
    
    # Removed _add_none_propagation_constraints - no auto-repair of errors
    # Removed add_missing_parameters - no auto-repair of errors
    # Removed _infer_domain - no auto-repair of errors
    
    def simplify_constraints_with_z3(self) -> List[Constraint]:
        """Simplify constraints using Z3 (similar to acts_wrapper_z3.py strategy)"""
        if not Z3_AVAILABLE:
            print("  Z3 not available, skipping constraint simplification")
            return self.model.constraints
        
        print(f"\nNote: Z3 simplification is available but currently disabled.")
        print(f"  All {len(self.model.constraints)} constraints will be kept as-is.")
        print(f"  To enable Z3 simplification, use the Z3ConstraintSimplifier from acts_wrapper_z3.py")
        
        # For now, return constraints as-is
        # The main goal (removing length constraints) has been achieved
        # Full Z3 simplification can be integrated later using acts_wrapper_z3.py's Z3ConstraintSimplifier
        return self.model.constraints
    
    def _parse_constraint_to_z3(self, constraint_str: str) -> Optional[BoolRef]:
        """Parse ACTS constraint string to Z3 expression (simplified version)"""
        if not Z3_AVAILABLE:
            return None
        
        try:
            constraint_str = constraint_str.strip()
            
            # Handle negation: !(...)
            if constraint_str.startswith('!'):
                inner = constraint_str[1:].strip()
                if inner.startswith('(') and inner.endswith(')'):
                    inner = inner[1:-1].strip()
                inner_expr = self._parse_constraint_to_z3(inner)
                return Not(inner_expr) if inner_expr is not None else None
            
            # Handle implication: A => B
            if '=>' in constraint_str:
                parts = constraint_str.split('=>', 1)
                if len(parts) == 2:
                    left = parts[0].strip().strip('()')
                    right = parts[1].strip().strip('()')
                    left_expr = self._parse_constraint_to_z3(left)
                    right_expr = self._parse_constraint_to_z3(right)
                    if left_expr is not None and right_expr is not None:
                        return Implies(left_expr, right_expr)
                    elif right_expr is not None:
                        return right_expr
            
            # Handle && and ||
            if ' && ' in constraint_str:
                parts = self._split_by_operator(constraint_str, ' && ')
                exprs = [self._parse_constraint_to_z3(p.strip()) for p in parts]
                exprs = [e for e in exprs if e is not None]
                if exprs:
                    return And(exprs) if len(exprs) > 1 else exprs[0]
            
            if ' || ' in constraint_str:
                parts = self._split_by_operator(constraint_str, ' || ')
                exprs = [self._parse_constraint_to_z3(p.strip()) for p in parts]
                exprs = [e for e in exprs if e is not None]
                if exprs:
                    return Or(exprs) if len(exprs) > 1 else exprs[0]
            
            # Handle equality: param="value"
            match = re.match(r'(\w+)\s*=\s*"([^"]+)"', constraint_str)
            if match:
                param_name = match.group(1)
                value = match.group(2)
                # Create a Z3 boolean variable for this parameter-value pair
                # For simplicity, we'll use a string-based encoding
                var_name = f"{param_name}_{value}".replace(' ', '_').replace('-', '_')
                return Bool(var_name)
            
            # Handle inequality: param!="value"
            match = re.match(r'(\w+)\s*!=\s*"([^"]+)"', constraint_str)
            if match:
                param_name = match.group(1)
                value = match.group(2)
                var_name = f"{param_name}_{value}".replace(' ', '_').replace('-', '_')
                return Not(Bool(var_name))
            
            return None
        except Exception:
            return None
    
    def _split_by_operator(self, expr: str, op: str) -> List[str]:
        """Split expression by operator, respecting parentheses"""
        parts = []
        current = ""
        depth = 0
        
        i = 0
        while i < len(expr):
            if expr[i] == '(':
                depth += 1
                current += expr[i]
            elif expr[i] == ')':
                depth -= 1
                current += expr[i]
            elif expr[i:i+len(op)] == op and depth == 0:
                parts.append(current.strip())
                current = ""
                i += len(op) - 1
            else:
                current += expr[i]
            i += 1
        
        if current.strip():
            parts.append(current.strip())
        
        return parts if len(parts) > 1 else [expr]
    
    def _z3_to_acts_format(self, z3_expr: BoolRef, original: str) -> str:
        """Convert Z3 expression back to ACTS format (simplified version)"""
        try:
            decl_name = z3_expr.decl().name()
            
            if decl_name == 'true':
                return original  # Keep original for true constraints
            if decl_name == 'false':
                return original  # Keep original for false constraints
            
            # For complex expressions, try to convert back
            # This is a simplified version - for full conversion, see acts_wrapper_z3.py
            return original  # For now, return original if conversion is complex
        except:
            return original
    
    def generate_model(self, output_file: str, simplify: bool = True):
        """Generate final combined model"""
        print(f"\nGenerating combined model: {output_file}")
        
        # Note: No auto-repair - missing parameters or invalid values will cause errors
        
        # Simplify constraints with Z3 if requested
        if simplify and Z3_AVAILABLE:
            self.model.constraints = self.simplify_constraints_with_z3()
        
        # Write to file
        self.model.to_acts_file(output_file)
        
        # Print statistics
        print(f"\nModel statistics:")
        print(f"  Parameters: {len(self.model.parameters)}")
        print(f"  Constraints: {len(self.model.constraints)}")
        print(f"  Model saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined CT model from task, environment, and WP analysis'
    )
    parser.add_argument('task_model', help='Task model file (ACTS format)')
    parser.add_argument('env_model', help='Environment model file (ACTS format)')
    parser.add_argument('wp_analysis', help='WP analysis result file')
    parser.add_argument('output', help='Output combined model file')
    parser.add_argument('--name', default='CombinedModel', help='Model name')
    parser.add_argument('--strength', type=int, default=2, help='Interaction strength')
    parser.add_argument('--no-simplify', action='store_true', help='Disable Z3 constraint simplification')
    
    args = parser.parse_args()
    
    # Generate combined model
    generator = CombinedModelGenerator(args.name, args.strength)
    generator.load_task_model(args.task_model)
    generator.load_environment_model(args.env_model)
    generator.load_wp_analysis(args.wp_analysis)
    generator.generate_model(args.output, simplify=not args.no_simplify)
    
    print("\n✓ Combined model generation complete!")


if __name__ == "__main__":
    main()

