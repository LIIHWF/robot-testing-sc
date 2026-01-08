"""
Task Expression Parser using PLY (Python Lex-Yacc)

This module provides a formal parser for task expressions in the format:
    - if <condition> then <action> else <action>
    - <action> ; <action> ; ...
    
Where:
    - condition: loc <object> <location>
    - action: open <target> | close <target> | turn_on <target> | turn_off <target> | put <object> <target>

Usage:
    from falsifier.task_parser import TaskParser
    
    parser = TaskParser()
    ast = parser.parse("if loc vegetable plate then close cabinet else turn_on microwave")
    print(ast)
"""

import ply.lex as lex
import ply.yacc as yacc
from dataclasses import dataclass
from typing import List, Optional, Union, Any


# ---------------------------------------------------------------------------
# AST Node Definitions
# ---------------------------------------------------------------------------

@dataclass
class LocCondition:
    """Represents a location condition: loc <object> <location>"""
    object: str
    location: str
    
    def __repr__(self):
        return f"LocCondition({self.object}, {self.location})"
    
    def evaluate(self, object_locations: dict) -> bool:
        """Evaluate this condition against object locations."""
        actual_location = object_locations.get(self.object)
        return actual_location == self.location


@dataclass
class DoorOpenCondition:
    """Represents a door state condition: door_open <fixture>"""
    fixture: str
    
    def __repr__(self):
        return f"DoorOpenCondition({self.fixture})"
    
    def evaluate(self, door_states: dict) -> bool:
        """Evaluate this condition against door states."""
        state = door_states.get(self.fixture, "closed")
        return state == "open"


@dataclass
class Action:
    """Represents an action: <action_type> <target> [<object>]"""
    action_type: str  # open, close, turn_on, turn_off, put, pick
    target: str
    object: Optional[str] = None
    
    def __repr__(self):
        if self.object:
            return f"Action({self.action_type}, {self.object}, {self.target})"
        return f"Action({self.action_type}, {self.target})"


@dataclass
class ConditionalTask:
    """Represents: if <condition> then <action> else <action>"""
    condition: Union[LocCondition, DoorOpenCondition]
    then_action: Action
    else_action: Action
    
    def __repr__(self):
        return f"ConditionalTask(if {self.condition} then {self.then_action} else {self.else_action})"
    
    def get_action(self, object_locations: dict = None, door_states: dict = None) -> Action:
        """Get the action to execute based on the condition."""
        if object_locations is None:
            object_locations = {}
        if door_states is None:
            door_states = {}
        
        if isinstance(self.condition, LocCondition):
            if self.condition.evaluate(object_locations):
                return self.then_action
            else:
                return self.else_action
        elif isinstance(self.condition, DoorOpenCondition):
            if self.condition.evaluate(door_states):
                return self.then_action
            else:
                return self.else_action
        else:
            # Default to else action if condition type is unknown
            return self.else_action


@dataclass
class SequentialTask:
    """Represents: <action> ; <action> ; ..."""
    actions: List[Action]
    
    def __repr__(self):
        return f"SequentialTask({' ; '.join(str(a) for a in self.actions)})"


TaskAST = Union[ConditionalTask, SequentialTask]


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class TaskLexer:
    """Lexer for task expressions."""
    
    # Reserved words
    reserved = {
        'if': 'IF',
        'then': 'THEN',
        'else': 'ELSE',
        'loc': 'LOC',
        'door_open': 'DOOR_OPEN',
        'open': 'OPEN',
        'close': 'CLOSE',
        'turn_on': 'TURN_ON',
        'turn_off': 'TURN_OFF',
        'put': 'PUT',
        'pick': 'PICK',
    }
    
    # Token list
    tokens = [
        'IDENTIFIER',
        'SEMICOLON',
    ] + list(reserved.values())
    
    # Simple tokens
    t_SEMICOLON = r';'
    
    # Ignored characters
    t_ignore = ' \t\n'
    
    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        # Check for reserved words
        t.type = self.reserved.get(t.value, 'IDENTIFIER')
        return t
    
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)
    
    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)
        return self.lexer


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TaskParser:
    """Parser for task expressions."""
    
    tokens = TaskLexer.tokens
    
    def __init__(self):
        self.lexer = TaskLexer().build()
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
    
    def parse(self, expression: str) -> Optional[TaskAST]:
        """Parse a task expression and return the AST."""
        if not expression or not expression.strip():
            return None
        return self.parser.parse(expression, lexer=self.lexer)
    
    # Grammar rules
    
    def p_task_conditional(self, p):
        '''task : IF condition THEN action ELSE action'''
        p[0] = ConditionalTask(
            condition=p[2],
            then_action=p[4],
            else_action=p[6]
        )
    
    def p_task_sequential(self, p):
        '''task : action_list'''
        p[0] = SequentialTask(actions=p[1])
    
    def p_action_list_single(self, p):
        '''action_list : action'''
        p[0] = [p[1]]
    
    def p_action_list_multiple(self, p):
        '''action_list : action_list SEMICOLON action'''
        p[0] = p[1] + [p[3]]
    
    def p_condition_loc(self, p):
        '''condition : LOC IDENTIFIER IDENTIFIER'''
        p[0] = LocCondition(object=p[2], location=p[3])
    
    def p_condition_door_open(self, p):
        '''condition : DOOR_OPEN IDENTIFIER'''
        p[0] = DoorOpenCondition(fixture=p[2])
    
    def p_action_open(self, p):
        '''action : OPEN IDENTIFIER'''
        p[0] = Action(action_type='open', target=p[2])
    
    def p_action_close(self, p):
        '''action : CLOSE IDENTIFIER'''
        p[0] = Action(action_type='close', target=p[2])
    
    def p_action_turn_on(self, p):
        '''action : TURN_ON IDENTIFIER'''
        p[0] = Action(action_type='turn_on', target=p[2])
    
    def p_action_turn_off(self, p):
        '''action : TURN_OFF IDENTIFIER'''
        p[0] = Action(action_type='turn_off', target=p[2])
    
    def p_action_put(self, p):
        '''action : PUT IDENTIFIER IDENTIFIER'''
        p[0] = Action(action_type='put', object=p[2], target=p[3])
    
    def p_action_pick(self, p):
        '''action : PICK IDENTIFIER'''
        p[0] = Action(action_type='pick', target=p[2])
    
    def p_error(self, p):
        if p:
            print(f"Syntax error at '{p.value}'")
        else:
            print("Syntax error at EOF")


# ---------------------------------------------------------------------------
# Task Evaluator
# ---------------------------------------------------------------------------

class TaskEvaluator:
    """Evaluates parsed task AST against initial conditions."""
    
    def __init__(self, object_locations: dict = None, door_states: dict = None):
        """
        Initialize evaluator.
        
        Args:
            object_locations: Dict mapping object names to their locations
                             e.g., {"vegetable": "plate", "plate": "table"}
            door_states: Dict mapping fixture names to their door states
                        e.g., {"microwave": "open", "cabinet": "closed"}
        """
        self.object_locations = object_locations or {}
        self.door_states = door_states or {}
    
    def get_actions_to_execute(self, ast: TaskAST) -> List[Action]:
        """
        Get the list of actions to execute based on initial conditions.
        
        For ConditionalTask, evaluates the condition and returns the appropriate branch.
        For SequentialTask, returns all actions.
        """
        if isinstance(ast, ConditionalTask):
            action = ast.get_action(self.object_locations, self.door_states)
            return [action]
        elif isinstance(ast, SequentialTask):
            return ast.actions
        else:
            return []
    
    def generate_language(
        self, 
        actions: List[Action],
        name_mapping: Optional[dict] = None,
    ) -> str:
        """
        Generate natural language instruction from actions.
        
        Args:
            actions: List of actions to describe
            name_mapping: Optional dict mapping abstract names to concrete names
                         e.g., {"vegetable": "carrot", "fruit": "apple"}
        """
        name_mapping = name_mapping or {}
        
        def get_name(abstract_name: str) -> str:
            return name_mapping.get(abstract_name, abstract_name)
        
        lang_parts = []
        for action in actions:
            action_type = action.action_type
            target = action.target
            obj = action.object
            
            if action_type == "open":
                if target == "cabinet":
                    lang_parts.append("open the cabinet door")
                elif target == "drawer":
                    lang_parts.append("open the drawer")
                elif target == "microwave":
                    lang_parts.append("open the microwave door")
                else:
                    lang_parts.append(f"open the {target}")
                    
            elif action_type == "close":
                if target == "cabinet":
                    lang_parts.append("close the cabinet door")
                elif target == "drawer":
                    lang_parts.append("close the drawer")
                elif target == "microwave":
                    lang_parts.append("close the microwave door")
                else:
                    lang_parts.append(f"close the {target}")
                    
            elif action_type == "turn_on":
                if target == "microwave":
                    lang_parts.append("press the start button on the microwave")
                else:
                    lang_parts.append(f"turn on the {target}")
                    
            elif action_type == "turn_off":
                if target == "microwave":
                    lang_parts.append("press the stop button on the microwave")
                else:
                    lang_parts.append(f"turn off the {target}")
                    
            elif action_type == "pick":
                concrete_name = get_name(target)
                lang_parts.append(f"pick the {concrete_name}")
                
            elif action_type == "put":
                concrete_obj = get_name(obj) if obj else "object"
                if target in ("basket", "bowl", "plate"):
                    concrete_target = get_name(target)
                    lang_parts.append(f"place the {concrete_obj} in the {concrete_target}")
                elif target in ("cabinet", "drawer", "microwave"):
                    lang_parts.append(f"place the {concrete_obj} in the {target}")
                else:
                    concrete_target = get_name(target)
                    lang_parts.append(f"place the {concrete_obj} on the {concrete_target}")
        
        if lang_parts:
            return " and then ".join(lang_parts)
        else:
            return "complete the task"
    
    @staticmethod
    def generate_full_task_language(
        ast: TaskAST,
        name_mapping: Optional[dict] = None,
    ) -> str:
        """
        Generate FULL task language instruction without simplification.
        
        For conditional tasks, returns the complete conditional expression.
        For sequential tasks, returns all actions in sequence.
        
        Args:
            ast: Parsed task AST
            name_mapping: Optional dict mapping abstract names to concrete names
        """
        name_mapping = name_mapping or {}
        
        def get_name(abstract_name: str) -> str:
            return name_mapping.get(abstract_name, abstract_name)
        
        def action_to_language(action: Action) -> str:
            """Convert a single action to natural language."""
            action_type = action.action_type
            target = action.target
            obj = action.object
            
            if action_type == "open":
                if target == "cabinet":
                    return "open the cabinet door"
                elif target == "drawer":
                    return "open the drawer"
                elif target == "microwave":
                    return "open the microwave door"
                else:
                    return f"open the {target}"
                    
            elif action_type == "close":
                if target == "cabinet":
                    return "close the cabinet door"
                elif target == "drawer":
                    return "close the drawer"
                elif target == "microwave":
                    return "close the microwave door"
                else:
                    return f"close the {target}"
                    
            elif action_type == "turn_on":
                if target == "microwave":
                    return "press the start button on the microwave"
                else:
                    return f"turn on the {target}"
                    
            elif action_type == "turn_off":
                if target == "microwave":
                    return "press the stop button on the microwave"
                else:
                    return f"turn off the {target}"
                    
            elif action_type == "pick":
                concrete_name = get_name(target)
                return f"pick the {concrete_name}"
                
            elif action_type == "put":
                concrete_obj = get_name(obj) if obj else "object"
                if target in ("basket", "bowl", "plate"):
                    concrete_target = get_name(target)
                    return f"place the {concrete_obj} in the {concrete_target}"
                elif target in ("cabinet", "drawer", "microwave"):
                    return f"place the {concrete_obj} in the {target}"
                else:
                    concrete_target = get_name(target)
                    return f"place the {concrete_obj} on the {concrete_target}"
            
            return f"{action_type} {target}"
        
        if isinstance(ast, ConditionalTask):
            # Generate full conditional expression
            then_action = action_to_language(ast.then_action)
            else_action = action_to_language(ast.else_action)
            
            # Handle different condition types
            if isinstance(ast.condition, LocCondition):
                condition_obj = get_name(ast.condition.object)
                condition_loc = get_name(ast.condition.location)
                return f"if the {condition_obj} is on the {condition_loc}, then {then_action}, else {else_action}"
            elif isinstance(ast.condition, DoorOpenCondition):
                fixture_name = get_name(ast.condition.fixture)
                if fixture_name == "microwave":
                    return f"if the microwave door is open, then {then_action}, else {else_action}"
                elif fixture_name == "cabinet":
                    return f"if the cabinet door is open, then {then_action}, else {else_action}"
                elif fixture_name == "drawer":
                    return f"if the drawer is open, then {then_action}, else {else_action}"
                else:
                    return f"if the {fixture_name} door is open, then {then_action}, else {else_action}"
            else:
                # Fallback for unknown condition types
                return f"if condition is met, then {then_action}, else {else_action}"
            
        elif isinstance(ast, SequentialTask):
            # Generate sequential actions
            lang_parts = [action_to_language(action) for action in ast.actions]
            return " and then ".join(lang_parts)
        
        return "complete the task"


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def parse_and_evaluate(
    expression: str,
    object_locations: dict,
    name_mapping: Optional[dict] = None,
) -> tuple:
    """
    Parse a task expression, evaluate it, and generate language.
    
    Args:
        expression: Task expression string
        object_locations: Dict of object locations for condition evaluation
        name_mapping: Optional dict mapping abstract to concrete names
        
    Returns:
        Tuple of (actions, language_instruction)
    """
    parser = TaskParser()
    ast = parser.parse(expression)
    
    if ast is None:
        return [], "complete the task"
    
    evaluator = TaskEvaluator(object_locations)
    actions = evaluator.get_actions_to_execute(ast)
    language = evaluator.generate_language(actions, name_mapping)
    
    return actions, language


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test the parser
    test_expressions = [
        "if loc vegetable plate then close cabinet else turn_on microwave",
        "if loc vegetable basket then open cabinet else close microwave",
        "close drawer ; open cabinet ; put plate cabinet",
        "open microwave",
        "turn_on microwave",
        "pick vegetable",
    ]
    
    parser = TaskParser()
    
    for expr in test_expressions:
        print(f"\nExpression: {expr}")
        ast = parser.parse(expr)
        print(f"  AST: {ast}")
        
        # Test evaluation
        object_locations = {"vegetable": "plate", "plate": "table", "fruit": "na"}
        evaluator = TaskEvaluator(object_locations)
        actions = evaluator.get_actions_to_execute(ast)
        print(f"  Actions: {actions}")
        
        name_mapping = {"vegetable": "carrot", "plate": "plate"}
        language = evaluator.generate_language(actions, name_mapping)
        print(f"  Language: {language}")

