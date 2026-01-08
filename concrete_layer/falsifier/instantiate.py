"""
Configuration Instantiation Module

This module instantiates concrete configurations from abstract configurations
defined in the 2-way_ct_configurations_trans.json file.

Usage:
    python instantiate.py --config-number 1 [--seed 42] [--output config.json]
    
    # Or use as a library:
    from falsifier.instantiate import ConfigurationInstantiator
    instantiator = ConfigurationInstantiator()
    concrete_config = instantiator.instantiate(config_number=1, seed=42)
"""

import json
import random
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Union
from copy import deepcopy

import numpy as np

# Default seed for deterministic behavior
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Domain Definitions - Available concrete values for sampling
# ---------------------------------------------------------------------------

@dataclass
class ObjectModelDomain:
    """Domain of available object models for each category.
    
    Model paths are relative to robocasa.models.assets_root/objects/
    Based on actual available models in robocasa assets.
    """
    
    # Movable objects (graspable items) - all in objaverse
    # Force bread to be croissant
    bread: List[str] = field(default_factory=lambda: [
        "objaverse/bread/bread_0",
    ])
    
    fruit: List[str] = field(default_factory=lambda: [
        "objaverse/pear/pear_0",
        "objaverse/pear/pear_1",
        "objaverse/banana/banana_1",
        "objaverse/banana/banana_10",
    ])
    
    # Force vegetable to be eggplant
    vegetable: List[str] = field(default_factory=lambda: [
        "objaverse/eggplant/eggplant_0",
        "objaverse/eggplant/eggplant_1",
    ])
    
    # Plate is in objaverse
    plate: List[str] = field(default_factory=lambda: [
        # "objaverse/plate/plate_0",
        # "objaverse/plate/plate_1",
        "objaverse/bowl/bowl_1"
    ])
    
    # Containers
    # Basket is in sketchfab (not objaverse!)
    basket: List[str] = field(default_factory=lambda: [
        "sketchfab/basket/basket_0",
        "sketchfab/basket/basket_2",
        "sketchfab/basket/basket_3",
    ])
    
    # Bowl is in objaverse
    bowl: List[str] = field(default_factory=lambda: [
        "objaverse/bowl/bowl_0",
        "objaverse/bowl/bowl_1",
    ])
    
    # Cup is in objaverse (note: starts from cup_2)
    cup: List[str] = field(default_factory=lambda: [
        # "objaverse/cup/cup_2",
        "objaverse/cup/cup_3",
        # "objaverse/cup/cup_4",
    ])
    
    # Estimated object sizes (horizontal radius in meters)
    # These are conservative estimates based on typical object dimensions
    # Format: (half_width, half_depth) - represents the bounding box half-size in x and y
    # Based on actual sizes from layout YAML files
    object_sizes: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        # Small objects (fruits, vegetables)
        "fruit": (0.04, 0.04),      # ~8cm diameter
        "vegetable": (0.06, 0.06),  # ~12cm diameter (eggplant - larger than typical vegetables)
        "bread": (0.07, 0.06),      # ~14cm x 12cm
        
        # Medium objects (containers)
        "plate": (0.12, 0.12),      # ~24cm diameter
        "bowl": (0.10, 0.10),       # ~20cm diameter
        "cup": (0.05, 0.05),        # ~10cm diameter
        "basket": (0.15, 0.15),     # ~30cm diameter
        
        # Large fixtures (these are typically fixed, but included for reference)
        # Based on layout YAML: cabinet size is [0.3, 0.3, ...] -> half_size = 0.15
        "cabinet": (0.15, 0.15),    # ~30cm x 30cm footprint (from layout)
        # Based on layout YAML: microwave size is [0.35, 0.4, ...] -> half_size = (0.175, 0.2)
        "microwave": (0.175, 0.20), # ~35cm x 40cm footprint (from layout)
        # Based on layout YAML: drawer size varies, use conservative estimate
        "drawer": (0.175, 0.125),  # ~35cm x 25cm footprint (from layout)
    })
    
    def get_models(self, category: str) -> List[str]:
        """Get available models for a given category."""
        category = category.lower()
        if hasattr(self, category):
            return getattr(self, category)
        raise ValueError(f"Unknown object category: {category}")
    
    def get_object_size(self, category: str) -> Tuple[float, float]:
        """Get estimated size (half_width, half_depth) for an object category.
        
        Returns:
            Tuple[float, float]: (half_width, half_depth) in meters
        """
        category = category.lower()
        # TEMPORARY MAPPING: Map fruit to cup
        if category == "fruit":
            category = "cup"
        return self.object_sizes.get(category, (0.05, 0.05))  # Default: 10cm x 10cm


@dataclass
class PositionDomain:
    """Domain of available positions for object placement."""
    
    # Position ranges on counter (x_min, x_max, y_min, y_max)
    counter_range: Tuple[float, float, float, float] = (-0.4, 0.5, -0.3, 0.3)
    
    # Predefined zones for structured placement with enough spacing
    # Each zone is separated by at least 0.15m to avoid overlaps
    zones: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        "left": (-0.4, -0.15, -0.25, 0.0),
        "center_left": (-0.15, 0.05, -0.25, 0.0),
        "center": (0.05, 0.25, -0.25, 0.0),
        "center_right": (0.25, 0.45, -0.25, 0.0),
        "right": (0.45, 0.5, -0.25, 0.0),
        "front_left": (-0.4, -0.15, -0.3, -0.15),
        "front_center": (-0.15, 0.15, -0.3, -0.15),
        "front_right": (0.15, 0.5, -0.3, -0.15),
        "back_left": (-0.4, -0.15, 0.0, 0.25),
        "back_center": (-0.15, 0.15, 0.0, 0.25),
        "back_right": (0.15, 0.5, 0.0, 0.25),
    })
    
    # Rotation range (min, max) in radians
    rotation_range: Tuple[float, float] = (0.0, 6.28)  # 0 to 2*pi
    
    # Minimum gap between objects (in meters)
    # Increased to 8cm to ensure objects don't overlap, especially large fixtures
    min_gap: float = 0.08  # 8cm minimum gap between objects
    
    def check_collision(
        self,
        pos: Tuple[float, float],
        size: Tuple[float, float],
        placed_objects: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> bool:
        """Check if a position collides with any placed objects.
        
        Args:
            pos: (x, y) position to check
            size: (half_width, half_depth) size of the object
            placed_objects: List of ((x, y), (half_width, half_depth)) for each placed object
        
        Returns:
            True if collision detected, False otherwise
        """
        x, y = pos
        half_w, half_d = size
        
        for (other_x, other_y), (other_half_w, other_half_d) in placed_objects:
            # Calculate distance between centers
            dx = abs(x - other_x)
            dy = abs(y - other_y)
            
            # Calculate minimum required distance (sum of half-sizes + gap)
            min_dist_x = half_w + other_half_w + self.min_gap
            min_dist_y = half_d + other_half_d + self.min_gap
            
            # Check if bounding boxes overlap (with gap)
            if dx < min_dist_x and dy < min_dist_y:
                return True  # Collision detected
        
        return False  # No collision
    
    def sample_position(
        self,
        zone: Optional[str] = None,
        deterministic: bool = False,
        placed_objects: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        object_size: Optional[Tuple[float, float]] = None,
        max_attempts: int = 50,
    ) -> Tuple[float, float]:
        """Sample a position from a zone or the full counter range.
        
        In deterministic mode, uses the center of the zone with a small deterministic offset
        to avoid exact overlaps if multiple objects are in the same zone.
        In non-deterministic mode, samples randomly within the zone.
        
        Args:
            zone: Zone name to sample from
            deterministic: Whether to use deterministic placement
            placed_objects: List of ((x, y), (half_width, half_depth)) for collision checking
            object_size: (half_width, half_depth) size of the object being placed
            max_attempts: Maximum number of attempts to find a non-colliding position
        
        Returns:
            (x, y) position tuple
        """
        if placed_objects is None:
            placed_objects = []
        if object_size is None:
            object_size = (0.05, 0.05)  # Default size
        
        if zone and zone in self.zones:
            x_min, x_max, y_min, y_max = self.zones[zone]
        else:
            x_min, x_max, y_min, y_max = self.counter_range
        
        # Initialize default position (center of zone/range)
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        
        # Try to find a non-colliding position
        for attempt in range(max_attempts):
            if deterministic:
                # Use center of zone for deterministic placement
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                
                # Add offset based on attempt number to try different positions
                # Use a DETERMINISTIC offset pattern (golden angle spiral)
                offset_scale = (attempt + 1) * 0.03  # 3cm per attempt
                golden_angle = 2.399963  # Golden angle in radians (~137.5 degrees)
                angle = attempt * golden_angle
                
                # Deterministic spiral pattern - no randomness
                x_offset = offset_scale * math.cos(angle) if attempt > 0 else 0
                y_offset = offset_scale * math.sin(angle) if attempt > 0 else 0
                
                x = x_center + x_offset
                y = y_center + y_offset
                
                # Clamp to bounds
                x = max(x_min, min(x, x_max))
                y = max(y_min, min(y, y_max))
            else:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
            
            # Check collision
            if not self.check_collision((x, y), object_size, placed_objects):
                return (round(x, 3), round(y, 3))
        
        # If we couldn't find a non-colliding position, return the last attempt
        # (This should rarely happen if zones are large enough)
        return (round(x, 3), round(y, 3))
    
    def sample_rotation(self) -> float:
        """Sample a random rotation."""
        return round(random.uniform(*self.rotation_range), 3)
    
    def sample_exact_rotation(self) -> float:
        """Sample a rotation for deterministic mode (use specific values)."""
        # Use common rotation values for determinism
        rotations = [0.0, 1.57, 3.14, 4.71]  # 0, 90, 180, 270 degrees
        return random.choice(rotations)


@dataclass
class FixtureDomain:
    """Domain of available fixtures."""
    
    fixtures: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "counter": {
            "id": "FixtureType.COUNTER",
            "size": (0.45, 0.55),
        },
        "cabinet": {
            "id": "FixtureType.CABINET",
            "has_door": True,
        },
        "drawer": {
            "id": "FixtureType.DRAWER", 
            "has_door": True,
        },
        "microwave": {
            "id": "FixtureType.MICROWAVE",
            "has_door": True,
            "has_running_state": True,
        },
    })
    
    # Positions relative to each fixture type
    fixture_positions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "counter": (0.0, 0.0),
        "cabinet": (0.1, 0.0),
        "drawer": (0.0, 0.0),
        "microwave": (0.3, 0.2),  # Matches layout file position
    })


# ---------------------------------------------------------------------------
# Abstract Configuration Parser
# ---------------------------------------------------------------------------

class AbstractConfigParser:
    """Parser for abstract configuration from JSON."""
    
    # Mapping from abstract location names to fixture/container types
    LOCATION_TYPES = {
        "table": "fixture",      # On the table/counter surface
        "counter": "fixture",    # On the counter
        "basket": "container",   # In/on a basket
        "bowl": "container",     # In/on a bowl
        "plate": "container",    # On a plate (plate is both movable and container)
        "cabinet": "fixture",    # In the cabinet
        "drawer": "fixture",     # In the drawer
        "microwave": "fixture",  # In the microwave
        "na": None,              # Not placed (not relevant)
    }
    
    # Objects that can be both movable and containers
    DUAL_ROLE_OBJECTS = {"plate"}
    
    @staticmethod
    def parse_initial_conditions(conditions: Dict[str, bool]) -> Dict[str, Any]:
        """Parse initial conditions into structured format.
        
        IMPORTANT: If location is "na", the object should NOT be present.
        Only objects with non-"na" locations are included.
        """
        parsed = {
            "object_locations": {},
            "door_states": {},
            "running_states": {},
        }
        
        for condition, value in conditions.items():
            if condition.startswith("loc("):
                # Parse location condition: loc(object, location)
                inner = condition[4:-1]  # Remove "loc(" and ")"
                parts = inner.split(", ")
                if len(parts) == 2:
                    obj, loc = parts
                    # Only include objects with non-"na" locations AND value=True
                    # If value=False or loc="na", the object should NOT be present
                    if value and loc != "na":
                        parsed["object_locations"][obj] = loc
                    # If loc="na" or value=False, do not add to object_locations
                    # This ensures objects with "na" location are not created
                    
            elif condition.startswith("door_open("):
                # Parse door state: door_open(fixture)
                fixture = condition[10:-1]  # Remove "door_open(" and ")"
                parsed["door_states"][fixture] = "open" if value else "closed"
                
            elif condition.startswith("running("):
                # Parse running state: running(fixture)
                fixture = condition[8:-1]  # Remove "running(" and ")"
                parsed["running_states"][fixture] = "running" if value else "stopped"
        
        return parsed
    
    @staticmethod
    def parse_task_expression(expression: str) -> Dict[str, Any]:
        """Parse task expression into structured format."""
        parsed = {
            "type": None,
            "condition": None,
            "then_actions": [],
            "else_actions": [],
            "sequence_actions": [],
        }
        
        expression = expression.strip()
        
        # Check if it's a conditional expression
        if expression.startswith("if "):
            parsed["type"] = "conditional"
            
            # Split by "then" and "else"
            try:
                # Format: "if <condition> then <action1> else <action2>"
                parts = expression[3:]  # Remove "if "
                
                if " then " in parts:
                    cond_part, rest = parts.split(" then ", 1)
                    parsed["condition"] = cond_part.strip()
                    
                    if " else " in rest:
                        then_part, else_part = rest.split(" else ", 1)
                        parsed["then_actions"] = AbstractConfigParser._parse_actions(then_part)
                        parsed["else_actions"] = AbstractConfigParser._parse_actions(else_part)
                    else:
                        parsed["then_actions"] = AbstractConfigParser._parse_actions(rest)
            except Exception:
                pass
        else:
            # Sequential actions separated by ";"
            parsed["type"] = "sequence"
            parsed["sequence_actions"] = AbstractConfigParser._parse_actions(expression)
        
        return parsed
    
    @staticmethod
    def _parse_actions(action_str: str) -> List[Dict[str, Any]]:
        """Parse action string into list of action dictionaries."""
        actions = []
        
        # Split by ";" for sequential actions
        for action in action_str.split(";"):
            action = action.strip()
            if not action:
                continue
                
            parts = action.split()
            if len(parts) >= 2:
                action_type = parts[0]
                
                if action_type in ("open", "close"):
                    actions.append({
                        "action": action_type,
                        "target": parts[1],
                    })
                elif action_type == "turn_on":
                    actions.append({
                        "action": "turn_on",
                        "target": parts[1],
                    })
                elif action_type == "turn_off":
                    actions.append({
                        "action": "turn_off",
                        "target": parts[1],
                    })
                elif action_type == "put":
                    if len(parts) >= 3:
                        actions.append({
                            "action": "place",
                            "object": parts[1],
                            "target": parts[2],
                        })
                elif action_type == "pick":
                    actions.append({
                        "action": "pick",
                        "object": parts[1],
                    })
        
        return actions


# ---------------------------------------------------------------------------
# Configuration Instantiator
# ---------------------------------------------------------------------------

class ConfigurationInstantiator:
    """
    Instantiates concrete configurations from abstract configurations.
    
    This class:
    1. Loads abstract configurations from JSON
    2. Constructs a domain of concrete parameter values
    3. Samples from the domain to create concrete configurations
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        object_domain: Optional[ObjectModelDomain] = None,
        position_domain: Optional[PositionDomain] = None,
        fixture_domain: Optional[FixtureDomain] = None,
    ):
        """
        Initialize the instantiator.
        
        Args:
            config_path: Path to the abstract configuration JSON file.
                        Defaults to 'data/2-way_ct_configurations_trans.json'
            object_domain: Domain of available object models
            position_domain: Domain of available positions
            fixture_domain: Domain of available fixtures
        """
        if config_path is None:
            # Find the config file relative to this module
            module_dir = Path(__file__).parent.parent
            # config_path = module_dir / "data" / "2-way_ct_configurations_trans.json"
            config_path = module_dir / "data" / "single_all.json"
        
        self.config_path = Path(config_path)
        self.object_domain = object_domain or ObjectModelDomain()
        self.position_domain = position_domain or PositionDomain()
        self.fixture_domain = fixture_domain or FixtureDomain()
        
        # Load configurations
        self._configurations: Optional[List[Dict]] = None
    
    @property
    def configurations(self) -> List[Dict]:
        """Lazy load configurations from JSON file."""
        if self._configurations is None:
            with open(self.config_path, 'r') as f:
                self._configurations = json.load(f)
        return self._configurations
    
    def get_abstract_config(self, config_number: int) -> Dict:
        """Get abstract configuration by number."""
        for config in self.configurations:
            if config.get("configuration_number") == config_number:
                return config
        raise ValueError(f"Configuration number {config_number} not found")
    
    def get_total_configurations(self) -> int:
        """Get total number of configurations available."""
        return len(self.configurations)
    
    def instantiate(
        self,
        config_number: int,
        seed: Optional[int] = None,
        enforce_determinism: bool = True,
    ) -> Dict[str, Any]:
        """
        Instantiate a concrete configuration from an abstract configuration.
        
        Args:
            config_number: The configuration number to instantiate
            seed: Random seed for reproducibility. If None, uses DEFAULT_SEED for determinism.
            enforce_determinism: Whether to use deterministic placement
            
        Returns:
            Dictionary containing concrete configuration matching run_give.py format:
            {
                "object_plan": [...],
                "task_plan": [...],
                "success_plan": [...],
                "fixture_plan": {...},
                "abstract_config": {...},  # Original abstract config for reference
            }
        """
        # Always set seed for reproducibility (use default if not provided)
        actual_seed = seed if seed is not None else DEFAULT_SEED
        random.seed(actual_seed)
        np.random.seed(actual_seed)
        
        # Get abstract configuration
        abstract_config = self.get_abstract_config(config_number)
        
        # Parse the abstract configuration
        parsed_conditions = AbstractConfigParser.parse_initial_conditions(
            abstract_config.get("initial_conditions", {})
        )
        task_expression = abstract_config.get("task_expression", "")
        parsed_task = AbstractConfigParser.parse_task_expression(task_expression)
        # Store original expression for PLY parser
        parsed_task["_original_expression"] = task_expression
        
        # Extract objects that need to be picked up from task expression
        objects_to_pick = self._extract_objects_to_pick(parsed_task)
        
        # Check if task involves drawer or cabinet to determine source object position
        has_drawer_or_cabinet = self._check_has_drawer_or_cabinet(parsed_task)
        
        # Check if task is specifically "put to cabinet" (place/put action targeting cabinet)
        is_put_to_cabinet = self._check_is_put_to_cabinet(parsed_task)
        
        # Build concrete configuration
        object_plan = self._build_object_plan(
            parsed_conditions, 
            abstract_config.get("raw_parameters", {}),
            enforce_determinism,
            objects_to_pick=objects_to_pick,
            has_drawer_or_cabinet=has_drawer_or_cabinet,
            is_put_to_cabinet=is_put_to_cabinet
        )
        
        fixture_plan = self._build_fixture_plan(parsed_conditions, parsed_task)
        
        # Get fixture requirements to filter initial states
        fixture_requirements = fixture_plan.get("_fixture_requirements", {})
        initial_state = self._build_initial_state(parsed_conditions, fixture_requirements)
        
        task_plan = self._build_task_plan(parsed_task, object_plan)
        
        success_plan = self._build_success_plan(parsed_task, object_plan)
        
        # Build language instruction for the robot (using concrete names from object_plan)
        lang = self._build_language_instruction(parsed_task, parsed_conditions, object_plan)
        
        # Determine layout_id based on required fixtures
        # This matches the logic in executor.py DynamicEnvironmentFactory
        layout_id = self._determine_layout_id(fixture_plan)
        
        return {
            "object_plan": object_plan,
            "task_plan": task_plan,
            "success_plan": success_plan,
            "fixture_plan": fixture_plan,
            "initial_state": initial_state,
            "lang": lang,
            "task_expression": abstract_config.get("task_expression", ""),
            "abstract_config": abstract_config,
            "enforce_determinism": enforce_determinism,
            "layout_id": layout_id,
            "use_distractors": False,  # Disable distractor fixtures (including paper_towel)
            "distractor_config": None,  # Explicitly set to None to exclude paper_towel
        }
    
    def _extract_objects_to_pick(self, parsed_task: Dict) -> set:
        """Extract set of object names that need to be picked up from task expression.
        
        This includes:
        1. Objects with explicit "pick" actions
        2. Objects in "place"/"put" actions (source objects that need to be moved)
        """
        objects_to_pick = set()
        
        # Get all actions (from either conditional or sequence)
        all_actions = []
        if parsed_task["type"] == "conditional":
            all_actions.extend(parsed_task.get("then_actions", []))
            all_actions.extend(parsed_task.get("else_actions", []))
        else:
            all_actions.extend(parsed_task.get("sequence_actions", []))
        
        # Find objects with pick or place actions
        for action in all_actions:
            action_type = action.get("action")
            if action_type == "pick":
                obj_name = action.get("object", "obj")
                if obj_name:
                    objects_to_pick.add(obj_name)
            elif action_type in ("place", "put"):
                # Objects in place/put actions are source objects that need to be picked up first
                obj_name = action.get("object", "obj")
                if obj_name:
                    objects_to_pick.add(obj_name)
        
        return objects_to_pick
    
    def _check_has_drawer_or_cabinet(self, parsed_task: Dict) -> bool:
        """Check if task involves drawer or cabinet."""
        # Get all actions (from either conditional or sequence)
        all_actions = []
        if parsed_task["type"] == "conditional":
            all_actions.extend(parsed_task.get("then_actions", []))
            all_actions.extend(parsed_task.get("else_actions", []))
        else:
            all_actions.extend(parsed_task.get("sequence_actions", []))
        
        # Check if any action targets drawer or cabinet
        for action in all_actions:
            target = action.get("target", "")
            if target in ("drawer", "cabinet"):
                return True
        
        return False
    
    def _check_is_put_to_cabinet(self, parsed_task: Dict) -> bool:
        """Check if task is specifically 'put to cabinet' (place/put action targeting cabinet)."""
        # Get all actions (from either conditional or sequence)
        all_actions = []
        if parsed_task["type"] == "conditional":
            all_actions.extend(parsed_task.get("then_actions", []))
            all_actions.extend(parsed_task.get("else_actions", []))
        else:
            all_actions.extend(parsed_task.get("sequence_actions", []))
        
        # Check if any place/put action targets cabinet
        for action in all_actions:
            action_type = action.get("action", "")
            target = action.get("target", "")
            if action_type in ("place", "put") and target == "cabinet":
                return True
        
        return False
    
    def _build_object_plan(
        self,
        parsed_conditions: Dict,
        raw_params: Dict,
        enforce_determinism: bool,
        objects_to_pick: Optional[set] = None,
        has_drawer_or_cabinet: bool = False,
        is_put_to_cabinet: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build concrete object plan from parsed conditions.
        
        Uses collision detection to ensure objects don't overlap, considering their sizes.
        Objects that need to be picked up are placed at:
        - (0.5, -0.8) if no drawer or cabinet
        - (-0.5, -0.7) if drawer or cabinet is present
        - (-0.5, -0.8) if specifically "put to cabinet" (left side for cabinet tasks)
        """
        if objects_to_pick is None:
            objects_to_pick = set()
        
        object_plan = []
        object_locations = parsed_conditions.get("object_locations", {})
        
        # Track zones used to avoid overlapping placements
        used_zones = []
        available_zones = list(self.position_domain.zones.keys())
        
        # Track placed objects for collision detection: ((x, y), (half_width, half_depth))
        placed_objects: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        
        # Track objects to pick for special placement
        objects_to_pick_list = []
        
        # First pass: create containers/fixtures that other objects reference
        # If an object is placed on/in a container, that container must exist
        potential_container_objects = set()
        for obj_name, location in object_locations.items():
            if location and location != "na":
                # Check if location is another object type (container)
                if location in ("plate", "basket", "bowl"):
                    potential_container_objects.add(location)
        
        # Also check raw_params for container locations
        # This handles cases where containers are specified in raw_params but not in initial_conditions
        for container_name in ("plate", "basket", "bowl"):
            container_location_key = f"{container_name}_Location"
            container_location_in_raw = raw_params.get(container_location_key, "na")
            if container_location_in_raw != "na":
                potential_container_objects.add(container_name)
                # Also add to object_locations if not already present
                if container_name not in object_locations:
                    object_locations[container_name] = container_location_in_raw
        
        # Track which containers are actually created
        created_container_objects = set()
        
        # Add containers first
        # If container is referenced by another object, it must be created
        # All containers are placed on table/counter
        for container_name in potential_container_objects:
            # Check if the container itself has a location in raw_params
            # If container location is "na" in raw_params, it shouldn't exist
            container_location_key = f"{container_name}_Location"
            container_location_in_raw = raw_params.get(container_location_key, "na")
            
            # Also check object_locations (may have been added above)
            container_location_in_object_locations = object_locations.get(container_name, "na")
            
            # If container location is "na" in both raw_params and object_locations, don't create it
            # This means the container should not exist, even if other objects reference it
            if container_location_in_raw == "na" and container_location_in_object_locations == "na":
                continue  # Skip creating this container
            
            # Mark this container as actually created
            created_container_objects.add(container_name)
            
            # Check if the container itself has a location in object_locations
            container_location = object_locations.get(container_name)
            
            # If container location is "na" in object_locations but not in raw_params,
            # use default "table" (this shouldn't happen if logic is correct)
            if container_location == "na":
                container_location = "table"
            
            # All containers go on table/counter, regardless of their specified location
            # (location like "cabinet", "drawer" are just references, not actual placement)
            if container_location and container_location not in ("table", "counter"):
                # If location is a fixture name, still place on table
                container_location = "table"
            
            # Create the container
            zone = self._get_available_zone(used_zones, available_zones)
            used_zones.append(zone)
            
            # Basket and plate always at center (0, 0) for first iteration
            custom_pos = (0.0, 0.0) if container_name in ("basket", "plate") else None
            
            container_entry = self._create_object_entry(
                name=container_name,
                category=container_name,
                location="table",  # Always place containers on table/counter
                zone=zone,
                enforce_determinism=enforce_determinism,
                is_container=True,
                placed_objects=placed_objects,
                custom_position=custom_pos,
            )
            
            # Ensure basket and plate are at center (0, 0)
            if container_name in ("basket", "plate") and "placement" in container_entry:
                container_entry["placement"]["pos"] = (0.0, 0.0)
            
            object_plan.append(container_entry)
            
            # Add to placed_objects for collision detection
            if "placement" in container_entry and "pos" in container_entry["placement"]:
                pos = container_entry["placement"]["pos"]
                size = self.object_domain.get_object_size(container_name)
                placed_objects.append((pos, size))
        
        # Add movable objects
        # First, collect objects that need to be picked up
        for obj_name, location in object_locations.items():
            if location == "na" or location is None:
                continue
            
            # Skip if this object is already added as a container
            if obj_name in created_container_objects:
                continue
            
            # If this object needs to be picked up, collect it for special placement
            if obj_name in objects_to_pick:
                objects_to_pick_list.append((obj_name, location))
                continue
        
        # Place objects that need to be picked up
        # For "put to cabinet" tasks, use left side position (-0.5, -0.8) to match tabletop_cabinet_pnp.py
        # For other drawer/cabinet tasks, use (-0.5, -0.7)
        # Otherwise use (0.5, -0.8)
        if is_put_to_cabinet:
            pick_base_pos = (-0.5, -0.8)  # Left side for cabinet tasks (matches tabletop_cabinet_pnp.py left-handed)
        elif has_drawer_or_cabinet:
            pick_base_pos = (-0.5, -0.7)
        else:
            pick_base_pos = (0.5, -0.8)
        pick_placed_objects = []
        for idx, (obj_name, location) in enumerate(objects_to_pick_list):
            # Get object size for collision detection
            object_size = self.object_domain.get_object_size(obj_name)
            
            # Calculate position around base position
            if len(objects_to_pick_list) == 1:
                # Single object: use base position exactly
                pos = pick_base_pos
            elif idx == 0:
                # First object: always at base position (0.5, -0.8)
                pos = pick_base_pos
            else:
                # Multiple objects: arrange others around the first object
                # Calculate radius based on object size and minimum gap
                half_w, half_d = object_size
                max_size = max(half_w, half_d)
                base_radius = max_size + self.position_domain.min_gap
                
                # Try to find a non-colliding position
                max_attempts = 50
                pos = None
                for attempt in range(max_attempts):
                    if enforce_determinism:
                        # Deterministic: use angle-based circular distribution
                        # Distribute objects evenly around the base position
                        # Start from idx=1 (skip first object which is at center)
                        angle_step = 2 * math.pi / (len(objects_to_pick_list) - 1)
                        angle = (idx - 1) * angle_step
                        # Use spiral pattern if needed (increase radius for each attempt)
                        current_radius = base_radius * (1 + attempt * 0.15)
                        # Use cosine and sine for circular distribution
                        x_offset = current_radius * math.cos(angle)
                        y_offset = current_radius * math.sin(angle)
                    else:
                        # Non-deterministic: use angle-based with some randomness
                        angle_step = 2 * math.pi / (len(objects_to_pick_list) - 1)
                        angle = (idx - 1) * angle_step + random.uniform(-0.2, 0.2)  # Add some randomness
                        current_radius = base_radius * (1 + attempt * 0.15) * random.uniform(0.9, 1.1)
                        x_offset = current_radius * math.cos(angle)
                        y_offset = current_radius * math.sin(angle)
                    
                    candidate_pos = (
                        round(pick_base_pos[0] + x_offset, 3),
                        round(pick_base_pos[1] + y_offset, 3)
                    )
                    
                    # Check collision with other pick objects
                    if not self.position_domain.check_collision(candidate_pos, object_size, pick_placed_objects):
                        pos = candidate_pos
                        break
                
                if pos is None:
                    # Fallback: use grid-like pattern
                    if enforce_determinism:
                        # Deterministic fallback: grid-like pattern
                        # Account for first object at center
                        grid_size = int(math.ceil(math.sqrt(len(objects_to_pick_list) - 1)))
                        row = (idx - 1) // grid_size
                        col = (idx - 1) % grid_size
                        spacing = base_radius * 2.5
                        x_offset = (col - (grid_size - 1) / 2) * spacing
                        y_offset = (row - (grid_size - 1) / 2) * spacing
                    else:
                        # Non-deterministic fallback: random offset
                        x_offset = base_radius * random.uniform(-1.5, 1.5)
                        y_offset = base_radius * random.uniform(-1.5, 1.5)
                    
                    pos = (
                        round(pick_base_pos[0] + x_offset, 3),
                        round(pick_base_pos[1] + y_offset, 3)
                    )
            
            # Create object entry with special placement
            obj_entry = self._create_object_entry(
                name=obj_name,
                category=obj_name,
                location="table",
                zone=None,  # Not using zone for pick objects
                enforce_determinism=enforce_determinism,
                is_container=False,
                placed_objects=placed_objects,
                custom_position=pos,  # Use custom position
            )
            
            object_plan.append(obj_entry)
            
            # Add to placed_objects for collision detection
            placed_objects.append((pos, object_size))
            pick_placed_objects.append((pos, object_size))
        
        # Add other movable objects (not to be picked up)
        for obj_name, location in object_locations.items():
            if location == "na" or location is None:
                continue
            
            # Skip if this object is already added as a container
            if obj_name in created_container_objects:
                continue
            
            # Skip if this object is already added as a pick object
            if obj_name in objects_to_pick:
                continue
            
            # Determine if placement is relative to a container or fixture
            # Only use container reference if the container was actually created
            if location in created_container_objects:
                # Place relative to another object (container)
                # No collision check needed for object_ref placements (they're on top of containers)
                obj_entry = self._create_object_entry(
                    name=obj_name,
                    category=obj_name,
                    location=location,
                    zone=None,  # Will use object_ref
                    enforce_determinism=enforce_determinism,
                    is_container=False,
                    object_ref=location,
                    placed_objects=placed_objects,
                )
            elif location in ("plate", "basket", "bowl"):
                # Object references a container that wasn't created (location was "na")
                # Place it on table instead
                zone = self._get_available_zone(used_zones, available_zones)
                used_zones.append(zone)
                
                obj_entry = self._create_object_entry(
                    name=obj_name,
                    category=obj_name,
                    location="table",  # Place on table since container doesn't exist
                    zone=zone,
                    enforce_determinism=enforce_determinism,
                    is_container=False,
                    placed_objects=placed_objects,
                )
                # Add to placed_objects for collision detection
                if "placement" in obj_entry and "pos" in obj_entry["placement"]:
                    pos = obj_entry["placement"]["pos"]
                    size = self.object_domain.get_object_size(obj_name)
                    placed_objects.append((pos, size))
            elif location in ("cabinet", "drawer", "microwave"):
                # Objects placed in fixtures should still be on the counter/table
                # The fixture location is just for reference, but object goes on counter
                zone = self._get_available_zone(used_zones, available_zones)
                used_zones.append(zone)
                
                obj_entry = self._create_object_entry(
                    name=obj_name,
                    category=obj_name,
                    location="table",  # Always place on table/counter
                    zone=zone,
                    enforce_determinism=enforce_determinism,
                    is_container=False,
                    placed_objects=placed_objects,
                )
                # Add to placed_objects for collision detection
                if "placement" in obj_entry and "pos" in obj_entry["placement"]:
                    pos = obj_entry["placement"]["pos"]
                    size = self.object_domain.get_object_size(obj_name)
                    placed_objects.append((pos, size))
            else:
                # Place relative to fixture (table/counter)
                zone = self._get_available_zone(used_zones, available_zones)
                used_zones.append(zone)
                
                obj_entry = self._create_object_entry(
                    name=obj_name,
                    category=obj_name,
                    location="table",  # Always use table/counter
                    zone=zone,
                    enforce_determinism=enforce_determinism,
                    is_container=False,
                    placed_objects=placed_objects,
                )
                # Add to placed_objects for collision detection
                if "placement" in obj_entry and "pos" in obj_entry["placement"]:
                    pos = obj_entry["placement"]["pos"]
                    size = self.object_domain.get_object_size(obj_name)
                    placed_objects.append((pos, size))
            
            object_plan.append(obj_entry)
        
        return object_plan
    
    def _create_object_entry(
        self,
        name: str,
        category: str,
        location: str,
        zone: Optional[str],
        enforce_determinism: bool,
        is_container: bool = False,
        object_ref: Optional[str] = None,
        placed_objects: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        custom_position: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Create a single object entry for the object plan.
        
        Args:
            name: Object name
            category: Object category (for model selection)
            location: Abstract location (table, cabinet, etc.)
            zone: Zone name for placement
            enforce_determinism: Whether to use deterministic placement
            is_container: Whether this is a container object
            object_ref: Reference to another object for relative placement
            placed_objects: List of ((x, y), (half_width, half_depth)) for collision detection
        """
        if placed_objects is None:
            placed_objects = []
        
        # TEMPORARY MAPPING: Map fruit to cup
        actual_category = category
        if category == "fruit":
            actual_category = "cup"
        
        # Sample model
        try:
            models = self.object_domain.get_models(actual_category)
            model = random.choice(models)
        except ValueError:
            # Unknown category, use a default
            model = f"objaverse/{actual_category}/{actual_category}_0"
        
        # Extract concrete object name from model path
        # e.g., "objaverse/carrot/carrot_0" -> "carrot"
        concrete_name = self._extract_concrete_name(model)
        
        entry = {
            "name": name,
            "model": model,
            "display_name": concrete_name,
            "category": category,  # Keep original category for reference
        }
        
        if not is_container:
            entry["graspable"] = True
        
        # Build placement
        if object_ref:
            # Placement relative to another object
            # Use concrete rotation value (90 degrees = π/2) for deterministic placement
            rotation = math.pi / 2 if enforce_determinism else self.position_domain.sample_rotation()
            entry["placement"] = {
                "object_ref": object_ref,
                "pos": (0.0, 0.0),  # Concrete position
                "on_top": True,
                "rotation": rotation,  # Concrete rotation value (not a range)
            }
        else:
            # Placement relative to fixture
            fixture = self._map_location_to_fixture(location)
            
            # Get object size for collision detection
            # Use actual_category for size calculation (fruit -> cup)
            object_size = self.object_domain.get_object_size(actual_category)
            
            # Use custom position if provided (for objects to pick), otherwise sample
            if custom_position is not None:
                pos = custom_position
            elif name in ("basket", "plate"):
                # Basket and plate always at center (0, 0) for first iteration
                pos = (0.0, 0.0)
            else:
                # Sample position with collision detection
                pos = self.position_domain.sample_position(
                    zone=zone,
                    deterministic=enforce_determinism,
                    placed_objects=placed_objects,
                    object_size=object_size,
                )
            
            # Use concrete rotation value (not a range) for deterministic placement
            if enforce_determinism:
                rotation = math.pi / 2  # Use exact 90 degrees (π/2) for deterministic mode
            else:
                rotation = self.position_domain.sample_rotation()
            
            entry["placement"] = {
                "fixture": fixture,
                "pos": pos,  # Concrete position tuple
                "rotation": rotation,  # Concrete rotation value (not a range)
            }
        
        return entry
    
    def _map_location_to_fixture(self, location: str) -> str:
        """Map abstract location to concrete fixture name."""
        location_to_fixture = {
            "table": "counter",
            "counter": "counter",
            "cabinet": "cabinet",
            "drawer": "drawer",
            "microwave": "microwave",
        }
        return location_to_fixture.get(location, "counter")
    
    def _extract_concrete_name(self, model_path: str) -> str:
        """Extract concrete object name from model path.
        
        Examples:
            "objaverse/carrot/carrot_0" -> "carrot"
            "objaverse/pear/pear_1" -> "pear"
            "objaverse/potato/potato_0" -> "potato"
            "objaverse/bowl/bowl_2" -> "bowl"
        """
        # Split by "/" and get the category folder name
        parts = model_path.split("/")
        if len(parts) >= 2:
            # The category is typically the second-to-last part
            # e.g., "objaverse/carrot/carrot_0" -> parts = ["objaverse", "carrot", "carrot_0"]
            return parts[-2]
        elif len(parts) == 1:
            # Just a name like "carrot_0" -> extract base name
            base = parts[0]
            # Remove trailing number suffix like "_0", "_1", etc.
            if "_" in base:
                return base.rsplit("_", 1)[0]
            return base
        return model_path
    
    def _get_available_zone(
        self, 
        used_zones: List[str], 
        available_zones: List[str]
    ) -> str:
        """Get an available zone that hasn't been used."""
        for zone in available_zones:
            if zone not in used_zones:
                return zone
        # All zones used, return a random one
        return random.choice(available_zones)
    
    def _build_fixture_plan(
        self, 
        parsed_conditions: Dict,
        parsed_task: Dict,
    ) -> Dict[str, Any]:
        """Build fixture plan from parsed conditions and task actions.
        
        Only counter is registered in fixture_plan. Other fixtures (cabinet, drawer,
        microwave) are found at runtime using get_fixture() and their states are
        stored in initial_state for _reset_internal.
        """
        from .task_parser import TaskParser, TaskEvaluator
        
        # Core fixture plan - only counter is pre-registered
        fixture_plan = {
            "counter": {
                "id": "FixtureType.COUNTER",
                "size": (0.45, 0.55),
            },
            "init_robot_base_pos": "counter",
        }
        
        # Parse task to find which fixtures are needed and their initial states
        task_expression = parsed_task.get("_original_expression", "")
        object_locations = parsed_conditions.get("object_locations", {})
        
        from .task_parser import TaskParser, ConditionalTask, SequentialTask
        parser = TaskParser()
        ast = parser.parse(task_expression)
        
        # Store fixture requirements in a separate field
        # These will be used by the executor to set up fixtures at runtime
        fixture_requirements = {}
        
        if ast is not None:
            # For conditional tasks, we need fixtures from BOTH branches
            # For sequential tasks, we need fixtures from all actions
            if isinstance(ast, ConditionalTask):
                # Get fixtures from both then and else branches
                all_actions = [ast.then_action, ast.else_action]
            elif isinstance(ast, SequentialTask):
                all_actions = ast.actions
            else:
                evaluator = TaskEvaluator(object_locations)
                all_actions = evaluator.get_actions_to_execute(ast)
            
            for action in all_actions:
                target = action.target
                action_type = action.action_type
                
                if target == "cabinet":
                    if "cabinet" not in fixture_requirements:
                        # Use concrete door state values: closed = (0.0, 0.0), open = (0.90, 1.0)
                        if action_type == "close":
                            # Door starts open (0.90-1.0) so it can be closed
                            fixture_requirements["cabinet"] = {
                                "type": "DOOR_HINGE_SINGLE",
                                "door_state_min": 0.90,
                                "door_state_max": 1.0,
                            }
                        elif action_type == "open":
                            # Door starts closed (0.0) so it can be opened
                            fixture_requirements["cabinet"] = {
                                "type": "DOOR_HINGE_SINGLE",
                                "door_state_min": 0.0,
                                "door_state_max": 0.0,
                            }
                        elif action_type in ("put", "place"):
                            # For place actions, cabinet needs to be open to place objects inside
                            fixture_requirements["cabinet"] = {
                                "type": "DOOR_HINGE_SINGLE",
                                "door_state_min": 0.90,
                                "door_state_max": 1.0,
                            }
                    else:
                        # Update initial state based on action
                        if action_type == "close":
                            fixture_requirements["cabinet"]["door_state_min"] = 0.90
                            fixture_requirements["cabinet"]["door_state_max"] = 1.0
                        elif action_type == "open":
                            fixture_requirements["cabinet"]["door_state_min"] = 0.0
                            fixture_requirements["cabinet"]["door_state_max"] = 0.0
                        elif action_type in ("put", "place"):
                            # Ensure cabinet is open for place actions
                            fixture_requirements["cabinet"]["door_state_min"] = 0.90
                            fixture_requirements["cabinet"]["door_state_max"] = 1.0
                            
                elif target == "drawer":
                    if "drawer" not in fixture_requirements:
                        # Use concrete door state values: closed = (0.0, 0.0), open = (0.90, 1.0)
                        if action_type == "close":
                            # Drawer starts open (0.90-1.0) so it can be closed
                            fixture_requirements["drawer"] = {
                                "type": "DRAWER",
                                "door_state_min": 0.90,
                                "door_state_max": 1.0,
                            }
                        elif action_type == "open":
                            # Drawer starts closed (0.0) so it can be opened
                            fixture_requirements["drawer"] = {
                                "type": "DRAWER",
                                "door_state_min": 0.0,
                                "door_state_max": 0.0,
                            }
                        elif action_type in ("put", "place"):
                            # For place actions, drawer needs to be open to place objects inside
                            fixture_requirements["drawer"] = {
                                "type": "DRAWER",
                                "door_state_min": 0.90,
                                "door_state_max": 1.0,
                            }
                    else:
                        if action_type == "close":
                            fixture_requirements["drawer"]["door_state_min"] = 0.90
                            fixture_requirements["drawer"]["door_state_max"] = 1.0
                        elif action_type == "open":
                            fixture_requirements["drawer"]["door_state_min"] = 0.0
                            fixture_requirements["drawer"]["door_state_max"] = 0.0
                        elif action_type in ("put", "place"):
                            # Ensure drawer is open for place actions
                            fixture_requirements["drawer"]["door_state_min"] = 0.90
                            fixture_requirements["drawer"]["door_state_max"] = 1.0
                            
                elif target == "microwave":
                    if "microwave" not in fixture_requirements:
                        req = {"type": "MICROWAVE"}
                        if action_type == "close":
                            # Microwave door starts open (0.90-1.0) so it can be closed
                            req["door_state_min"] = 0.90
                            req["door_state_max"] = 1.0
                        elif action_type == "open":
                            # Microwave door starts closed (0.0) so it can be opened
                            req["door_state_min"] = 0.0
                            req["door_state_max"] = 0.0
                        elif action_type in ("put", "place"):
                            # For place actions, microwave needs to be open to place objects inside
                            req["door_state_min"] = 0.90
                            req["door_state_max"] = 1.0
                        elif action_type == "turn_on":
                            req["running_state"] = False  # Starts off, needs to be turned on
                        elif action_type == "turn_off":
                            req["running_state"] = True  # Starts on, needs to be turned off
                        fixture_requirements["microwave"] = req
                    else:
                        # Update state
                        if action_type == "close":
                            fixture_requirements["microwave"]["door_state_min"] = 0.90
                            fixture_requirements["microwave"]["door_state_max"] = 1.0
                        elif action_type == "open":
                            fixture_requirements["microwave"]["door_state_min"] = 0.0
                            fixture_requirements["microwave"]["door_state_max"] = 0.0
                        elif action_type in ("put", "place"):
                            # Ensure microwave is open for place actions
                            fixture_requirements["microwave"]["door_state_min"] = 0.90
                            fixture_requirements["microwave"]["door_state_max"] = 1.0
                        elif action_type == "turn_on":
                            fixture_requirements["microwave"]["running_state"] = False
                        elif action_type == "turn_off":
                            fixture_requirements["microwave"]["running_state"] = True
        
        # Store fixture_requirements in fixture_plan for the executor
        fixture_plan["_fixture_requirements"] = fixture_requirements
        
        # Add fixture entries to fixture_plan for fixtures that are required
        # This allows _register_fixture to properly find and register them
        # The executor will handle finding the actual fixture (e.g., drawer_tabletop_main_group)
        if "cabinet" in fixture_requirements:
            fixture_plan["cabinet"] = {
                "id": "FixtureType.DOOR_HINGE_SINGLE",
            }
        if "drawer" in fixture_requirements:
            # Drawer will be found by executor as "drawer_tabletop_main_group"
            # This matches tabletop_drawer_pnp.py which uses get_fixture("drawer_tabletop_main_group")
            fixture_plan["drawer"] = {
                "id": "FixtureType.DRAWER",
            }
        if "microwave" in fixture_requirements:
            fixture_plan["microwave"] = {
                "id": "FixtureType.MICROWAVE",
            }
        
        # Set init_robot_base_pos based on the primary fixture being manipulated
        # Priority: cabinet > drawer > microwave > counter (default)
        # This matches the behavior of TabletopCabinetDoor, TabletopDrawerDoor, etc.
        if "cabinet" in fixture_requirements:
            fixture_plan["init_robot_base_pos"] = "cabinet"
        elif "drawer" in fixture_requirements:
            fixture_plan["init_robot_base_pos"] = "drawer"
        elif "microwave" in fixture_requirements:
            fixture_plan["init_robot_base_pos"] = "microwave"
        # else: keep default "counter"
        
        return fixture_plan
    
    def _determine_layout_id(self, fixture_plan: Dict[str, Any]) -> int:
        """Determine layout_id based on required fixtures.
        
        Layout reference:
            0: TABLETOP (basic, no special fixtures)
            2: TABLETOP_WITH_MICROWAVE (has microwave on table)
            4: TABLETOP_WITH_DRAWER (has drawer on table)
            5: TABLETOP_WITH_CABINET (has cabinet on table, may also have microwave)
        
        This matches the logic in executor.py DynamicEnvironmentFactory.
        """
        fixture_requirements = fixture_plan.get("_fixture_requirements", {})
        required_fixtures = set(fixture_requirements.keys())
        
        if "cabinet" in required_fixtures:
            # Layout 5 has cabinet (and may also have microwave)
            return 5
        elif "microwave" in required_fixtures:
            # Layout 2 has microwave
            return 2
        elif "drawer" in required_fixtures:
            # Layout 4 has drawer
            return 4
        else:
            # Default layout
            return 0
    
    def _build_initial_state(
        self, 
        parsed_conditions: Dict,
        fixture_requirements: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Build initial state information (door states, running states).
        
        IMPORTANT: Only include states for fixtures that are actually required.
        Use concrete min/max values from fixture_requirements.
        This is stored for reference and task evaluation but is not
        directly passed to the environment constructor.
        
        Args:
            parsed_conditions: Parsed initial conditions
            fixture_requirements: Required fixtures with concrete state values
        """
        if fixture_requirements is None:
            fixture_requirements = {}
        
        door_states = {}
        # Extract concrete door state values from fixture_requirements
        for fixture_name, req in fixture_requirements.items():
            if "door_state_min" in req and "door_state_max" in req:
                door_states[fixture_name] = {
                    "min": req["door_state_min"],
                    "max": req["door_state_max"],
                }
        
        running_states = {}
        # Extract concrete running state values from fixture_requirements
        for fixture_name, req in fixture_requirements.items():
            if "running_state" in req:
                running_states[fixture_name] = req["running_state"]  # Already a boolean
        
        return {
            "door_states": door_states,
            "running_states": running_states,
        }
    
    def _build_language_instruction(
        self,
        parsed_task: Dict,
        parsed_conditions: Dict,
        object_plan: List[Dict[str, Any]],
    ) -> str:
        """Build FULL natural language instruction for the robot.
        
        IMPORTANT: The task instruction must NOT be simplified.
        For conditional tasks, the complete conditional expression is returned.
        Uses PLY-based parser for formal task expression parsing.
        Uses concrete object names from the object_plan.
        """
        from .task_parser import TaskParser, TaskEvaluator
        
        # Build a mapping from abstract names to concrete display names
        name_mapping = {}
        for obj in object_plan:
            name_mapping[obj["name"]] = obj.get("display_name", obj["name"])
            # Also map the category to display_name
            if "category" in obj:
                name_mapping[obj["category"]] = obj.get("display_name", obj["name"])
        
        # TEMPORARY MAPPING: Map fruit-related terms to cup in language instruction
        # Find cup display_name from object_plan (either from actual cup or from fruit->cup mapping)
        cup_display_name = "cup"
        for obj in object_plan:
            if obj.get("category") == "cup":
                cup_display_name = obj.get("display_name", "cup")
                break
            elif obj.get("category") == "fruit":
                # This fruit was mapped to cup, so use its display_name
                cup_display_name = obj.get("display_name", "cup")
                break
        
        # Replace all fruit-related mappings with cup
        keys_to_update = [k for k in name_mapping.keys() if k == "fruit" or (isinstance(k, str) and "fruit" in k.lower())]
        for key in keys_to_update:
            name_mapping[key] = cup_display_name
        
        # Also add direct mapping for "fruit" -> cup display_name
        name_mapping["fruit"] = cup_display_name
        
        # Get task expression from abstract config
        task_expression = parsed_task.get("_original_expression", "")
        
        # Parse the task expression
        parser = TaskParser()
        ast = parser.parse(task_expression)
        
        if ast is None:
            return "complete the task"
        
        # Generate FULL task language (not simplified)
        language = TaskEvaluator.generate_full_task_language(ast, name_mapping)
        
        return language
    
    def _build_task_plan(
        self, 
        parsed_task: Dict, 
        object_plan: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Build task plan from parsed task expression."""
        task_plan = []
        
        # Get all actions (from either conditional or sequence)
        all_actions = []
        if parsed_task["type"] == "conditional":
            all_actions.extend(parsed_task.get("then_actions", []))
            all_actions.extend(parsed_task.get("else_actions", []))
        else:
            all_actions.extend(parsed_task.get("sequence_actions", []))
        
        # Find objects that need manipulation
        object_names = {obj["name"] for obj in object_plan}
        
        for action in all_actions:
            action_type = action.get("action")
            
            if action_type == "pick":
                obj_name = action.get("object", "obj")
                task_plan.append({
                    "action": "pick",
                    "object_ref": obj_name if obj_name in object_names else "obj",
                    "gripper_name": "right",
                    "subtask_term_signal": "grasp_object",
                    "subtask_term_offset_range": (5, 10),
                })
                
            elif action_type == "place":
                obj_name = action.get("object", "obj")
                target = action.get("target", "container")
                
                # Map target to object name
                target_ref = target if target in object_names else "container"
                
                task_plan.append({
                    "action": "place",
                    "object_ref": obj_name if obj_name in object_names else "obj",
                    "target_ref": target_ref,
                    "gripper_name": "right",
                })
                
            elif action_type in ("open", "close"):
                task_plan.append({
                    "action": action_type,
                    "target_ref": action.get("target"),
                    "gripper_name": "right",
                })
                
            elif action_type in ("turn_on", "turn_off"):
                task_plan.append({
                    "action": action_type,
                    "target_ref": action.get("target"),
                })
        
        # If no explicit task plan, create a default pick-and-place task
        if not task_plan and object_plan:
            graspable_objs = [obj for obj in object_plan if obj.get("graspable")]
            containers = [obj for obj in object_plan if not obj.get("graspable")]
            
            if graspable_objs and containers:
                task_plan = [
                    {
                        "action": "pick",
                        "object_ref": graspable_objs[0]["name"],
                        "gripper_name": "right",
                        "subtask_term_signal": "grasp_object",
                        "subtask_term_offset_range": (5, 10),
                    },
                    {
                        "action": "place",
                        "object_ref": graspable_objs[0]["name"],
                        "target_ref": containers[0]["name"],
                        "gripper_name": "right",
                    },
                ]
        
        return task_plan
    
    def _build_success_plan(
        self, 
        parsed_task: Dict, 
        object_plan: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Build success plan from parsed task."""
        success_plan = []
        
        # Get all actions
        all_actions = []
        if parsed_task["type"] == "conditional":
            all_actions.extend(parsed_task.get("then_actions", []))
            all_actions.extend(parsed_task.get("else_actions", []))
        else:
            all_actions.extend(parsed_task.get("sequence_actions", []))
        
        object_names = {obj["name"] for obj in object_plan}
        
        for action in all_actions:
            action_type = action.get("action")
            
            if action_type == "place":
                obj_name = action.get("object", "obj")
                target = action.get("target", "container")
                
                success_plan.append({
                    "type": "obj_in_receptacle",
                    "params": {
                        "obj_name": obj_name if obj_name in object_names else "obj",
                        "receptacle_name": target,  # Use target directly - it can be an object name or fixture name
                    },
                })
                
            elif action_type == "open":
                success_plan.append({
                    "type": "door_open",
                    "params": {
                        "fixture_name": action.get("target"),
                    },
                })
                
            elif action_type == "close":
                success_plan.append({
                    "type": "door_closed",
                    "params": {
                        "fixture_name": action.get("target"),
                    },
                })
                
            elif action_type == "turn_on":
                success_plan.append({
                    "type": "fixture_running",
                    "params": {
                        "fixture_name": action.get("target"),
                        "running": True,
                    },
                })
                
            elif action_type == "turn_off":
                success_plan.append({
                    "type": "fixture_running",
                    "params": {
                        "fixture_name": action.get("target"),
                        "running": False,
                    },
                })
        
        # Add gripper_far condition for pick-place tasks
        for action in all_actions:
            if action.get("action") == "place":
                obj_name = action.get("object", "obj")
                success_plan.append({
                    "type": "gripper_far",
                    "params": {
                        "obj_name": obj_name if obj_name in object_names else "obj",
                    },
                })
                break
        
        return success_plan
    
    def to_python_code(
        self,
        config_number: int,
        seed: Optional[int] = None,
        class_name: str = "InstantiatedPickPlace",
    ) -> str:
        """
        Generate Python code for a concrete configuration.
        
        This generates code similar to run_give.py format.
        """
        concrete = self.instantiate(config_number, seed)
        
        code_lines = [
            '"""',
            f'Auto-generated configuration for abstract config #{config_number}',
            f'Task expression: {concrete["task_expression"]}',
            '"""',
            '',
            'from pathlib import Path',
            'from typing import Dict, List, Any',
            'from copy import deepcopy',
            '',
            'import numpy as np',
            'import robocasa',
            'from robocasa.environments.tabletop.tabletop_give import TabletopGive',
            'from robocasa.models.fixtures import FixtureType',
            '',
            '# ---------------------------------------------------------------------------',
            '# Configuration Parameters',
            '# ---------------------------------------------------------------------------',
            '',
            'ROBOTS_NAME = "GR1FixedLowerBody"',
            'LAYOUT_ID = 0',
            'HANDEDNESS = "right"',
            f'ENFORCE_DETERMINISM = {concrete["enforce_determinism"]}',
            '',
            '# Object plan',
            f'OBJECT_PLAN: List[Dict] = {self._format_python_value(concrete["object_plan"])}',
            '',
            '# Task plan',
            f'TASK_PLAN: List[Dict] = {self._format_python_value(concrete["task_plan"])}',
            '',
            '# Success plan',
            f'SUCCESS_PLAN: List[Dict] = {self._format_python_value(concrete["success_plan"])}',
            '',
            '# Fixture plan',
            f'FIXTURE_PLAN: Dict = {self._format_fixture_plan(concrete["fixture_plan"])}',
            '',
            '',
            f'class {class_name}(TabletopGive):',
            '    """Auto-generated deterministic environment."""',
            '',
            '    def __init__(self, *args, **kwargs):',
            '        super().__init__(',
            '            object_plan=OBJECT_PLAN,',
            '            task_plan=TASK_PLAN,',
            '            success_plan=SUCCESS_PLAN,',
            '            fixture_plan=FIXTURE_PLAN,',
            '            enforce_determinism=ENFORCE_DETERMINISM,',
            '            handedness=HANDEDNESS,',
            '            layout_id=LAYOUT_ID,',
            '            *args,',
            '            **kwargs,',
            '        )',
        ]
        
        return '\n'.join(code_lines)
    
    def _format_python_value(self, value: Any, indent: int = 0) -> str:
        """Format a Python value for code generation."""
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = []
            for k, v in value.items():
                formatted_v = self._format_python_value(v, indent + 4)
                items.append(f'"{k}": {formatted_v}')
            inner = ", ".join(items)
            return f"{{{inner}}}"
        elif isinstance(value, list):
            if not value:
                return "[]"
            items = [self._format_python_value(v, indent + 4) for v in value]
            return f"[{', '.join(items)}]" if len(str(items)) < 60 else f"[\n{'    ' * (indent // 4 + 1)}" + f",\n{'    ' * (indent // 4 + 1)}".join(items) + f"\n{'    ' * (indent // 4)}]"
        elif isinstance(value, tuple):
            return f"({', '.join(str(v) for v in value)})"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "True" if value else "False"
        else:
            return str(value)
    
    def _format_fixture_plan(self, fixture_plan: Dict) -> str:
        """Format fixture plan with FixtureType references."""
        lines = ["{"]
        for key, value in fixture_plan.items():
            if isinstance(value, dict):
                inner_items = []
                for k, v in value.items():
                    if k == "id" and isinstance(v, str) and v.startswith("FixtureType."):
                        inner_items.append(f'"{k}": {v}')
                    elif isinstance(v, tuple):
                        inner_items.append(f'"{k}": {v}')
                    elif isinstance(v, str):
                        inner_items.append(f'"{k}": "{v}"')
                    else:
                        inner_items.append(f'"{k}": {v}')
                lines.append(f'    "{key}": {{{", ".join(inner_items)}}},')
            else:
                if isinstance(value, str):
                    lines.append(f'    "{key}": "{value}",')
                else:
                    lines.append(f'    "{key}": {value},')
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Instantiate concrete configurations from abstract configurations"
    )
    parser.add_argument(
        "--config-number", "-n",
        type=int,
        default=None,
        help="Configuration number to instantiate (1-based)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (JSON format, or .py for Python code)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "python"],
        default="json",
        help="Output format (json or python)"
    )
    parser.add_argument(
        "--config-path", "-c",
        type=str,
        default=None,
        help="Path to abstract configuration JSON file"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available configurations"
    )
    
    args = parser.parse_args()
    
    instantiator = ConfigurationInstantiator(config_path=args.config_path)
    
    if args.list:
        print(f"Total configurations: {instantiator.get_total_configurations()}")
        print("\nFirst 10 configurations:")
        for i in range(1, min(11, instantiator.get_total_configurations() + 1)):
            config = instantiator.get_abstract_config(i)
            print(f"  {i}: {config.get('task_expression', 'N/A')}")
        return
    
    if args.config_number is None:
        parser.error("--config-number is required when not using --list")
    
    if args.format == "python":
        output = instantiator.to_python_code(
            args.config_number,
            seed=args.seed,
            class_name=f"Config{args.config_number}PickPlace",
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Python code written to {args.output}")
        else:
            print(output)
    else:
        concrete_config = instantiator.instantiate(
            args.config_number,
            seed=args.seed,
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(concrete_config, f, indent=2, default=str)
            print(f"Configuration written to {args.output}")
        else:
            print(json.dumps(concrete_config, indent=2, default=str))


if __name__ == "__main__":
    main()

