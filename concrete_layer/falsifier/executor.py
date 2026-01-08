"""
Configuration Executor Module

This module executes simulations using JSON configuration files generated
by the instantiate.py module.

Usage:
    # Execute a configuration file
    python -m falsifier.executor --config config.json
    
    # Execute with custom parameters
    python -m falsifier.executor --config config.json --episodes 5 --host localhost --port 5555
    
    # Execute directly from a configuration number (instantiate + execute)
    python -m falsifier.executor --config-number 1 --seed 42
    
    # As a library
    from falsifier.executor import ConfigurationExecutor
    executor = ConfigurationExecutor()
    results = executor.execute_config(config_dict)

Requirements:
    - The inference server must be running (see scripts/simulation_service.py)
    - A display environment is required for rendering (X11/Wayland)
    - For headless environments, use xvfb-run:
        xvfb-run -a python -m falsifier.executor --config-number 1
"""

import json
import argparse
import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Tuple
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

# Add parent directory to path to allow imports when running script directly
# This ensures 'falsifier' and 'gr00t' modules can be imported
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Default seed for deterministic behavior
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Execution Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """Configuration for simulation execution."""
    
    # Server settings
    host: str = "localhost"
    port: int = 5555
    
    # Simulation settings
    n_episodes: int = 1
    n_envs: int = 1
    n_action_steps: int = 2
    max_episode_steps: int = 600
    
    # Output settings
    video_dir: str = "./videos"
    save_video: bool = True
    record_path: Optional[str] = None
    replay_recording: Optional[str] = None
    record_signals: bool = True  # Whether to record signals in env_infos
    
    # Robot settings
    robots_name: str = "GR1FixedLowerBody"
    handedness: str = "right"
    layout_id: int = 0
    
    # Determinism settings
    seed: int = DEFAULT_SEED  # Random seed for reproducibility


# ---------------------------------------------------------------------------
# Dynamic Environment Factory
# ---------------------------------------------------------------------------

class DynamicEnvironmentFactory:
    """
    Factory for creating dynamic environment classes from configuration.
    
    This allows creating TabletopGive subclasses at runtime from JSON configs.
    """
    
    # Cache for created environment classes
    _class_cache: Dict[str, Type] = {}
    
    @classmethod
    def create_environment_class(
        cls,
        config: Dict[str, Any],
        class_name: str = "DynamicPickPlace",
    ) -> Type:
        """
        Create a dynamic environment class from configuration.
        
        Args:
            config: Configuration dictionary with object_plan, task_plan, etc.
            class_name: Name for the generated class
            
        Returns:
            A new class that can be used as a TabletopGive environment
        """
        # Special-case cabinet-only open / close tasks to mirror the dedicated
        # TabletopCabinetDoor settings (layout 5, init pose, language).
        cabinet_behavior = cls._get_cabinet_door_behavior(config)
        if cabinet_behavior:
            return cls._build_cabinet_env_class(
                config=config,
                class_name=class_name,
                behavior=cabinet_behavior,
            )

        # Import here to avoid circular imports and allow lazy loading
        from robocasa.environments.tabletop.tabletop_give import TabletopGive
        from robocasa.models.fixtures import FixtureType
        
        # Get raw fixture plan and extract fixture requirements
        raw_fixture_plan = deepcopy(config.get("fixture_plan", {}))
        fixture_requirements = raw_fixture_plan.pop("_fixture_requirements", {})
        
        # Build clean fixture_plan (remove initial_state, running_state)
        clean_fixture_plan = {}
        for name, cfg in raw_fixture_plan.items():
            if isinstance(cfg, dict):
                clean_cfg = {k: v for k, v in cfg.items() 
                            if k not in ("initial_state", "running_state", "_fixture_requirements")}
                clean_fixture_plan[name] = clean_cfg
            else:
                clean_fixture_plan[name] = cfg
        
        # Add required fixtures to fixture_plan so they get registered by TabletopGive
        # Use string IDs first, then convert to FixtureType enums
        fixture_type_map = {
            "cabinet": "DOOR_HINGE_SINGLE",
            "drawer": "DRAWER",
            "microwave": "MICROWAVE",
        }
        
        for name, req in fixture_requirements.items():
            if name not in clean_fixture_plan:
                # Add fixture to fixture_plan with string ID (will be converted later)
                fixture_type_name = req.get("type", fixture_type_map.get(name, name.upper()))
                clean_fixture_plan[name] = {"id": f"FixtureType.{fixture_type_name}"}
        
        # Extract initial states for _reset_internal
        # Use concrete door state values (min/max) instead of "open"/"closed" strings
        fixture_initial_states = {}
        print(f"[DEBUG create_env] fixture_requirements = {fixture_requirements}")
        for name, req in fixture_requirements.items():
            if "door_state_min" in req and "door_state_max" in req:
                # Store concrete door state values
                fixture_initial_states[name] = {
                    "min": req["door_state_min"],
                    "max": req["door_state_max"],
                }
            if "running_state" in req:
                # Store concrete boolean value
                fixture_initial_states[name + "_running"] = req["running_state"]
        print(f"[DEBUG create_env] fixture_initial_states = {fixture_initial_states}")
        
        # Convert fixture plan IDs from strings to actual FixtureType values
        fixture_plan = cls._convert_fixture_plan(clean_fixture_plan)
        
        # Get other configuration values
        object_plan = config.get("object_plan", [])
        task_plan = config.get("task_plan", [])
        success_plan = config.get("success_plan", [])
        enforce_determinism = config.get("enforce_determinism", True)
        lang_instruction = config.get("lang", "complete the task")
        
        # Convert any list positions to tuples (JSON doesn't support tuples)
        object_plan = cls._convert_positions(object_plan)
        task_plan = cls._convert_positions(task_plan)
        success_plan = cls._convert_positions(success_plan)
        fixture_plan = cls._convert_positions(fixture_plan)
        
        # Create a new class dynamically with language instruction
        # (fixture_initial_states was extracted earlier)
        class DynamicEnv(TabletopGive):
            """Dynamically generated environment from configuration."""
            
            # Store language instruction at class level
            _dynamic_lang = lang_instruction
            _fixture_initial_states = fixture_initial_states
            _fixture_requirements = fixture_requirements  # Store for fixture creation
            
            def __init__(self, robots=None, *args, **kwargs):
                # Remove enable_render if present (Tabletop doesn't accept it)
                kwargs.pop("enable_render", None)
                
                # Force deterministic settings to eliminate all randomness
                # 1. Set seed for reproducibility (use DEFAULT_SEED from module)
                kwargs.setdefault("seed", DEFAULT_SEED)
                # 2. Disable robot initialization noise (None -> magnitude 0.0)
                kwargs.setdefault("initialization_noise", None)
                
                # Handle robots parameter - Tabletop.__init__ requires it as first positional arg
                # If robots is in kwargs, extract it
                if robots is None:
                    robots = kwargs.pop("robots", "GR1ArmsAndWaistFourierHands")
                
                # Auto-select layout based on required fixtures
                # Layout reference:
                #   0: TABLETOP (basic, no special fixtures)
                #   2: TABLETOP_WITH_MICROWAVE (has microwave on table)
                #   4: TABLETOP_WITH_DRAWER (has drawer on table)
                #   5: TABLETOP_WITH_CABINET (modified to have BOTH cabinet and microwave on table)
                layout_id = kwargs.pop("layout_id", None)
                if layout_id is None:
                    # Determine layout based on required fixtures
                    required_fixtures = set(self._fixture_requirements.keys())
                    
                    if "cabinet" in required_fixtures:
                        # Layout 5 has cabinet (and microwave now)
                        layout_id = 5
                        print(f"[INFO] Auto-selected layout {layout_id} for cabinet (and microwave)")
                    elif "microwave" in required_fixtures:
                        # Layout 2 has microwave
                        layout_id = 2
                        print(f"[INFO] Auto-selected layout {layout_id} for microwave")
                    elif "drawer" in required_fixtures:
                        layout_id = 4
                        print(f"[INFO] Auto-selected layout {layout_id} for drawer")
                    else:
                        layout_id = 0
                        print(f"[INFO] Using default layout {layout_id}")
                
                # Pass robots to kwargs so TabletopGive can handle it
                kwargs["robots"] = robots
                
                # Ensure camera_names is set correctly for GR1 robot
                # GR1 doesn't have "agentview", so we must specify valid cameras
                if "camera_names" not in kwargs:
                    kwargs["camera_names"] = [
                        "egoview",
                        "robot0_eye_in_left_hand",
                        "robot0_eye_in_right_hand",
                    ]
                
                print(f"[DEBUG] About to call super().__init__ with layout_id={layout_id}")
                print(f"[DEBUG] robots={robots}, object_plan length={len(object_plan)}")
                print(f"[DEBUG] kwargs before cleanup: style_ids={kwargs.get('style_ids')}, generative_textures={kwargs.get('generative_textures')}")
                
                # Remove these from kwargs to ensure our explicit values are used
                kwargs.pop("style_ids", None)
                kwargs.pop("generative_textures", None)
                
                try:
                    super().__init__(
                        object_plan=object_plan,
                        task_plan=task_plan,
                        success_plan=success_plan,
                        fixture_plan=fixture_plan,
                        enforce_determinism=enforce_determinism,
                        handedness=kwargs.pop("handedness", "right"),
                        layout_id=layout_id,
                        style_ids=[0],  # Fix style to 0 for deterministic appearance
                        generative_textures=False,  # Disable random textures for deterministic appearance
                        use_distractors=False,  # Disable distractor fixtures (including paper_towel)
                        distractor_config=None,  # Explicitly set to None to exclude paper_towel
                        *args,
                        **kwargs,
                    )
                    print(f"[DEBUG] super().__init__ completed successfully")
                except Exception as e:
                    print(f"[ERROR] super().__init__ failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            def _build_placement_cfg(self, entry: Dict[str, Any]) -> Dict[str, Any]:
                """
                Override to support placement relative to objects.
                
                If placement contains "object_ref", the object will be placed relative
                to that object instead of a fixture. The "pos" will be relative to the
                referenced object's position.
                """
                from copy import deepcopy
                
                if "placement" not in entry:
                    raise ValueError(f"Object plan for '{entry['name']}' missing placement.")

                placement = deepcopy(entry["placement"])
                
                # Check if placement is relative to an object
                object_ref = placement.get("object_ref", None)
                if object_ref is not None:
                    # Placement relative to another object
                    # We need to set up sample_args to reference the object
                    # First, ensure the referenced object exists in object_plan
                    ref_obj_exists = any(
                        obj_entry.get("name") == object_ref 
                        for obj_entry in self.object_plan
                    )
                    if not ref_obj_exists:
                        raise ValueError(
                            f"Object '{entry['name']}' references non-existent object '{object_ref}' "
                            "in placement.object_ref. Make sure the referenced object is defined "
                            "earlier in object_plan."
                        )
                    
                    # Set up sample_args to reference the object
                    # The reference will be resolved during sampling
                    if "sample_args" not in placement:
                        placement["sample_args"] = {}
                    placement["sample_args"]["reference"] = object_ref
                    placement["sample_args"]["on_top"] = placement.get("on_top", True)
                    
                    # For object-relative placement, we still need a fixture for the base reference
                    # But the actual position will be relative to the object
                    # Use a minimal fixture reference (will be overridden by sample_args)
                    fixture_ref = placement.get("fixture", "counter")
                    placement["fixture"] = self._resolve_fixture_reference(fixture_ref)
                    
                    # Set a minimal size since position will be relative to object
                    placement["size"] = (self.REGION_EPSILON, self.REGION_EPSILON)
                else:
                    # Standard fixture-relative placement
                    fixture_ref = placement.get("fixture", "counter")
                    placement["fixture"] = self._resolve_fixture_reference(fixture_ref)

                # Resolve deterministic XY location (exact position, not a region)
                pos = placement.get("pos") or placement.get("xy")
                handed_pos = placement.get("handed_pos")
                if pos is None and handed_pos:
                    pos = handed_pos[self.handedness]
                if callable(pos):
                    pos = pos(self.handedness)
                if pos is None:
                    raise ValueError(f"Placement for '{entry['name']}' must include 'pos' or 'xy'.")
                placement["pos"] = pos

                # For deterministic mode, use exact position (size should be minimal/zero)
                if self.enforce_determinism:
                    # Force size to be minimal for exact positioning
                    if "size" not in placement:
                        placement["size"] = (self.REGION_EPSILON, self.REGION_EPSILON)
                    # Ensure rotation is exact (not a range)
                    rotation = placement.get("rotation")
                    yaw = placement.get("yaw")
                    if rotation is None and yaw is not None:
                        # Single yaw value -> exact rotation
                        rotation = (yaw, yaw)
                    elif isinstance(rotation, (int, float)):
                        # Single rotation value -> exact rotation
                        rotation = (rotation, rotation)
                    elif isinstance(rotation, (list, tuple)) and len(rotation) == 2:
                        # Range -> check if it's actually a range or exact value
                        if abs(rotation[0] - rotation[1]) > 1e-6:
                            # It's a range, use the first value as exact
                            rotation = (rotation[0], rotation[0])
                    else:
                        rotation = (0.0, 0.0)
                    placement["rotation"] = rotation
                else:
                    # Non-deterministic mode: allow regions
                    default_size = entry.get("size", (self.REGION_EPSILON, self.REGION_EPSILON))
                    if "size" not in placement:
                        placement["size"] = default_size
                    
                    # Normalize rotation specification (can be a range)
                    rotation = placement.get("rotation")
                    yaw = placement.get("yaw")
                    if rotation is None and yaw is not None:
                        rotation = (yaw, yaw)
                    elif isinstance(rotation, (int, float)):
                        rotation = (rotation, rotation)
                    placement["rotation"] = rotation or (0.0, 0.0)

                placement.setdefault("ensure_object_boundary_in_range", False)
                placement.setdefault("ensure_object_in_ref_region", False)
                
                return placement
            
            def _register_fixture(self, name: str, cfg: Any):
                """Override to handle fixtures that may not exist in the layout.
                
                Note: FixtureType.DOOR_HINGE_SINGLE matches both SingleCabinet AND Microwave!
                So we need to be careful to get the right fixture.
                
                Strategy:
                - For microwave: use FixtureType.MICROWAVE (only matches Microwave)
                - For cabinet: use FixtureType.CABINET or find by name pattern
                - For drawer: use FixtureType.DRAWER
                """
                from robocasa.models.fixtures import FixtureType, Microwave, SingleCabinet, HingeCabinet, Cabinet, Drawer
                
                # Handle required fixtures that may not exist in the layout
                if name in self._fixture_requirements:
                    req = self._fixture_requirements[name]
                    fixture_type_name = req.get("type", name.upper())
                    
                    try:
                        if name == "microwave" or fixture_type_name == "MICROWAVE":
                            # Use FixtureType.MICROWAVE which only matches Microwave
                            fixture = self.get_fixture(FixtureType.MICROWAVE)
                            setattr(self, name, fixture)
                            self._fixture_aliases[name] = fixture
                            print(f"[DEBUG] Successfully registered fixture '{name}': {type(fixture).__name__}")
                            return fixture
                            
                        elif name == "cabinet" or fixture_type_name == "DOOR_HINGE_SINGLE":
                            # Get the tabletop cabinet (not the ones in cabinet stacks)
                            # The tabletop cabinet is named "cabinet_tabletop_main_group"
                            tabletop_cabinet_name = "cabinet_tabletop_main_group"
                            found_cabinet = None
                            
                            if tabletop_cabinet_name in self.fixtures:
                                found_cabinet = self.fixtures[tabletop_cabinet_name]
                            else:
                                # Fallback: find a cabinet with "tabletop" in its name
                                for fxtr_name, fxtr in self.fixtures.items():
                                    if isinstance(fxtr, (SingleCabinet, HingeCabinet)) and not isinstance(fxtr, Microwave):
                                        if "tabletop" in fxtr_name:
                                            found_cabinet = fxtr
                                            break
                                
                                # If no tabletop cabinet, try any SingleCabinet/HingeCabinet (not microwave)
                                if found_cabinet is None:
                                    for fxtr_name, fxtr in self.fixtures.items():
                                        if isinstance(fxtr, (SingleCabinet, HingeCabinet)) and not isinstance(fxtr, Microwave):
                                            found_cabinet = fxtr
                                            break
                                
                                # Last resort: try FixtureType.CABINET
                                if found_cabinet is None:
                                    try:
                                        found_cabinet = self.get_fixture(FixtureType.CABINET)
                                    except AssertionError:
                                        pass
                            
                            if found_cabinet is not None:
                                setattr(self, name, found_cabinet)
                                self._fixture_aliases[name] = found_cabinet
                                print(f"[DEBUG] Successfully registered fixture '{name}': {type(found_cabinet).__name__} (actual name: {found_cabinet.name})")
                                return found_cabinet
                            else:
                                print(f"[WARNING] Fixture '{name}' (cabinet) not found in layout {self.layout_id}")
                                return None
                                
                        elif name == "drawer" or fixture_type_name == "DRAWER":
                            # Get the tabletop drawer (not the ones in cabinet stacks)
                            # The tabletop drawer is named "drawer_tabletop_main_group"
                            tabletop_drawer_name = "drawer_tabletop_main_group"
                            if tabletop_drawer_name in self.fixtures:
                                fixture = self.fixtures[tabletop_drawer_name]
                            else:
                                # Fallback: find a drawer with "tabletop" in its name
                                fixture = None
                                for fxtr_name, fxtr in self.fixtures.items():
                                    if isinstance(fxtr, Drawer) and "tabletop" in fxtr_name:
                                        fixture = fxtr
                                        break
                                if fixture is None:
                                    # Last resort: use FixtureType.DRAWER
                                    fixture = self.get_fixture(FixtureType.DRAWER)
                            setattr(self, name, fixture)
                            self._fixture_aliases[name] = fixture
                            print(f"[DEBUG] Successfully registered fixture '{name}': {type(fixture).__name__} (actual name: {fixture.name})")
                            return fixture
                            
                        else:
                            # Unknown fixture type, try parent implementation
                            if isinstance(cfg, dict):
                                ref = self.register_fixture_ref(name, cfg)
                            else:
                                ref = cfg
                            setattr(self, name, ref)
                            self._fixture_aliases[name] = ref
                            return ref
                            
                    except AssertionError:
                        print(f"[WARNING] Fixture '{name}' not found in layout {self.layout_id}")
                        print(f"  Available fixtures: {list(self.fixtures.keys())}")
                        return None
                    except Exception as e:
                        print(f"[WARNING] Error registering fixture '{name}': {e}")
                        import traceback
                        traceback.print_exc()
                        return None
                else:
                    # For non-required fixtures, use parent implementation
                    return super()._register_fixture(name, cfg)
            
            def _setup_table_references(self):
                """Override to ensure required fixtures are registered.
                
                Fixtures should already be in fixture_plan and registered by parent class.
                This override handles fixtures that may not exist in the layout.
                """
                # Call parent's _setup_table_references, which will call our overridden _register_fixture
                super()._setup_table_references()
                
                # Verify that required fixtures were registered
                for fixture_name in self._fixture_requirements.keys():
                    if hasattr(self, fixture_name):
                        fixture = getattr(self, fixture_name)
                        print(f"[DEBUG] Fixture '{fixture_name}' successfully registered: {type(fixture).__name__}")
                    else:
                        print(f"[WARNING] Fixture '{fixture_name}' not available in layout {self.layout_id}")
                        # Try to find it manually as a fallback
                        from robocasa.models.fixtures import FixtureType
                        fixture_types = {
                            "cabinet": FixtureType.DOOR_HINGE_SINGLE,
                            "drawer": FixtureType.DRAWER,
                            "microwave": FixtureType.MICROWAVE,
                        }
                        if fixture_name in fixture_types:
                            try:
                                fixture = self.get_fixture(fixture_types[fixture_name])
                                setattr(self, fixture_name, fixture)
                                self._fixture_aliases[fixture_name] = fixture
                                print(f"[DEBUG] Manually registered fixture '{fixture_name}'")
                            except Exception as e:
                                print(f"[WARNING] Could not manually register fixture '{fixture_name}': {e}")
            
            def _reset_internal(self):
                """Override to set initial fixture states using concrete values."""
                super()._reset_internal()
                
                # Debug: print fixture initial states
                print(f"[DEBUG _reset_internal] _fixture_initial_states = {self._fixture_initial_states}")
                
                # Set door states for fixtures using concrete min/max values
                for name, state in self._fixture_initial_states.items():
                    if name.endswith("_running"):
                        # Handle running state for microwave (concrete boolean)
                        fixture_name = name.replace("_running", "")
                        if hasattr(self, fixture_name):
                            fixture = getattr(self, fixture_name)
                            if hasattr(fixture, "_turned_on"):
                                # state is already a boolean value
                                fixture._turned_on = bool(state)
                    else:
                        # Handle door state using concrete min/max values
                        fixture = None
                        if hasattr(self, name):
                            fixture = getattr(self, name)
                        else:
                            # Fallback: try to get fixture by name or type
                            # Import here to avoid circular imports
                            from robocasa.models.fixtures import (
                                FixtureType, SingleCabinet, HingeCabinet, Microwave, Drawer as DrawerFixture
                            )
                            try:
                                if name == "drawer":
                                    # First try to get the tabletop drawer by name
                                    # This is the drawer placed ON the table, not in a cabinet stack
                                    tabletop_drawer_name = "drawer_tabletop_main_group"
                                    if tabletop_drawer_name in self.fixtures:
                                        fixture = self.fixtures[tabletop_drawer_name]
                                    else:
                                        # Fallback: find a drawer that's a table accessory (not in a stack)
                                        for fxtr_name, fxtr in self.fixtures.items():
                                            if isinstance(fxtr, DrawerFixture) and "tabletop" in fxtr_name:
                                                fixture = fxtr
                                                break
                                        if fixture is None:
                                            # Last resort: use FixtureType.DRAWER
                                            fixture = self.get_fixture(FixtureType.DRAWER)
                                elif name == "cabinet":
                                    # First try to get the tabletop cabinet by name
                                    # This is the cabinet placed ON the table
                                    tabletop_cabinet_name = "cabinet_tabletop_main_group"
                                    if tabletop_cabinet_name in self.fixtures:
                                        fixture = self.fixtures[tabletop_cabinet_name]
                                    else:
                                        # Fallback: find a cabinet with "tabletop" in its name
                                        for fxtr_name, fxtr in self.fixtures.items():
                                            if isinstance(fxtr, (SingleCabinet, HingeCabinet)) and not isinstance(fxtr, Microwave):
                                                if "tabletop" in fxtr_name:
                                                    fixture = fxtr
                                                    break
                                        # If no tabletop cabinet, try any cabinet (not microwave)
                                        if fixture is None:
                                            for fxtr_name, fxtr in self.fixtures.items():
                                                if isinstance(fxtr, (SingleCabinet, HingeCabinet)) and not isinstance(fxtr, Microwave):
                                                    fixture = fxtr
                                                    break
                                        if fixture is None:
                                            fixture = self.get_fixture(FixtureType.DOOR_HINGE_SINGLE)
                                elif name == "microwave":
                                    fixture = self.get_fixture(FixtureType.MICROWAVE)
                            except (AssertionError, KeyError, AttributeError):
                                pass
                        
                        if fixture is not None and hasattr(fixture, "set_door_state"):
                            if isinstance(state, dict) and "min" in state and "max" in state:
                                # Use concrete min/max values
                                fixture.set_door_state(
                                    min=state["min"],
                                    max=state["max"],
                                    env=self,
                                    rng=self.rng
                                )
                                print(f"[DEBUG] Set {name} door state to min={state['min']}, max={state['max']}, fixture={fixture.name}")
                            else:
                                # Fallback for old format (should not happen)
                                if state == "open":
                                    fixture.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
                                elif state == "closed":
                                    fixture.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
                        else:
                            print(f"[WARNING] Could not find fixture '{name}' to set door state, hasattr={hasattr(self, name)}")
                
                # Forward the simulation to apply joint position changes
                self.sim.forward()
            
            def get_ep_meta(self):
                """Override to provide the configured language instruction."""
                ep_meta = super().get_ep_meta()
                # Override lang with the configured instruction
                ep_meta["lang"] = self._dynamic_lang
                return ep_meta
            
            def _load_model(self):
                """
                Override to reset rng state before loading model.
                This ensures deterministic behavior even if previous placement attempts failed.
                
                The problem: self.rng state accumulates changes through multiple calls during
                _load_model (layout choice, fixture placement, object placement). If any step
                fails and retries, the rng state changes, affecting all subsequent random calls.
                
                Solution: Reset rng to fixed seed at the start of each _load_model call.
                """
                import numpy as np
                # Reset rng to ensure deterministic behavior
                self.rng = np.random.default_rng(DEFAULT_SEED)
                super()._load_model()
        
        # Set the class name
        DynamicEnv.__name__ = class_name
        DynamicEnv.__qualname__ = class_name
        
        return DynamicEnv

    @classmethod
    def _get_cabinet_door_behavior(cls, config: Dict[str, Any]) -> Optional[str]:
        """
        Detect simple cabinet-door tasks that should use TabletopCabinetDoor settings.
        Returns "open" / "close" or None if not applicable.
        """
        fixture_reqs = config.get("fixture_plan", {}).get("_fixture_requirements", {})
        if set(fixture_reqs.keys()) != {"cabinet"}:
            return None
        
        # Only handle tasks that don't spawn extra objects and only manipulate the cabinet
        if config.get("object_plan"):
            return None
        
        actions = config.get("task_plan", [])
        if not actions:
            return None
        
        action_types = {a.get("action") for a in actions}
        targets = {a.get("target_ref") for a in actions}
        if not action_types.issubset({"open", "close"}):
            return None
        if not targets.issubset({"cabinet", None}):
            return None
        
        if "close" in action_types:
            return "close"
        if "open" in action_types:
            return "open"
        return None
    
    @classmethod
    def _build_cabinet_env_class(
        cls,
        config: Dict[str, Any],
        class_name: str,
        behavior: str,
    ) -> Type:
        """Build a TabletopCabinetDoor-based env that mirrors its settings."""
        from robocasa.environments.tabletop.tabletop_cabinet_door import (
            TabletopCabinetDoor,
        )
        
        lang_instruction = config.get("lang", f"{behavior} the cabinet door")
        
        class CabinetDoorEnv(TabletopCabinetDoor):
            _dynamic_lang = lang_instruction
            
            def __init__(self, robots=None, *args, **kwargs):
                kwargs.pop("enable_render", None)
                kwargs.setdefault("seed", DEFAULT_SEED)
                kwargs.setdefault("initialization_noise", None)
                
                if robots is None:
                    robots = kwargs.pop("robots", "GR1ArmsAndWaistFourierHands")
                
                # Force layout 5 to match TabletopCabinetDoor.VALID_LAYOUTS
                kwargs.setdefault("layout_ids", [5])
                kwargs["robots"] = robots
                
                if "camera_names" not in kwargs:
                    kwargs["camera_names"] = [
                        "egoview",
                        "robot0_eye_in_left_hand",
                        "robot0_eye_in_right_hand",
                    ]
                
                # Disable distractor fixtures (including paper_towel)
                kwargs.setdefault("use_distractors", False)
                kwargs.setdefault("distractor_config", None)
                
                # Fix style to 0 for deterministic appearance
                kwargs["style_ids"] = [0]
                # Disable random textures for deterministic appearance
                kwargs["generative_textures"] = False
                
                print(f"[DEBUG CabinetDoorEnv] Calling super().__init__ with style_ids={kwargs.get('style_ids')}, generative_textures={kwargs.get('generative_textures')}")
                
                super().__init__(behavior=behavior, *args, **kwargs)
            
            def get_ep_meta(self):
                ep_meta = super().get_ep_meta()
                ep_meta["lang"] = self._dynamic_lang
                return ep_meta
        
        CabinetDoorEnv.__name__ = class_name
        CabinetDoorEnv.__qualname__ = class_name
        return CabinetDoorEnv
    
    @classmethod
    def _convert_fixture_plan(cls, fixture_plan: Dict) -> Dict:
        """Convert fixture plan string IDs to FixtureType enums."""
        from robocasa.models.fixtures import FixtureType
        
        converted = {}
        for key, value in fixture_plan.items():
            if isinstance(value, dict):
                converted_value = deepcopy(value)
                # Convert id to FixtureType enum
                if "id" in converted_value:
                    id_str = converted_value["id"]
                    if isinstance(id_str, str) and id_str.startswith("FixtureType."):
                        fixture_name = id_str.replace("FixtureType.", "")
                        try:
                            converted_value["id"] = getattr(FixtureType, fixture_name)
                        except AttributeError:
                            # Keep as string if not found
                            pass
                # Convert ref to FixtureType enum if present
                if "ref" in converted_value:
                    ref_str = converted_value["ref"]
                    if isinstance(ref_str, str) and ref_str.startswith("FixtureType."):
                        fixture_name = ref_str.replace("FixtureType.", "")
                        try:
                            converted_value["ref"] = getattr(FixtureType, fixture_name)
                        except AttributeError:
                            pass
                converted[key] = converted_value
            else:
                converted[key] = value
        
        return converted
    
    @classmethod
    def _convert_positions(cls, data: Any) -> Any:
        """Recursively convert list positions to tuples."""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in ("pos", "size", "rotation", "subtask_term_offset_range"):
                    # These should be tuples
                    if isinstance(value, list):
                        result[key] = tuple(value)
                    else:
                        result[key] = value
                else:
                    result[key] = cls._convert_positions(value)
            return result
        elif isinstance(data, list):
            return [cls._convert_positions(item) for item in data]
        else:
            return data


# ---------------------------------------------------------------------------
# Configuration Executor
# ---------------------------------------------------------------------------

class ConfigurationExecutor:
    """
    Executor for running simulations with generated configurations.
    
    This class handles:
    1. Loading configuration from JSON files
    2. Creating dynamic environment classes
    3. Connecting to the simulation server
    4. Running simulations and collecting results
    """
    
    def __init__(self, exec_config: Optional[ExecutionConfig] = None):
        """
        Initialize the executor.
        
        Args:
            exec_config: Execution configuration settings
        """
        self.exec_config = exec_config or ExecutionConfig()
        self._simulation_client = None
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def execute_config(
        self,
        config: Dict[str, Any],
        class_name: str = "DynamicPickPlace",
    ) -> Dict[str, Any]:
        """
        Execute a simulation with the given configuration.
        
        Args:
            config: Configuration dictionary
            class_name: Name for the dynamic environment class
            
        Returns:
            Dictionary containing execution results:
            {
                "success_rate": float,
                "episode_successes": List[bool],
                "env_name": str,
                "task_expression": str,
            }
        """
        from gr00t.eval.simulation import (
            MultiStepConfig,
            SimulationConfig,
            SimulationInferenceClient,
            VideoConfig,
        )
        
        # Set global random seeds for deterministic behavior
        seed = self.exec_config.seed
        print(f"Setting random seeds to {seed} for deterministic execution...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create the dynamic environment class
        print(f"[DEBUG] Creating environment class '{class_name}'...")
        env_class = DynamicEnvironmentFactory.create_environment_class(
            config, 
            class_name=class_name,
        )
        print(f"[DEBUG] Environment class created successfully")
        
        # Register the environment class globally so robocasa can find it
        print(f"[DEBUG] Registering environment '{class_name}'...")
        self._register_environment(env_class, class_name)
        print(f"[DEBUG] Environment registered successfully")
        
        # Build the environment name
        env_name = f"gr1_unified/{class_name}_GR1ArmsAndWaistFourierHands_Env"
        print(f"[DEBUG] Environment name: {env_name}")
        
        # Connect to simulation server
        print(f"Connecting to inference server at {self.exec_config.host}:{self.exec_config.port}...")
        simulation_client = SimulationInferenceClient(
            host=self.exec_config.host,
            port=self.exec_config.port,
        )
        print(f"[DEBUG] Simulation client created")
        
        # Check server connection
        print("Checking server connection...")
        try:
            modality_config = simulation_client.get_modality_config()
            print(f"Available modality configs: {list(modality_config.keys())}")
        except RuntimeError as e:
            raise RuntimeError(
                f"Cannot connect to inference server at {self.exec_config.host}:{self.exec_config.port}. "
                f"Error: {e}\n"
                f"Please ensure the inference server is running:\n"
                f"  python scripts/simulation_service.py --model_path <MODEL_PATH> --port {self.exec_config.port}"
            )
        
        # Reset server history before running this configuration
        print("Resetting server history...")
        try:
            reset_result = simulation_client.reset()
            print(f"Server reset: {reset_result.get('message', 'success')}")
        except Exception as e:
            print(f"Warning: Could not reset server history: {e}")
            # Continue execution even if reset fails (for backward compatibility)
        
        # Create video directory
        if self.exec_config.save_video:
            Path(self.exec_config.video_dir).mkdir(parents=True, exist_ok=True)
        
        # Build simulation config
        sim_config = SimulationConfig(
            env_name=env_name,
            n_episodes=self.exec_config.n_episodes,
            n_envs=self.exec_config.n_envs,
            video=VideoConfig(video_dir=self.exec_config.video_dir) if self.exec_config.save_video else None,
            multistep=MultiStepConfig(
                n_action_steps=self.exec_config.n_action_steps,
                max_episode_steps=self.exec_config.max_episode_steps,
            ),
            seed=self.exec_config.seed,  # Pass seed for deterministic reset
            record_path=self.exec_config.record_path,
            record_signals=self.exec_config.record_signals,  # Enable signal recording
            recording_metadata={
                "env_name": env_name,
                "class_name": class_name,
                "task_config": config,
                "task_expression": config.get("task_expression"),
                "config_number": config.get("abstract_config", {}).get(
                    "configuration_number"
                ),
            },
        )
        
        # Run simulation
        task_expression = config.get("task_expression", "unknown")
        print(f"Running simulation for task: {task_expression}")
        print(f"Environment: {env_name}")
        print(f"[DEBUG] About to call run_simulation...")
        print(f"[DEBUG] Sim config: env_name={sim_config.env_name}, n_episodes={sim_config.n_episodes}, n_envs={sim_config.n_envs}")
        
        try:
            env_name_result, episode_successes = simulation_client.run_simulation(sim_config)
            print(f"[DEBUG] run_simulation completed successfully")
        except Exception as e:
            print(f"[ERROR] run_simulation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Calculate results
        success_rate = float(np.mean(episode_successes))
        
        results = {
            "success_rate": success_rate,
            "episode_successes": list(episode_successes),
            "env_name": env_name_result,
            "task_expression": task_expression,
            "config_number": config.get("abstract_config", {}).get("configuration_number"),
        }
        
        print(f"\nResults for {env_name_result}:")
        print(f"  Task: {task_expression}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Episodes: {episode_successes}")
        
        return results
    
    def execute_file(self, config_path: str) -> Dict[str, Any]:
        """
        Execute a simulation from a JSON configuration file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Execution results dictionary
        """
        config = self.load_config(config_path)
        
        # Generate class name from config number if available
        config_num = config.get("abstract_config", {}).get("configuration_number", 0)
        class_name = f"Config{config_num}PickPlace" if config_num else "DynamicPickPlace"
        
        return self.execute_config(config, class_name=class_name)
    
    def execute_from_number(
        self,
        config_number: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Instantiate and execute a configuration by number.
        
        Args:
            config_number: Configuration number to instantiate and execute
            seed: Random seed for instantiation
            
        Returns:
            Execution results dictionary
        """
        from .instantiate import ConfigurationInstantiator
        
        # Instantiate configuration
        instantiator = ConfigurationInstantiator()
        config = instantiator.instantiate(config_number, seed=seed)
        
        # Execute
        class_name = f"Config{config_number}PickPlace"
        return self.execute_config(config, class_name=class_name)

    def execute_with_perturbations(
        self,
        config_number: int,
        position_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
        rotation_offsets: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        class_name_suffix: str = "",
    ) -> Dict[str, Any]:
        """
        Execute a configuration with position/rotation perturbations.
        
        This method is designed for falsification - it allows small perturbations
        to object positions while using the standard simulation infrastructure.
        
        Args:
            config_number: Configuration number to instantiate
            position_offsets: Dict mapping object names to (x_offset, y_offset) tuples.
                             Offsets should be small (< 0.05 meters recommended).
            rotation_offsets: Dict mapping object names to rotation offsets (radians)
            seed: Random seed for instantiation
            class_name_suffix: Optional suffix for the class name (for uniqueness)
            
        Returns:
            Execution results dictionary with success rate and episode results
            
        Example:
            executor.execute_with_perturbations(
                config_number=176,
                position_offsets={"fruit": (0.02, -0.01)},
                rotation_offsets={"fruit": 0.1},
            )
        """
        from .instantiate import ConfigurationInstantiator
        
        # Instantiate configuration
        instantiator = ConfigurationInstantiator()
        config = instantiator.instantiate(config_number, seed=seed)
        
        # Apply position perturbations
        if position_offsets:
            for obj_cfg in config.get("object_plan", []):
                obj_name = obj_cfg.get("name", "")
                if obj_name in position_offsets:
                    offset_x, offset_y = position_offsets[obj_name]
                    if "placement" in obj_cfg and "pos" in obj_cfg["placement"]:
                        current_pos = obj_cfg["placement"]["pos"]
                        # Apply offset (clamped to small values for safety)
                        offset_x = max(-0.05, min(0.05, offset_x))
                        offset_y = max(-0.05, min(0.05, offset_y))
                        obj_cfg["placement"]["pos"] = (
                            current_pos[0] + offset_x,
                            current_pos[1] + offset_y
                        )
                        print(f"[Perturbation] {obj_name}: pos {current_pos} -> {obj_cfg['placement']['pos']}")
        
        # Apply rotation perturbations
        if rotation_offsets:
            for obj_cfg in config.get("object_plan", []):
                obj_name = obj_cfg.get("name", "")
                if obj_name in rotation_offsets:
                    offset_rot = rotation_offsets[obj_name]
                    if "placement" in obj_cfg:
                        current_rot = obj_cfg["placement"].get("rotation", 0.0)
                        # Apply offset (clamped for safety)
                        offset_rot = max(-0.5, min(0.5, offset_rot))
                        obj_cfg["placement"]["rotation"] = current_rot + offset_rot
                        print(f"[Perturbation] {obj_name}: rot {current_rot} -> {obj_cfg['placement']['rotation']}")
        
        # Execute with unique class name
        class_name = f"Config{config_number}Falsify{class_name_suffix}"
        return self.execute_config(config, class_name=class_name)

    def replay_recording(self, recording_path: str) -> Dict[str, Any]:
        """
        Replay a previously recorded simulation without querying the model server.

        Args:
            recording_path: Path to the msgpack recording created by a prior run.

        Returns:
            Execution results dictionary similar to execute_config.
        """
        from gr00t.eval.simulation import (
            MultiStepConfig,
            SimulationConfig,
            SimulationInferenceClient,
            VideoConfig,
            load_recording,
        )

        recording = load_recording(Path(recording_path))
        metadata = recording.get("metadata", {})
        sim_cfg_dict = recording.get("simulation_config", {})
        recorded_env_name = metadata.get("env_name") or sim_cfg_dict.get("env_name")
        class_name = metadata.get("class_name", "DynamicPickPlace")
        task_config = metadata.get("task_config")

        # Re-register the dynamic environment if task config is present
        if task_config is not None:
            env_class = DynamicEnvironmentFactory.create_environment_class(
                task_config, class_name=class_name
            )
            self._register_environment(env_class, class_name)

        env_name = recorded_env_name or f"gr1_unified/{class_name}_GR1ArmsAndWaistFourierHands_Env"
        video_cfg_dict = sim_cfg_dict.get("video", {}) if sim_cfg_dict else {}
        multistep_cfg = sim_cfg_dict.get("multistep", {}) if sim_cfg_dict else {}
        # Always honor CLI video path if provided; otherwise fall back to recorded path
        video_dir = self.exec_config.video_dir if self.exec_config.save_video else None
        if video_dir is None:
            video_dir = video_cfg_dict.get("video_dir")

        sim_config = SimulationConfig(
            env_name=env_name,
            n_episodes=sim_cfg_dict.get("n_episodes", self.exec_config.n_episodes),
            n_envs=sim_cfg_dict.get("n_envs", 1),
            video=VideoConfig(
                video_dir=video_dir,
                steps_per_render=video_cfg_dict.get("steps_per_render", 2),
                fps=video_cfg_dict.get("fps", 10),
                codec=video_cfg_dict.get("codec", "h264"),
                input_pix_fmt=video_cfg_dict.get("input_pix_fmt", "rgb24"),
                crf=video_cfg_dict.get("crf", 22),
                thread_type=video_cfg_dict.get("thread_type", "FRAME"),
                thread_count=video_cfg_dict.get("thread_count", 1),
            ),
            multistep=MultiStepConfig(
                video_delta_indices=np.array(
                    multistep_cfg.get("video_delta_indices", [0])
                ),
                state_delta_indices=np.array(
                    multistep_cfg.get("state_delta_indices", [0])
                ),
                n_action_steps=multistep_cfg.get(
                    "n_action_steps", self.exec_config.n_action_steps
                ),
                max_episode_steps=multistep_cfg.get(
                    "max_episode_steps", self.exec_config.max_episode_steps
                ),
            ),
            replay_path=str(recording_path),
        )

        simulation_client = SimulationInferenceClient(
            host=self.exec_config.host,
            port=self.exec_config.port,
        )

        env_name_result, episode_successes = simulation_client.run_simulation(sim_config)
        success_rate = float(np.mean(episode_successes))
        results = {
            "success_rate": success_rate,
            "episode_successes": list(episode_successes),
            "env_name": env_name_result,
            "task_expression": metadata.get("task_expression", "replay"),
            "config_number": metadata.get("config_number"),
        }

        print(f"\nReplay results for {env_name_result}:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Episodes: {episode_successes}")

        return results
    
    def _register_environment(self, env_class: Type, class_name: str):
        """
        Register the environment class with robosuite and gymnasium.
        
        This follows the same pattern as robocasa's gymnasium_groot.py
        """
        import sys
        from gymnasium.envs.registration import register
        from robosuite.environments.base import REGISTERED_ENVS
        
        # Step 1: Register with robosuite's REGISTERED_ENVS
        REGISTERED_ENVS[class_name] = env_class
        
        # Step 2: Add to robocasa.environments.tabletop module
        import robocasa.environments.tabletop as tabletop_module
        setattr(tabletop_module, class_name, env_class)
        
        # Step 3: Create the gymnasium wrapper class
        from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv
        
        gym_class_name = f"{class_name}_GR1ArmsAndWaistFourierHands_Env"
        
        # Create the wrapper class dynamically
        def make_init(env_name, robots_name):
            def __init__(self, **kwargs):
                super(self.__class__, self).__init__(
                    env_name=env_name,
                    robots_name=robots_name,
                    **kwargs,
                )
            return __init__
        
        gym_env_class = type(
            gym_class_name,
            (GrootRoboCasaEnv,),
            {
                "__init__": make_init(class_name, "GR1ArmsAndWaistFourierHands"),
            },
        )
        
        # Step 4: Add the gym wrapper to the gymnasium_groot module
        import robocasa.utils.gym_utils.gymnasium_groot as gym_groot_module
        setattr(gym_groot_module, gym_class_name, gym_env_class)
        
        # Step 5: Register with gymnasium for gr1_unified namespace
        env_id = f"gr1_unified/{gym_class_name}"
        
        try:
            register(
                id=env_id,
                entry_point=f"robocasa.utils.gym_utils.gymnasium_groot:{gym_class_name}",
            )
        except Exception as e:
            # Environment might already be registered, which is fine
            if "already registered" not in str(e).lower():
                print(f"Warning: Could not register {env_id}: {e}")


# ---------------------------------------------------------------------------
# Batch Executor
# ---------------------------------------------------------------------------

class BatchExecutor:
    """
    Executor for running multiple configurations in batch.
    
    Useful for testing multiple configurations or running experiments.
    """
    
    def __init__(self, exec_config: Optional[ExecutionConfig] = None):
        """Initialize the batch executor."""
        self.executor = ConfigurationExecutor(exec_config)
    
    def execute_range(
        self,
        start: int,
        end: int,
        seed: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a range of configurations.
        
        Args:
            start: Starting configuration number (inclusive)
            end: Ending configuration number (inclusive)
            seed: Random seed for instantiation
            output_file: Optional file to save results
            
        Returns:
            List of execution results
        """
        results = []
        
        for config_num in range(start, end + 1):
            print(f"\n{'='*60}")
            print(f"Executing configuration {config_num}")
            print(f"{'='*60}")
            
            try:
                result = self.executor.execute_from_number(config_num, seed=seed)
                results.append(result)
            except Exception as e:
                print(f"Error executing config {config_num}: {e}")
                results.append({
                    "config_number": config_num,
                    "error": str(e),
                    "success_rate": 0.0,
                })
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def execute_list(
        self,
        config_numbers: List[int],
        seed: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a list of specific configurations.
        
        Args:
            config_numbers: List of configuration numbers to execute
            seed: Random seed for instantiation
            output_file: Optional file to save results
            
        Returns:
            List of execution results
        """
        results = []
        
        for config_num in config_numbers:
            print(f"\n{'='*60}")
            print(f"Executing configuration {config_num}")
            print(f"{'='*60}")
            
            try:
                result = self.executor.execute_from_number(config_num, seed=seed)
                results.append(result)
            except Exception as e:
                print(f"Error executing config {config_num}: {e}")
                results.append({
                    "config_number": config_num,
                    "error": str(e),
                    "success_rate": 0.0,
                })
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of batch execution."""
        print(f"\n{'='*60}")
        print("BATCH EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        total = len(results)
        successful = sum(1 for r in results if "error" not in r)
        failed = total - successful
        
        success_rates = [r["success_rate"] for r in results if "error" not in r]
        avg_success = np.mean(success_rates) if success_rates else 0.0
        
        print(f"Total configurations: {total}")
        print(f"Successfully executed: {successful}")
        print(f"Failed: {failed}")
        print(f"Average success rate: {avg_success:.2%}")
        
        if success_rates:
            print(f"Best success rate: {max(success_rates):.2%}")
            print(f"Worst success rate: {min(success_rates):.2%}")


# ---------------------------------------------------------------------------
# Condition Evaluator
# ---------------------------------------------------------------------------

class ConditionEvaluator:
    """
    Evaluator for checking initial conditions and determining which branch
    of a conditional task should be executed.
    """
    
    @staticmethod
    def evaluate_condition(
        condition: str,
        initial_conditions: Dict[str, bool],
    ) -> bool:
        """
        Evaluate a condition against initial conditions.
        
        Args:
            condition: Condition string (e.g., "loc vegetable basket")
            initial_conditions: Dictionary of initial conditions
            
        Returns:
            True if condition is satisfied, False otherwise
        """
        parts = condition.strip().split()
        
        if len(parts) >= 3 and parts[0] == "loc":
            obj = parts[1]
            location = parts[2]
            
            # Build the condition key
            condition_key = f"loc({obj}, {location})"
            return initial_conditions.get(condition_key, False)
        
        return False
    
    @staticmethod
    def get_expected_actions(
        config: Dict[str, Any],
    ) -> Tuple[List[Dict], str]:
        """
        Determine which actions should be executed based on initial conditions.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Tuple of (actions_to_execute, branch_name)
        """
        from .instantiate import AbstractConfigParser
        
        task_expression = config.get("task_expression", "")
        initial_conditions = config.get("abstract_config", {}).get("initial_conditions", {})
        
        parsed = AbstractConfigParser.parse_task_expression(task_expression)
        
        if parsed["type"] == "conditional":
            condition = parsed.get("condition", "")
            condition_met = ConditionEvaluator.evaluate_condition(condition, initial_conditions)
            
            if condition_met:
                return parsed.get("then_actions", []), "then"
            else:
                return parsed.get("else_actions", []), "else"
        else:
            return parsed.get("sequence_actions", []), "sequence"


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Execute simulations with generated configurations"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration JSON file"
    )
    input_group.add_argument(
        "--config-number", "-n",
        type=int,
        help="Configuration number to instantiate and execute"
    )
    input_group.add_argument(
        "--batch-range",
        type=str,
        help="Range of configurations to execute (e.g., '1-10')"
    )
    input_group.add_argument(
        "--batch-list",
        type=str,
        help="Comma-separated list of configurations (e.g., '1,5,10,15')"
    )
    input_group.add_argument(
        "--replay-recording",
        type=str,
        help="Replay a previously recorded simulation (msgpack file)",
    )
    
    # Instantiation options
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for instantiation"
    )
    
    # Server options
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Inference server host (default: localhost)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5555,
        help="Inference server port (default: 5555)"
    )
    
    # Simulation options
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum steps per episode (default: 600)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)"
    )
    
    # Output options
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="Directory to save videos (default: ./videos)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help="Path to store a full environment recording for offline replay"
    )
    parser.add_argument(
        "--no-record-signals",
        action="store_true",
        help="Disable recording of signals in env_infos (signals are recorded by default)"
    )
    
    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be executed without running"
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Use offscreen rendering (for headless environments)"
    )
    
    args = parser.parse_args()
    
    # Build execution config
    # Use provided seed or default for deterministic behavior
    exec_seed = args.seed if args.seed is not None else DEFAULT_SEED
    exec_config = ExecutionConfig(
        host=args.host,
        port=args.port,
        n_episodes=args.episodes,
        n_envs=args.n_envs,
        max_episode_steps=args.max_steps,
        video_dir=args.video_dir,
        save_video=not args.no_video,
        record_path=args.record_path,
        replay_recording=args.replay_recording,
        record_signals=not args.no_record_signals,  # Default to True unless --no-record-signals is set
        seed=exec_seed,
    )
    
    # Handle dry run
    if args.dry_run:
        print("DRY RUN - Configuration preview:")
        print(f"  Host: {exec_config.host}:{exec_config.port}")
        print(f"  Episodes: {exec_config.n_episodes}")
        print(f"  Max steps: {exec_config.max_episode_steps}")
        print(f"  Video: {'disabled' if args.no_video else args.video_dir}")
        if exec_config.record_path:
            print(f"  Recording to: {exec_config.record_path}")
        print(f"  Signal recording: {'enabled' if exec_config.record_signals else 'disabled'}")
        
        if args.config:
            print(f"  Config file: {args.config}")
        elif args.config_number:
            print(f"  Config number: {args.config_number}")
        elif args.batch_range:
            print(f"  Batch range: {args.batch_range}")
        elif args.batch_list:
            print(f"  Batch list: {args.batch_list}")
        elif args.replay_recording:
            print(f"  Replay recording: {args.replay_recording}")
        return
    
    # Handle offscreen rendering
    if args.offscreen:
        import os
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        print("Offscreen rendering enabled (MUJOCO_GL=osmesa)")
    
    # Execute based on input type
    try:
        if args.config:
            # Execute from config file
            executor = ConfigurationExecutor(exec_config)
            result = executor.execute_file(args.config)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {args.output}")
                
        elif args.replay_recording:
            executor = ConfigurationExecutor(exec_config)
            result = executor.replay_recording(args.replay_recording)

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {args.output}")

        elif args.config_number:
            # Execute from config number
            executor = ConfigurationExecutor(exec_config)
            result = executor.execute_from_number(args.config_number, seed=args.seed)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {args.output}")
                
        elif args.batch_range:
            # Execute range of configs
            start, end = map(int, args.batch_range.split('-'))
            batch_executor = BatchExecutor(exec_config)
            batch_executor.execute_range(start, end, seed=args.seed, output_file=args.output)
            
        elif args.batch_list:
            # Execute list of configs
            config_numbers = [int(x.strip()) for x in args.batch_list.split(',')]
            batch_executor = BatchExecutor(exec_config)
            batch_executor.execute_list(config_numbers, seed=args.seed, output_file=args.output)
            
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nError during execution: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
