"""
STL-based Falsification Framework for Robot Tasks

This module provides a falsification approach that:
1. Converts task success conditions to STL (Signal Temporal Logic) specifications
2. Monitors specification robustness using rtamt
3. Uses optimization (nevergrad) to find specification violations

Key concepts:
- STL atomic predicates are of the form f(s) > 0, where s is the state
- Robustness ρ(φ, s, t) measures how much the signal satisfies/violates the spec
- Falsification aims to find inputs that minimize robustness (make it negative)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np

# Try to import torch for seed setting (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# STL SPECIFICATION TYPES
# =============================================================================

class TemporalOperator(Enum):
    """Temporal operators for STL specifications"""
    EVENTUALLY = "eventually"  # F[a,b] φ: φ holds at some point in [t+a, t+b]
    GLOBALLY = "globally"      # G[a,b] φ: φ holds at all points in [t+a, t+b]
    UNTIL = "until"           # φ U[a,b] ψ: φ holds until ψ, within [t+a, t+b]


@dataclass
class AtomicPredicate:
    """
    Atomic predicate: f(s) > 0
    
    The robustness value is simply f(s), where:
    - f(s) > 0 means the predicate is satisfied
    - f(s) < 0 means the predicate is violated
    - |f(s)| indicates the margin of satisfaction/violation
    
    Attributes:
        name: Human-readable name for the predicate
        func: Function that computes f(s) given environment state
        description: Optional description of what this predicate checks
    """
    name: str
    func: Callable[[Any], float]
    description: str = ""
    
    def evaluate(self, state: Any) -> float:
        """Compute robustness value f(s)"""
        return self.func(state)
    
    def to_rtamt_var(self) -> str:
        """Convert to rtamt variable name (alphanumeric only)"""
        return self.name.replace("(", "_").replace(")", "").replace(",", "_").replace(" ", "")


@dataclass
class STLFormula:
    """
    STL Formula representation
    
    Can be:
    - An atomic predicate
    - A boolean combination (and, or, not, implies)
    - A temporal formula (eventually, globally, until)
    """
    operator: Optional[str] = None  # 'and', 'or', 'not', 'implies', 'eventually', 'globally', 'until'
    operands: List[Union['STLFormula', AtomicPredicate]] = field(default_factory=list)
    time_bounds: Tuple[float, float] = (0, float('inf'))  # [a, b] for temporal operators
    
    @classmethod
    def atomic(cls, predicate: AtomicPredicate) -> 'STLFormula':
        """Create formula from atomic predicate"""
        return cls(operator=None, operands=[predicate])
    
    @classmethod
    def eventually(cls, phi: 'STLFormula', time_bounds: Tuple[float, float]) -> 'STLFormula':
        """F[a,b] φ"""
        return cls(operator='eventually', operands=[phi], time_bounds=time_bounds)
    
    @classmethod
    def globally(cls, phi: 'STLFormula', time_bounds: Tuple[float, float]) -> 'STLFormula':
        """G[a,b] φ"""
        return cls(operator='globally', operands=[phi], time_bounds=time_bounds)
    
    @classmethod
    def until(cls, phi: 'STLFormula', psi: 'STLFormula', time_bounds: Tuple[float, float]) -> 'STLFormula':
        """φ U[a,b] ψ"""
        return cls(operator='until', operands=[phi, psi], time_bounds=time_bounds)
    
    @classmethod
    def and_(cls, *operands: 'STLFormula') -> 'STLFormula':
        """φ ∧ ψ"""
        return cls(operator='and', operands=list(operands))
    
    @classmethod
    def or_(cls, *operands: 'STLFormula') -> 'STLFormula':
        """φ ∨ ψ"""
        return cls(operator='or', operands=list(operands))
    
    @classmethod
    def not_(cls, phi: 'STLFormula') -> 'STLFormula':
        """¬φ"""
        return cls(operator='not', operands=[phi])
    
    @classmethod
    def implies(cls, phi: 'STLFormula', psi: 'STLFormula') -> 'STLFormula':
        """φ → ψ (equivalent to ¬φ ∨ ψ)"""
        return cls(operator='implies', operands=[phi, psi])
    
    def get_atomic_predicates(self) -> List[AtomicPredicate]:
        """Extract all atomic predicates from the formula"""
        predicates = []
        for op in self.operands:
            if isinstance(op, AtomicPredicate):
                predicates.append(op)
            elif isinstance(op, STLFormula):
                predicates.extend(op.get_atomic_predicates())
        return predicates
    
    def to_rtamt_spec(self) -> str:
        """Convert to rtamt specification string"""
        if self.operator is None:
            # Atomic predicate
            pred = self.operands[0]
            return f"({pred.to_rtamt_var()} >= 0)"
        
        elif self.operator == 'and':
            return "(" + " and ".join(op.to_rtamt_spec() for op in self.operands) + ")"
        
        elif self.operator == 'or':
            return "(" + " or ".join(op.to_rtamt_spec() for op in self.operands) + ")"
        
        elif self.operator == 'not':
            return f"(not {self.operands[0].to_rtamt_spec()})"
        
        elif self.operator == 'implies':
            return f"({self.operands[0].to_rtamt_spec()} -> {self.operands[1].to_rtamt_spec()})"
        
        elif self.operator == 'eventually':
            a, b = self.time_bounds
            b_str = str(int(b)) if b != float('inf') else "inf"
            return f"eventually[{int(a)},{b_str}] {self.operands[0].to_rtamt_spec()}"
        
        elif self.operator == 'globally':
            a, b = self.time_bounds
            b_str = str(int(b)) if b != float('inf') else "inf"
            return f"always[{int(a)},{b_str}] {self.operands[0].to_rtamt_spec()}"
        
        elif self.operator == 'until':
            a, b = self.time_bounds
            b_str = str(int(b)) if b != float('inf') else "inf"
            return f"({self.operands[0].to_rtamt_spec()} until[{int(a)},{b_str}] {self.operands[1].to_rtamt_spec()})"
        
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


# =============================================================================
# ROBUSTNESS FUNCTIONS FOR TABLETOP TASKS
# =============================================================================

class RobustnessExtractor:
    """
    Extracts continuous robustness functions from discrete success conditions.
    
    Converts boolean checks like `distance > threshold` to continuous functions
    like `distance - threshold` where positive means satisfied.
    """
    
    @staticmethod
    def gripper_obj_distance(env, obj_name: str, gripper_side: str = "right") -> float:
        """
        Compute distance between gripper and object or fixture.
        
        Robustness for "gripper far from object": distance - threshold
        """
        # Check if obj_name is a fixture (drawer, cabinet, microwave, etc.)
        # Fixtures are accessed via env.fixtures or env.get_fixture()
        if obj_name in ("drawer", "cabinet", "microwave", "counter"):
            # Try to get fixture from environment
            fixture = None
            if hasattr(env, 'fixtures') and obj_name in env.fixtures:
                fixture = env.fixtures[obj_name]
            elif hasattr(env, 'get_fixture'):
                try:
                    fixture = env.get_fixture(obj_name)
                except:
                    pass
            
            # Also try drawer_tabletop_main_group for drawer
            if fixture is None and obj_name == "drawer":
                if hasattr(env, 'fixtures'):
                    for name, fxtr in env.fixtures.items():
                        if "drawer" in name.lower() and "tabletop" in name.lower():
                            fixture = fxtr
                            break
            
            if fixture is not None:
                # Use fixture's position (fixture.pos is [x, y, z])
                obj_pos = np.array(fixture.pos)
            else:
                # Fallback: use a default position if fixture not found
                # This shouldn't happen, but handle gracefully
                print(f"Warning: Fixture '{obj_name}' not found, using default position")
                obj_pos = np.array([0.0, 0.0, 0.92])  # Default counter height
        else:
            # Regular object - use obj_body_id
            if obj_name not in env.obj_body_id:
                # Try to get object from env.objects and use root_body
                if hasattr(env, 'objects') and obj_name in env.objects:
                    obj = env.objects[obj_name]
                    if hasattr(obj, 'root_body'):
                        # Get body ID from root_body name
                        body_id = env.sim.model.body_name2id(obj.root_body)
                        obj_pos = env.sim.data.body_xpos[body_id]
                    else:
                        # Fallback: use default position
                        print(f"Warning: Object '{obj_name}' found in env.objects but has no root_body, using default position")
                        obj_pos = np.array([0.0, 0.0, 0.92])  # Default counter height
                else:
                    # Object not found at all - use default position
                    # print(f"Warning: Object '{obj_name}' not found in obj_body_id or env.objects. Available in obj_body_id: {list(env.obj_body_id.keys()) if hasattr(env, 'obj_body_id') else 'N/A'}, Available in objects: {list(env.objects.keys()) if hasattr(env, 'objects') else 'N/A'}. Using default position.")
                    obj_pos = np.array([0.0, 0.0, 0.92])  # Default counter height
            else:
                obj_pos = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
        
        gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id[gripper_side]]
        return float(np.linalg.norm(gripper_site_pos - obj_pos))
    
    @staticmethod
    def gripper_obj_far_robustness(env, obj_name: str, threshold: float = 0.25) -> float:
        """
        Robustness for gripper being far from object.
        Positive when gripper is far (satisfied), negative when close (violated).
        """
        left_dist = RobustnessExtractor.gripper_obj_distance(env, obj_name, "left")
        right_dist = RobustnessExtractor.gripper_obj_distance(env, obj_name, "right")
        # Both grippers must be far - take minimum
        return min(left_dist - threshold, right_dist - threshold)
    
    @staticmethod
    def obj_upright_robustness(env, obj_name: str, threshold: float = 0.8) -> float:
        """
        Robustness for object being upright.
        Based on z-component of object's up vector (should be close to 1).
        
        For fixtures (drawer, cabinet, microwave), always return positive robustness
        since fixtures are fixed and don't need upright check.
        """
        # Check if obj_name is a fixture - fixtures don't need upright check
        if obj_name in ("drawer", "cabinet", "microwave", "counter"):
            # Fixtures are fixed, so they're always "upright"
            return 1.0 - threshold  # Positive robustness (satisfied)
        
        # Regular object - check upright
        if obj_name not in env.obj_body_id:
            # Try to get object from env.objects and use root_body
            if hasattr(env, 'objects') and obj_name in env.objects:
                obj = env.objects[obj_name]
                if hasattr(obj, 'root_body'):
                    # Get body ID from root_body name
                    body_id = env.sim.model.body_name2id(obj.root_body)
                    obj_id = body_id
                else:
                    # Fallback: return neutral robustness
                    print(f"Warning: Object '{obj_name}' found in env.objects but has no root_body, returning neutral robustness")
                    return 0.0
            else:
                # Object not found - return neutral robustness
                # print(f"Warning: Object '{obj_name}' not found in obj_body_id or env.objects, returning neutral robustness")
                return 0.0
        else:
            obj_id = env.obj_body_id[obj_name]
        
        import robosuite.utils.transform_utils as T
        obj_quat = T.convert_quat(np.array(env.sim.data.body_xquat[obj_id]), to="xyzw")
        obj_mat = T.quat2mat(obj_quat)
        z_component = obj_mat[:, 2][2]  # z-component of up vector
        return float(z_component - threshold)
    
    @staticmethod
    def obj_in_receptacle_robustness(env, obj_name: str, receptacle_name: str, 
                                     threshold: float = None) -> float:
        """
        Robustness for object being in receptacle (container or fixture).
        Based on horizontal distance to receptacle center.
        """
        # Get object position
        if obj_name not in env.obj_body_id:
            # Try to get object from env.objects and use root_body
            if hasattr(env, 'objects') and obj_name in env.objects:
                obj = env.objects[obj_name]
                if hasattr(obj, 'root_body'):
                    # Get body ID from root_body name
                    body_id = env.sim.model.body_name2id(obj.root_body)
                    obj_pos = np.array(env.sim.data.body_xpos[body_id])
                else:
                    # Fallback: return negative robustness
                    print(f"Warning: Object '{obj_name}' found in env.objects but has no root_body, returning negative robustness")
                    return -1.0
            else:
                # Object not found - return negative robustness
                # print(f"Warning: Object '{obj_name}' not found in obj_body_id or env.objects, returning negative robustness")
                return -1.0
        else:
            obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        
        # Check if receptacle is a fixture
        if receptacle_name in ("drawer", "cabinet", "microwave"):
            # Get fixture
            fixture = None
            if hasattr(env, 'fixtures') and receptacle_name in env.fixtures:
                fixture = env.fixtures[receptacle_name]
            elif hasattr(env, 'get_fixture'):
                try:
                    fixture = env.get_fixture(receptacle_name)
                except:
                    pass
            
            # Also try drawer_tabletop_main_group for drawer
            if fixture is None and receptacle_name == "drawer":
                if hasattr(env, 'fixtures'):
                    for name, fxtr in env.fixtures.items():
                        if "drawer" in name.lower() and "tabletop" in name.lower():
                            fixture = fxtr
                            break
            
            if fixture is not None:
                recep_pos = np.array(fixture.pos)
                # Use fixture size for threshold
                if threshold is None:
                    # Estimate threshold from fixture size
                    if hasattr(fixture, 'size'):
                        threshold = max(fixture.size[0], fixture.size[1]) * 0.5
                    else:
                        threshold = 0.15  # Default threshold
                
                # For fixtures, check if object is inside based on position
                # Simple check: object should be near fixture position
                horiz_dist = np.linalg.norm(obj_pos[:2] - recep_pos[:2])
                distance_robustness = threshold - horiz_dist
                
                # For drawer/cabinet/microwave, also check z-height (object should be inside)
                z_diff = obj_pos[2] - recep_pos[2]
                # Object should be at similar or lower height than fixture opening
                height_robustness = 0.1 - abs(z_diff)  # Within 0.1m height
                
                return min(distance_robustness, height_robustness)
            else:
                # Fixture not found, return negative robustness
                return -1.0
        else:
            # Regular container object
            # Get receptacle position
            if receptacle_name not in env.obj_body_id:
                # Try to get receptacle from env.objects and use root_body
                if hasattr(env, 'objects') and receptacle_name in env.objects:
                    recep = env.objects[receptacle_name]
                    if hasattr(recep, 'root_body'):
                        # Get body ID from root_body name
                        body_id = env.sim.model.body_name2id(recep.root_body)
                        recep_pos = np.array(env.sim.data.body_xpos[body_id])
                    else:
                        # Fallback: return negative robustness
                        print(f"Warning: Receptacle '{receptacle_name}' found in env.objects but has no root_body, returning negative robustness")
                        return -1.0
                else:
                    # Receptacle not found - return negative robustness
                    print(f"Warning: Receptacle '{receptacle_name}' not found in obj_body_id or env.objects, returning negative robustness")
                    return -1.0
            else:
                recep_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[receptacle_name]])
            
            recep = env.objects[receptacle_name]
            if threshold is None:
                threshold = recep.horizontal_radius * 0.7
            
            # Check contact
            obj = env.objects[obj_name]
            has_contact = float(env.check_contact(obj, recep))
            
            # Distance-based robustness (negative distance means inside)
            horiz_dist = np.linalg.norm(obj_pos[:2] - recep_pos[:2])
            
            # Combine: need both contact AND within distance
            # Use min semantics for AND
            distance_robustness = threshold - horiz_dist
            contact_robustness = has_contact * 1.0 - 0.5  # +0.5 if contact, -0.5 if no contact
            
            return min(distance_robustness, contact_robustness)
    
    @staticmethod
    def obj_fixture_contact_robustness(env, obj_name: str, fixture) -> float:
        """
        Robustness for object being in contact with fixture.
        Returns positive if contact, negative if no contact.
        """
        obj = env.objects[obj_name]
        has_contact = env.check_contact(obj, fixture)
        # Return discrete robustness: +1 for contact, -1 for no contact
        return 1.0 if has_contact else -1.0
    
    @staticmethod
    def door_state_robustness(env, fixture, target_state: str = "closed", 
                              threshold: float = 0.005) -> float:
        """
        Robustness for door being in target state.
        
        Args:
            target_state: "closed" (state <= threshold) or "open" (state >= 1-threshold)
        """
        door_state = fixture.get_door_state(env=env)["door"]
        
        if target_state == "closed":
            return float(threshold - door_state)  # Positive when closed
        else:  # open
            return float(door_state - (1 - threshold))  # Positive when open
    
    @staticmethod
    def obj_inside_fixture_robustness(env, obj_name: str, fixture) -> float:
        """
        Robustness for object being inside a fixture (e.g., microwave).
        Uses the fixture's internal boundary checks.
        """
        import robosuite.utils.transform_utils as T
        
        obj = env.objects[obj_name]
        # Get object position
        if obj.name not in env.obj_body_id:
            # Try to use root_body
            if hasattr(obj, 'root_body'):
                # Get body ID from root_body name
                body_id = env.sim.model.body_name2id(obj.root_body)
                obj_pos = np.array(env.sim.data.body_xpos[body_id])
            else:
                # Fallback: return negative robustness
                print(f"Warning: Object '{obj.name}' found in env.objects but has no root_body and not in obj_body_id, returning negative robustness")
                return -1.0
        else:
            obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
        
        # Get fixture boundary
        fixtr_p0, fixtr_px, fixtr_py, fixtr_pz = fixture.get_int_sites(relative=False)
        u = fixtr_px - fixtr_p0
        v = fixtr_py - fixtr_p0
        w = fixtr_pz - fixtr_p0
        
        # Compute signed distances to each boundary
        # For a point to be inside, all these should be positive
        d_u_min = np.dot(u, obj_pos) - np.dot(u, fixtr_p0)
        d_u_max = np.dot(u, fixtr_px) - np.dot(u, obj_pos)
        d_v_min = np.dot(v, obj_pos) - np.dot(v, fixtr_p0)
        d_v_max = np.dot(v, fixtr_py) - np.dot(v, obj_pos)
        d_w_min = np.dot(w, obj_pos) - np.dot(w, fixtr_p0)
        d_w_max = np.dot(w, fixtr_pz) - np.dot(w, obj_pos)
        
        # Robustness is the minimum signed distance (most constraining)
        return float(min(d_u_min, d_u_max, d_v_min, d_v_max, d_w_min, d_w_max))


# =============================================================================
# TASK TO STL CONVERTER
# =============================================================================

class TaskSpecConverter(ABC):
    """
    Abstract base class for converting task success conditions to STL specifications.
    
    Each task type (PnP, Microwave, etc.) should implement its own converter.
    """
    
    @abstractmethod
    def get_atomic_predicates(self, env) -> Dict[str, AtomicPredicate]:
        """
        Extract atomic predicates from the environment.
        Returns a dict mapping predicate names to AtomicPredicate objects.
        """
        pass
    
    @abstractmethod
    def get_stl_spec(self, env, horizon: int) -> STLFormula:
        """
        Construct the STL specification for the task.
        
        Args:
            env: The environment instance
            horizon: Time horizon for the specification
        """
        pass
    
    @abstractmethod
    def get_subtask_specs(self, env, horizon: int) -> List[Tuple[str, STLFormula]]:
        """
        Get specifications for individual subtasks (for step-by-step monitoring).
        Returns list of (name, formula) tuples.
        """
        pass


class PnPTaskSpecConverter(TaskSpecConverter):
    """
    Converter for Pick-and-Place tasks.
    
    Success condition:
    - gripper_container_far AND gripper_obj_far AND obj_in_container 
      AND NOT obj_on_counter AND container_upright
    
    STL Spec:
    - eventually[0, H] (gripper_far AND obj_in_container AND container_upright)
    
    Subtask decomposition (from domain.pl style reasoning):
    1. Reach and grasp object
    2. Transport to container
    3. Place and release
    """
    
    def __init__(self, obj_name: str = "obj", container_name: str = "container",
                 gripper_threshold: float = 0.25, upright_threshold: float = 0.8):
        self.obj_name = obj_name
        self.container_name = container_name
        self.gripper_threshold = gripper_threshold
        self.upright_threshold = upright_threshold
    
    def get_atomic_predicates(self, env) -> Dict[str, AtomicPredicate]:
        predicates = {}
        
        # Gripper far from object
        predicates["gripper_obj_far"] = AtomicPredicate(
            name="gripper_obj_far",
            func=lambda s, env=env: RobustnessExtractor.gripper_obj_far_robustness(
                env, self.obj_name, self.gripper_threshold
            ),
            description=f"Gripper is far from {self.obj_name}"
        )
        
        # Gripper far from container
        predicates["gripper_container_far"] = AtomicPredicate(
            name="gripper_container_far",
            func=lambda s, env=env: RobustnessExtractor.gripper_obj_far_robustness(
                env, self.container_name, self.gripper_threshold
            ),
            description=f"Gripper is far from {self.container_name}"
        )
        
        # Object in container
        predicates["obj_in_container"] = AtomicPredicate(
            name="obj_in_container",
            func=lambda s, env=env: RobustnessExtractor.obj_in_receptacle_robustness(
                env, self.obj_name, self.container_name
            ),
            description=f"{self.obj_name} is in {self.container_name}"
        )
        
        # Container upright
        predicates["container_upright"] = AtomicPredicate(
            name="container_upright",
            func=lambda s, env=env: RobustnessExtractor.obj_upright_robustness(
                env, self.container_name, self.upright_threshold
            ),
            description=f"{self.container_name} is upright"
        )
        
        return predicates
    
    def get_stl_spec(self, env, horizon: int) -> STLFormula:
        predicates = self.get_atomic_predicates(env)
        
        # Success: eventually all conditions are met
        success_condition = STLFormula.and_(
            STLFormula.atomic(predicates["gripper_obj_far"]),
            STLFormula.atomic(predicates["gripper_container_far"]),
            STLFormula.atomic(predicates["obj_in_container"]),
            STLFormula.atomic(predicates["container_upright"])
        )
        
        return STLFormula.eventually(success_condition, (0, horizon))
    
    def get_subtask_specs(self, env, horizon: int) -> List[Tuple[str, STLFormula]]:
        """
        Decompose PnP into subtasks based on domain.pl reasoning:
        1. grasp: gripper close to object
        2. transport: object lifted (not on counter)
        3. place: object in container
        4. release: gripper far from object
        """
        predicates = self.get_atomic_predicates(env)
        
        # Approximate subtask timing
        t1 = horizon // 4
        t2 = horizon // 2
        t3 = 3 * horizon // 4
        
        subtasks = [
            # Grasp phase: gripper should approach object
            ("grasp", STLFormula.eventually(
                STLFormula.not_(STLFormula.atomic(predicates["gripper_obj_far"])),
                (0, t1)
            )),
            
            # Place phase: object should be in container
            ("place", STLFormula.eventually(
                STLFormula.atomic(predicates["obj_in_container"]),
                (t1, t3)
            )),
            
            # Release phase: gripper should be far
            ("release", STLFormula.eventually(
                STLFormula.and_(
                    STLFormula.atomic(predicates["gripper_obj_far"]),
                    STLFormula.atomic(predicates["obj_in_container"])
                ),
                (t2, horizon)
            ))
        ]
        
        return subtasks


class MicrowavePnPTaskSpecConverter(TaskSpecConverter):
    """
    Converter for Microwave Pick-and-Place tasks.
    
    Success condition:
    - obj_inside_microwave AND door_closed
    
    STL Spec:
    - eventually[0, H] (obj_inside AND door_closed)
    """
    
    def __init__(self, obj_name: str = "obj", door_threshold: float = 0.005):
        self.obj_name = obj_name
        self.door_threshold = door_threshold
    
    def get_atomic_predicates(self, env) -> Dict[str, AtomicPredicate]:
        predicates = {}
        
        # Object inside microwave
        predicates["obj_inside_microwave"] = AtomicPredicate(
            name="obj_inside_microwave",
            func=lambda s, env=env: RobustnessExtractor.obj_inside_fixture_robustness(
                env, self.obj_name, env.microwave
            ),
            description=f"{self.obj_name} is inside microwave"
        )
        
        # Door closed
        predicates["door_closed"] = AtomicPredicate(
            name="door_closed",
            func=lambda s, env=env: RobustnessExtractor.door_state_robustness(
                env, env.microwave, "closed", self.door_threshold
            ),
            description="Microwave door is closed"
        )
        
        # Gripper far from object
        predicates["gripper_obj_far"] = AtomicPredicate(
            name="gripper_obj_far",
            func=lambda s, env=env: RobustnessExtractor.gripper_obj_far_robustness(
                env, self.obj_name, 0.25
            ),
            description=f"Gripper is far from {self.obj_name}"
        )
        
        return predicates
    
    def get_stl_spec(self, env, horizon: int) -> STLFormula:
        predicates = self.get_atomic_predicates(env)
        
        success_condition = STLFormula.and_(
            STLFormula.atomic(predicates["obj_inside_microwave"]),
            STLFormula.atomic(predicates["door_closed"])
        )
        
        return STLFormula.eventually(success_condition, (0, horizon))
    
    def get_subtask_specs(self, env, horizon: int) -> List[Tuple[str, STLFormula]]:
        predicates = self.get_atomic_predicates(env)
        
        t1 = horizon // 3
        t2 = 2 * horizon // 3
        
        return [
            ("grasp", STLFormula.eventually(
                STLFormula.not_(STLFormula.atomic(predicates["gripper_obj_far"])),
                (0, t1)
            )),
            ("place_in_microwave", STLFormula.eventually(
                STLFormula.atomic(predicates["obj_inside_microwave"]),
                (t1, t2)
            )),
            ("close_door", STLFormula.eventually(
                STLFormula.and_(
                    STLFormula.atomic(predicates["obj_inside_microwave"]),
                    STLFormula.atomic(predicates["door_closed"])
                ),
                (t2, horizon)
            ))
        ]


class FixtureTaskSpecConverter(TaskSpecConverter):
    """
    Converter for fixture-only tasks (open/close drawer, cabinet, microwave door).
    
    These tasks don't involve picking up objects - they only manipulate fixtures.
    
    Success condition (for "close drawer"):
    - door_closed (drawer door is closed)
    
    STL Spec:
    - eventually[0, H] (door_closed)
    """
    
    def __init__(self, fixture_name: str = "drawer", target_state: str = "closed",
                 door_threshold: float = 0.005):
        """
        Args:
            fixture_name: Name of the fixture (drawer, cabinet, microwave)
            target_state: Target door state ("closed" or "open")
            door_threshold: Threshold for door state robustness
        """
        self.fixture_name = fixture_name
        self.target_state = target_state
        self.door_threshold = door_threshold
    
    def _get_fixture(self, env):
        """Get the fixture object from the environment."""
        fixture = None
        
        # Try direct fixture access
        if hasattr(env, 'fixtures') and self.fixture_name in env.fixtures:
            fixture = env.fixtures[self.fixture_name]
        elif hasattr(env, 'get_fixture'):
            try:
                fixture = env.get_fixture(self.fixture_name)
            except:
                pass
        
        # For drawer, also try drawer_tabletop_main_group
        if fixture is None and self.fixture_name == "drawer":
            if hasattr(env, 'fixtures'):
                for name, fxtr in env.fixtures.items():
                    if "drawer" in name.lower() and "tabletop" in name.lower():
                        fixture = fxtr
                        break
        
        # For cabinet, try cabinet_tabletop variants
        if fixture is None and self.fixture_name == "cabinet":
            if hasattr(env, 'fixtures'):
                for name, fxtr in env.fixtures.items():
                    if "cabinet" in name.lower() and "tabletop" in name.lower():
                        fixture = fxtr
                        break
        
        return fixture
    
    def get_atomic_predicates(self, env) -> Dict[str, AtomicPredicate]:
        predicates = {}
        
        fixture = self._get_fixture(env)
        
        if fixture is None:
            # Return a default predicate that always fails if fixture not found
            predicates["door_target_state"] = AtomicPredicate(
                name="door_target_state",
                func=lambda s: -1.0,  # Always violated
                description=f"{self.fixture_name} door is {self.target_state} (fixture not found)"
            )
            return predicates
        
        # Door state predicate
        predicates["door_target_state"] = AtomicPredicate(
            name="door_target_state",
            func=lambda s, env=env, fixture=fixture: RobustnessExtractor.door_state_robustness(
                env, fixture, self.target_state, self.door_threshold
            ),
            description=f"{self.fixture_name} door is {self.target_state}"
        )
        
        return predicates
    
    def get_stl_spec(self, env, horizon: int) -> STLFormula:
        predicates = self.get_atomic_predicates(env)
        
        # Success: eventually the door is in target state
        success_condition = STLFormula.atomic(predicates["door_target_state"])
        
        return STLFormula.eventually(success_condition, (0, horizon))
    
    def get_subtask_specs(self, env, horizon: int) -> List[Tuple[str, STLFormula]]:
        predicates = self.get_atomic_predicates(env)
        
        # For fixture tasks, there's typically just one subtask
        action_name = f"{self.target_state}_{self.fixture_name}"
        
        return [
            (action_name, STLFormula.eventually(
                STLFormula.atomic(predicates["door_target_state"]),
                (0, horizon)
            ))
        ]


# =============================================================================
# RTAMT-BASED MONITOR
# =============================================================================

class STLMonitor:
    """
    Monitors STL specifications using rtamt library.
    
    Tracks robustness values over time as signals are updated.
    """
    
    def __init__(self, formula: STLFormula):
        import rtamt
        
        self.formula = formula
        self.predicates = formula.get_atomic_predicates()
        self.spec = rtamt.StlDiscreteTimeSpecification()
        
        # Declare variables for each atomic predicate
        for pred in self.predicates:
            var_name = pred.to_rtamt_var()
            self.spec.declare_var(var_name, 'float')
        
        # Set the specification
        self.spec.spec = formula.to_rtamt_spec()
        
        # Parse and prepare for online monitoring
        try:
            self.spec.parse()
            self.spec.pastify()
        except Exception as e:
            raise RuntimeError(f"Failed to parse STL spec: {e}\nSpec: {formula.to_rtamt_spec()}")
        
        self.time_step = 0
        self.robustness_trace = []
    
    def reset(self):
        """Reset monitor for new episode"""
        self.time_step = 0
        self.robustness_trace = []
        # Note: rtamt doesn't have a reset method, we need to recreate
        self.__init__(self.formula)
    
    def update(self, state: Any) -> float:
        """
        Update monitor with new state and return robustness value.
        
        Args:
            state: Environment state (passed to predicate functions)
            
        Returns:
            Robustness value at current time step
        """
        # Evaluate all predicates
        signal_values = []
        for pred in self.predicates:
            var_name = pred.to_rtamt_var()
            value = pred.evaluate(state)
            signal_values.append((var_name, value))
        
        # Update rtamt
        robustness = self.spec.update(self.time_step, signal_values)
        
        self.robustness_trace.append(robustness)
        self.time_step += 1
        
        return robustness
    
    def get_final_robustness(self) -> float:
        """Get the final robustness value"""
        if not self.robustness_trace:
            return float('-inf')
        return self.robustness_trace[-1]
    
    def get_robustness_trace(self) -> List[float]:
        """Get the full robustness trace"""
        return self.robustness_trace.copy()


# =============================================================================
# FALSIFIER USING NEVERGRAD
# =============================================================================

class STLFalsifier:
    """
    STL-based falsifier using optimization to find specification violations.
    
    Uses nevergrad to search for:
    - Initial states that lead to failure
    - Control perturbations that cause violations
    - Environmental parameters that expose bugs
    """
    
    def __init__(self, 
                 env_factory: Callable[[], Any],
                 policy: Callable[[Any, Any], Any],
                 spec_converter: TaskSpecConverter,
                 horizon: int = 100,
                 budget: int = 100):
        """
        Initialize falsifier.
        
        Args:
            env_factory: Function that creates a new environment instance
            policy: Policy function: action = policy(env, observation)
            spec_converter: Converter that generates STL spec from task
            horizon: Time horizon for rollouts
            budget: Optimization budget (number of trials)
        """
        self.env_factory = env_factory
        self.policy = policy
        self.spec_converter = spec_converter
        self.horizon = horizon
        self.budget = budget
        
        self.best_robustness = float('inf')
        self.best_params = None
        self.falsifying_trace = None
    
    def _rollout(self, env, params: np.ndarray) -> Tuple[float, List[Dict]]:
        """
        Execute rollout with given parameters and return robustness.
        
        Args:
            env: Environment instance
            params: Parameters to perturb (could be initial state, action noise, etc.)
            
        Returns:
            Tuple of (final_robustness, trace)
        """
        # Get STL spec for this environment
        formula = self.spec_converter.get_stl_spec(env, self.horizon)
        monitor = STLMonitor(formula)
        
        obs = env.reset()
        trace = []
        
        for t in range(self.horizon):
            # Get action from policy
            action = self.policy(env, obs)
            
            # Optionally perturb action based on params
            # (This is a simple perturbation scheme - can be extended)
            if len(params) > 0:
                noise_scale = params[0] if len(params) >= 1 else 0.0
                action = action + noise_scale * np.random.randn(*action.shape)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Update monitor
            robustness = monitor.update(env)
            
            # Record trace
            trace.append({
                'time': t,
                'robustness': robustness,
                'action': action.copy() if hasattr(action, 'copy') else action,
                'done': done
            })
            
            if done:
                break
        
        return monitor.get_final_robustness(), trace
    
    def falsify(self, param_dim: int = 1) -> Dict[str, Any]:
        """
        Run falsification to find specification violations.
        
        Args:
            param_dim: Dimension of parameter space to search
            
        Returns:
            Dictionary with falsification results
        """
        import nevergrad as ng
        
        # Create optimizer
        optimizer = ng.optimizers.NGOpt(
            parametrization=ng.p.Array(shape=(param_dim,)).set_bounds(-1, 1),
            budget=self.budget
        )
        
        def objective(params: np.ndarray) -> float:
            """Objective: minimize robustness (find violations)"""
            env = self.env_factory()
            try:
                robustness, trace = self._rollout(env, params)
                
                # Track best (most violating) result
                if robustness < self.best_robustness:
                    self.best_robustness = robustness
                    self.best_params = params.copy()
                    self.falsifying_trace = trace
                
                return robustness
            finally:
                env.close()
        
        # Run optimization
        recommendation = optimizer.minimize(objective)
        
        # Results
        return {
            'falsified': self.best_robustness < 0,
            'best_robustness': self.best_robustness,
            'best_params': self.best_params,
            'falsifying_trace': self.falsifying_trace,
            'recommendation': recommendation.value
        }


# =============================================================================
# UTILITY: CREATE SPEC FROM ENVIRONMENT
# =============================================================================

def get_task_converter(env) -> TaskSpecConverter:
    """
    Factory function to get appropriate converter for an environment.
    
    Inspects the environment class to determine task type and extracts object names.
    """
    env_class = type(env).__name__
    
    # Check for fixture-only tasks first (no objects, only door/fixture operations)
    # This handles tasks like "close drawer", "open cabinet", etc.
    is_fixture_only = False
    fixture_name = None
    target_state = None
    
    if hasattr(env, 'success_plan') and env.success_plan:
        # Check if all success criteria are fixture-related (door_open, door_closed)
        fixture_criteria = []
        object_criteria = []
        
        for criterion in env.success_plan:
            crit_type = criterion.get('type', '')
            if crit_type in ('door_open', 'door_closed', 'fixture_running'):
                fixture_criteria.append(criterion)
            elif crit_type in ('obj_in_receptacle', 'gripper_far'):
                object_criteria.append(criterion)
        
        # If only fixture criteria and no object criteria, it's a fixture-only task
        if fixture_criteria and not object_criteria:
            is_fixture_only = True
            # Get the first fixture criterion for the converter
            first_criterion = fixture_criteria[0]
            params = first_criterion.get('params', {})
            fixture_name = params.get('fixture_name', 'drawer')
            if first_criterion.get('type') == 'door_closed':
                target_state = 'closed'
            elif first_criterion.get('type') == 'door_open':
                target_state = 'open'
            else:
                target_state = 'closed'  # Default
    
    # Also check if object_plan is empty (another indicator of fixture-only task)
    if not is_fixture_only:
        object_plan = getattr(env, 'object_plan', None)
        if object_plan is not None and len(object_plan) == 0:
            # No objects - check if we have fixtures
            if hasattr(env, 'fixtures') and env.fixtures:
                is_fixture_only = True
                # Try to determine fixture from task_plan or fixtures
                task_plan = getattr(env, 'task_plan', [])
                for action in task_plan:
                    action_type = action.get('action', '')
                    if action_type in ('open', 'close'):
                        fixture_name = action.get('target_ref') or action.get('target', 'drawer')
                        target_state = 'closed' if action_type == 'close' else 'open'
                        break
                
                # Fallback: use first fixture
                if fixture_name is None:
                    for name in env.fixtures.keys():
                        if 'drawer' in name.lower():
                            fixture_name = 'drawer'
                            target_state = 'closed'
                            break
                        elif 'cabinet' in name.lower():
                            fixture_name = 'cabinet'
                            target_state = 'closed'
                            break
    
    if is_fixture_only and fixture_name and target_state:
        return FixtureTaskSpecConverter(
            fixture_name=fixture_name, 
            target_state=target_state
        )
    
    # Extract object names from environment
    obj_name = "obj"
    container_name = "container"
    
    # Try to get object names from object_plan
    if hasattr(env, 'object_plan') and env.object_plan:
        # Find graspable object (usually the one to be picked)
        graspable_objs = [obj.get('name') for obj in env.object_plan if obj.get('graspable', False)]
        if graspable_objs:
            obj_name = graspable_objs[0]
        
        # Find container object (usually the target for placement)
        # Look for objects that are not graspable or have specific names
        container_objs = [obj.get('name') for obj in env.object_plan 
                          if not obj.get('graspable', False) or obj.get('name') in ['container', 'plate', 'basket', 'bowl']]
        if container_objs:
            container_name = container_objs[0]
        elif len(env.object_plan) > 1:
            # If multiple objects, second one is usually the container
            container_name = env.object_plan[1].get('name', 'container')
    
    # Try to get from success_plan if available
    if hasattr(env, 'success_plan') and env.success_plan:
        for criterion in env.success_plan:
            if criterion.get('type') == 'obj_in_receptacle':
                params = criterion.get('params', {})
                if 'obj_name' in params:
                    obj_name = params['obj_name']
                if 'receptacle_name' in params:
                    container_name = params['receptacle_name']
    
    if 'Microwave' in env_class and 'PnP' in env_class:
        return MicrowavePnPTaskSpecConverter(obj_name=obj_name)
    elif 'PnP' in env_class or 'Pnp' in env_class:
        return PnPTaskSpecConverter(obj_name=obj_name, container_name=container_name)
    else:
        # Default to PnP
        return PnPTaskSpecConverter(obj_name=obj_name, container_name=container_name)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_standalone():
    """
    Standalone example showing the STL specification framework.
    (Does not require robocasa environment)
    """
    print("=" * 60)
    print("STL Falsification Framework Demo")
    print("=" * 60)
    
    # Create atomic predicates with mock functions
    gripper_far = AtomicPredicate(
        name="gripper_far",
        func=lambda s: s.get('gripper_dist', 0) - 0.25,
        description="Gripper is far from object"
    )
    
    obj_in_container = AtomicPredicate(
        name="obj_in_container", 
        func=lambda s: 0.1 - s.get('obj_dist', 0),  # Negative distance = inside
        description="Object is in container"
    )
    
    container_upright = AtomicPredicate(
        name="container_upright",
        func=lambda s: s.get('upright', 0.9) - 0.8,
        description="Container is upright"
    )
    
    # Build STL formula: eventually[0,10] (gripper_far AND obj_in_container AND container_upright)
    success = STLFormula.and_(
        STLFormula.atomic(gripper_far),
        STLFormula.atomic(obj_in_container),
        STLFormula.atomic(container_upright)
    )
    
    spec = STLFormula.eventually(success, (0, 10))
    
    print(f"\nSTL Specification:\n  {spec.to_rtamt_spec()}")
    
    # Create monitor
    print("\nCreating monitor...")
    monitor = STLMonitor(spec)
    
    # Simulate a trajectory
    print("\nSimulating trajectory...")
    trajectory = [
        # Initial: gripper close, object far from container
        {'gripper_dist': 0.1, 'obj_dist': 0.5, 'upright': 0.95},
        {'gripper_dist': 0.05, 'obj_dist': 0.4, 'upright': 0.95},
        {'gripper_dist': 0.02, 'obj_dist': 0.3, 'upright': 0.95},
        # Grasped, moving
        {'gripper_dist': 0.02, 'obj_dist': 0.2, 'upright': 0.9},
        {'gripper_dist': 0.02, 'obj_dist': 0.1, 'upright': 0.85},
        # Placing
        {'gripper_dist': 0.05, 'obj_dist': 0.05, 'upright': 0.85},
        {'gripper_dist': 0.1, 'obj_dist': 0.03, 'upright': 0.88},
        # Released
        {'gripper_dist': 0.3, 'obj_dist': 0.02, 'upright': 0.9},
        {'gripper_dist': 0.4, 'obj_dist': 0.01, 'upright': 0.92},
        {'gripper_dist': 0.5, 'obj_dist': 0.01, 'upright': 0.95},
    ]
    
    for t, state in enumerate(trajectory):
        rob = monitor.update(state)
        print(f"  t={t}: gripper_dist={state['gripper_dist']:.2f}, "
              f"obj_dist={state['obj_dist']:.2f}, upright={state['upright']:.2f} "
              f"-> robustness={rob:.3f}")
    
    print(f"\nFinal robustness: {monitor.get_final_robustness():.3f}")
    print(f"Specification satisfied: {monitor.get_final_robustness() >= 0}")


# =============================================================================
# CT CONFIGURATION-BASED FALSIFICATION
# =============================================================================

@dataclass
class ParameterRange:
    """
    Represents a continuous or discrete parameter range for falsification.
    
    Attributes:
        name: Parameter name
        low: Lower bound (for continuous) or list of values (for discrete)
        high: Upper bound (for continuous) or None (for discrete)
        is_discrete: Whether this is a discrete parameter
    """
    name: str
    low: Union[float, List[Any]]
    high: Optional[float] = None
    is_discrete: bool = False
    
    def sample(self, value: float) -> Any:
        """
        Sample from the range given a normalized value in [0, 1].
        """
        if self.is_discrete:
            # For discrete, value in [0,1] maps to index
            values = self.low  # low contains the list of values
            idx = int(value * len(values)) % len(values)
            return values[idx]
        else:
            # For continuous, linear interpolation
            return self.low + value * (self.high - self.low)


@dataclass
class CTParameterSpace:
    """
    Parameter space extracted from a CT configuration for falsification.
    
    Contains ranges for:
    - Object positions (x, y)
    - Object rotations
    - Initial fixture states
    - Action noise
    """
    # Position ranges for each object
    object_positions: Dict[str, Tuple[ParameterRange, ParameterRange]] = field(default_factory=dict)
    
    # Rotation ranges for each object
    object_rotations: Dict[str, ParameterRange] = field(default_factory=dict)
    
    # Initial fixture states (door_open, running)
    fixture_states: Dict[str, Dict[str, ParameterRange]] = field(default_factory=dict)
    
    def get_dimension(self) -> int:
        """Get total dimension of parameter space."""
        dim = 0
        # 2 dimensions per object (x, y)
        dim += len(self.object_positions) * 2
        # 1 dimension per object rotation
        dim += len(self.object_rotations)
        # Fixture states
        for states in self.fixture_states.values():
            dim += len(states)
        return dim
    
    def get_parameter_names(self) -> List[str]:
        """Get ordered list of parameter names."""
        names = []
        for obj_name in self.object_positions:
            names.append(f"{obj_name}_x")
            names.append(f"{obj_name}_y")
        for obj_name in self.object_rotations:
            names.append(f"{obj_name}_rot")
        for fixture_name, states in self.fixture_states.items():
            for state_name in states:
                names.append(f"{fixture_name}_{state_name}")
        return names
    
    def decode_params(self, normalized_params: np.ndarray) -> Dict[str, Any]:
        """
        Decode normalized parameters [0,1]^n to actual values.
        
        Args:
            normalized_params: Array of values in [0, 1]
            
        Returns:
            Dictionary mapping parameter names to values
        """
        decoded = {}
        idx = 0
        
        # Object positions
        for obj_name, (x_range, y_range) in self.object_positions.items():
            decoded[f"{obj_name}_x"] = x_range.sample(normalized_params[idx])
            decoded[f"{obj_name}_y"] = y_range.sample(normalized_params[idx + 1])
            idx += 2
        
        # Object rotations
        for obj_name, rot_range in self.object_rotations.items():
            decoded[f"{obj_name}_rot"] = rot_range.sample(normalized_params[idx])
            idx += 1
        
        # Fixture states
        for fixture_name, states in self.fixture_states.items():
            for state_name, state_range in states.items():
                decoded[f"{fixture_name}_{state_name}"] = state_range.sample(normalized_params[idx])
                idx += 1
        
        return decoded


class CTConfigurationLoader:
    """
    Loads and parses CT configurations for falsification.
    """
    
    # Default position ranges (from PositionDomain in instantiate.py)
    DEFAULT_POSITION_RANGE = {
        "counter": (-0.4, 0.5, -0.3, 0.3),  # x_min, x_max, y_min, y_max
    }
    
    # Zone definitions for structured placement
    ZONES = {
        "left": (-0.4, -0.15, -0.25, 0.0),
        "center_left": (-0.15, 0.05, -0.25, 0.0),
        "center": (0.05, 0.25, -0.25, 0.0),
        "center_right": (0.25, 0.45, -0.25, 0.0),
        "right": (0.45, 0.5, -0.25, 0.0),
        "front_left": (-0.4, -0.15, -0.3, -0.15),
        "front_center": (-0.15, 0.15, -0.3, -0.15),
        "front_right": (0.15, 0.5, -0.3, -0.15),
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize loader.
        
        Args:
            config_path: Path to 2-way_ct_configurations_trans.json
        """
        import json
        from pathlib import Path
        
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent / "data" / "2-way_ct_configurations_trans.json"
        
        self.config_path = Path(config_path)
        
        with open(self.config_path, 'r') as f:
            self.configurations = json.load(f)
    
    def get_configuration(self, config_number: int) -> Dict[str, Any]:
        """Get a specific configuration by number (1-based)."""
        for config in self.configurations:
            if config.get("configuration_number") == config_number:
                return config
        raise ValueError(f"Configuration {config_number} not found")
    
    def get_total_configurations(self) -> int:
        """Get total number of configurations."""
        return len(self.configurations)
    
    def extract_parameter_space(self, config_number: int) -> CTParameterSpace:
        """
        Extract parameter ranges from a CT configuration.
        
        Aligns with instantiate.py coordinate system:
        - Objects to pick: (0.5, -0.8) if no drawer/cabinet, (-0.5, -0.7) if drawer/cabinet present
        - Other objects: zone-based positions (same as instantiate.py)
        
        Args:
            config_number: Configuration number (1-based)
            
        Returns:
            CTParameterSpace with all parameter ranges
        """
        config = self.get_configuration(config_number)
        raw_params = config.get("raw_parameters", {})
        initial_conditions = config.get("initial_conditions", {})
        task_expression = config.get("task_expression", "")
        
        space = CTParameterSpace()
        
        # Parse task expression to identify objects to pick (same logic as instantiate.py)
        # Use multiple import strategies to handle different execution contexts
        AbstractConfigParser = None
        try:
            from .instantiate import AbstractConfigParser
        except ImportError:
            try:
                from concrete_layer.falsifier.instantiate import AbstractConfigParser
            except ImportError:
                try:
                    from falsifier.instantiate import AbstractConfigParser
                except ImportError:
                    import sys
                    falsifier_dir = Path(__file__).parent
                    parent_dir = falsifier_dir.parent.parent
                    if str(parent_dir) not in sys.path:
                        sys.path.insert(0, str(parent_dir))
                    from concrete_layer.falsifier.instantiate import AbstractConfigParser
        
        parsed_task = AbstractConfigParser.parse_task_expression(task_expression)
        
        # Extract objects that need to be picked up (same logic as instantiate.py._extract_objects_to_pick)
        objects_to_pick = set()
        # Extract target objects (place/put targets: basket, plate, etc.)
        target_objects = set()
        all_actions = []
        if parsed_task["type"] == "conditional":
            all_actions.extend(parsed_task.get("then_actions", []))
            all_actions.extend(parsed_task.get("else_actions", []))
        else:
            all_actions.extend(parsed_task.get("sequence_actions", []))
        
        for action in all_actions:
            action_type = action.get("action")
            if action_type == "pick":
                obj_name = action.get("object", "obj")
                if obj_name:
                    objects_to_pick.add(obj_name)
            elif action_type in ("place", "put"):
                obj_name = action.get("object", "obj")
                if obj_name:
                    objects_to_pick.add(obj_name)
                # Extract target (where object is placed)
                target = action.get("target", "")
                if target and target in ("basket", "plate", "bowl"):
                    target_objects.add(target)
        
        # Check if task involves drawer or cabinet (same logic as instantiate.py._check_has_drawer_or_cabinet)
        has_drawer_or_cabinet = False
        for action in all_actions:
            target = action.get("target", "")
            if target in ("drawer", "cabinet"):
                has_drawer_or_cabinet = True
                break
        
        # Check if task is specifically "put to cabinet" (place/put action targeting cabinet)
        is_put_to_cabinet = False
        for action in all_actions:
            action_type = action.get("action", "")
            target = action.get("target", "")
            if action_type in ("place", "put") and target == "cabinet":
                is_put_to_cabinet = True
                break
        
        # Get handedness from config (default to "right")
        handedness = "right"  # Default
        if "handedness" in config:
            handedness = config["handedness"]
        elif "fixture_plan" in config:
            fixture_plan = config.get("fixture_plan", {})
            if isinstance(fixture_plan, dict) and "handedness" in fixture_plan:
                handedness = fixture_plan["handedness"]
        # Ensure handedness is a string
        if not isinstance(handedness, str):
            handedness = "right"
        
        # Extract objects and their locations
        movable_objects = ["bread", "fruit", "vegetable", "plate"]
        container_objects = ["basket"]
        
        # Small perturbation range for slight modifications (user request)
        # ±0.08m for position, ±0.5 rad (~30 degrees) for rotation
        POSITION_PERTURBATION = 0.08
        ROTATION_PERTURBATION = 0.5
        
        # Default positions for objects to pick (source objects)
        # For "put to cabinet" tasks, fruit should be at left side (-0.5, -0.8) to match tabletop_cabinet_pnp.py
        # Otherwise, fruit (cup) at (0.5, -0.8)
        # Other objects: Right hand: (0.5, -0.8) if no drawer/cabinet, (-0.5, -0.7) if drawer/cabinet present
        #                Left hand: always (-0.5, -0.7)
        if is_put_to_cabinet:
            FRUIT_POSITION = (-0.5, -0.8)  # Left side for cabinet tasks
        else:
            FRUIT_POSITION = (0.5, -0.8)  # Default right side
        if handedness == "left":
            pick_base_pos = (-0.5, -0.7)
        elif has_drawer_or_cabinet:
            pick_base_pos = (-0.5, -0.7)
        else:
            pick_base_pos = (0.5, -0.8)
        
        # Default center positions for objects on counter (from instantiate.py zones)
        # Used for objects that are NOT to be picked up
        # For fruit, use FRUIT_POSITION which depends on task type (put to cabinet -> left, otherwise -> right)
        DEFAULT_POSITIONS = {
            "fruit": FRUIT_POSITION,   # Depends on task type (put to cabinet -> left, otherwise -> right)
            "plate": (-0.00, 0.00),     # center zone
            "vegetable": (0.5, -0.8),  # center_right zone
            "bread": (0.5, -0.8),    # left zone
            "basket": (0.0, 0.0),     # right zone
        }
        
        # Default positions for target objects (basket, plate)
        # Always at center (0, 0) regardless of handedness
        TARGET_POSITION = (0.0, 0.0)
        
        for obj in movable_objects + container_objects:
            location_key = f"{obj}_Location"
            location = raw_params.get(location_key, "na")
            
            if location != "na":
                # Determine default position based on object role
                if obj in objects_to_pick:
                    # Fruit (cup) position depends on task type: (-0.5, -0.8) for "put to cabinet", (0.5, -0.8) otherwise
                    if obj == "fruit":
                        default_x, default_y = FRUIT_POSITION
                    else:
                        # Use pick_base_pos for other objects that need to be picked up (source objects)
                        # Left hand: (-0.5, -0.7)
                        # Right hand: (0.5, -0.8) if no drawer/cabinet, (-0.5, -0.7) if drawer/cabinet present
                        default_x, default_y = pick_base_pos
                elif obj in target_objects and obj in ("basket", "plate"):
                    # Target objects (basket, plate) always at center (0, 0) regardless of handedness
                    default_x, default_y = TARGET_POSITION
                else:
                    # Use zone-based position for other objects (aligned with instantiate.py)
                    default_x, default_y = DEFAULT_POSITIONS.get(obj, (0.0, -0.125))
                
                # Clamp to counter bounds
                counter_x_min, counter_x_max, counter_y_min, counter_y_max = self.DEFAULT_POSITION_RANGE["counter"]
                x_min = max(counter_x_min, default_x - POSITION_PERTURBATION)
                x_max = min(counter_x_max, default_x + POSITION_PERTURBATION)
                y_min = max(counter_y_min, default_y - POSITION_PERTURBATION)
                y_max = min(counter_y_max, default_y + POSITION_PERTURBATION)
                
                space.object_positions[obj] = (
                    ParameterRange(f"{obj}_x", low=x_min, high=x_max),
                    ParameterRange(f"{obj}_y", low=y_min, high=y_max)
                )
                
                # Small rotation perturbation around 0
                space.object_rotations[obj] = ParameterRange(
                    f"{obj}_rot", low=0.0, high=ROTATION_PERTURBATION
                )
        
        # Extract fixture states
        fixtures_with_doors = ["cabinet", "drawer", "microwave"]
        
        # Determine task type from task expression to set appropriate door ranges
        task_expression = config.get("task_expression", "").lower()
        
        for fixture in fixtures_with_doors:
            door_state_key = f"{fixture.capitalize()}_Door_State"
            door_state = raw_params.get(door_state_key, "na")
            
            if door_state != "na":
                space.fixture_states.setdefault(fixture, {})
                
                # Determine the appropriate range based on task and initial state
                # For "close X" tasks: X starts open, so range should be 0.5-1.0 (open states)
                # For "open X" tasks: X starts closed, so range should be 0.0-0.5 (closed states)
                # The door_state from raw_params indicates initial state
                if f"close {fixture}" in task_expression:
                    # Task is to close the fixture - it should start open
                    # Vary the initial open degree from slightly open (0.5) to fully open (1.0)
                    low, high = 0.5, 1.0
                elif f"open {fixture}" in task_expression:
                    # Task is to open the fixture - it should start closed
                    # Vary the initial closed degree from fully closed (0.0) to slightly open (0.3)
                    low, high = 0.0, 0.3
                else:
                    # Default: allow small variations around current state
                    if door_state == "open":
                        low, high = 0.7, 1.0
                    else:
                        low, high = 0.0, 0.3
                
                space.fixture_states[fixture]["door"] = ParameterRange(
                    f"{fixture}_door", low=low, high=high
                )
        
        # Microwave running state
        running_state = raw_params.get("Microwave_Running_State", "na")
        if running_state != "na":
            space.fixture_states.setdefault("microwave", {})
            # Running state is discrete: 0 = stopped, 1 = running
            space.fixture_states["microwave"]["running"] = ParameterRange(
                "microwave_running", low=[False, True], is_discrete=True
            )
        
        return space
    
    def get_task_expression(self, config_number: int) -> str:
        """Get the task expression for a configuration."""
        config = self.get_configuration(config_number)
        return config.get("task_expression", "")


class CTBasedFalsifier:
    """
    Falsifier that works with CT configurations.
    
    Uses executor.py infrastructure for environment creation and simulation.
    
    Given a configuration number, it:
    1. Extracts parameter ranges from the configuration
    2. Uses ConfigurationExecutor to create environments with perturbations
    3. Runs falsification using nevergrad to find specification violations
    """
    
    def __init__(
        self,
        config_number: int,
        policy: Callable[[Any, Any], Any],
        horizon: int = 300,
        budget: int = 50,
        config_path: Optional[str] = None,
        seed: int = 42,
        render: bool = False,
        output_dir: Optional[str] = None,
        save_video: bool = True,
    ):
        """
        Initialize CT-based falsifier.
        
        Args:
            config_number: Configuration number from CT JSON (1-based)
            policy: Policy function: action = policy(env, observation)
            horizon: Time horizon for rollouts
            budget: Optimization budget (number of trials)
            config_path: Optional path to CT configurations JSON
            seed: Random seed for reproducibility
            render: Whether to enable rendering
            output_dir: Directory to store results (videos, recordings, logs)
            save_video: Whether to save videos of rollouts
        """
        self.config_number = config_number
        self.policy = policy
        self.horizon = horizon
        self.budget = budget
        self.seed = seed
        self.render = render
        self.save_video = save_video
        self.config_path = config_path  # Store config path for ConfigurationInstantiator
        
        # Setup output directories
        if output_dir is None:
            output_dir = f"./falsify_results/config_{config_number}"
        self.output_dir = Path(output_dir)
        self.videos_dir = self.output_dir / "videos"
        self.recordings_dir = self.output_dir / "recordings"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.loader = CTConfigurationLoader(config_path)
        self.config = self.loader.get_configuration(config_number)
        self.task_expression = self.loader.get_task_expression(config_number)
        
        # Extract parameter space
        self.param_space = self.loader.extract_parameter_space(config_number)
        
        # Import executor components
        self._import_executor_components()
        
        # Best results
        self.best_robustness = float('inf')
        self.best_params = None
        self.best_decoded_params = None
        self.falsifying_trace = None
        
        # Trial counter for unique class names
        self._trial_counter = 0
        
        # All trial results for logging
        self.all_trials = []
    
    def _import_executor_components(self):
        """Import components from executor.py."""
        import sys
        from pathlib import Path
        
        # Try multiple import strategies
        self.DynamicEnvironmentFactory = None
        self.ConfigurationInstantiator = None
        self.ConfigurationExecutor = None
        
        # Strategy 1: Try relative imports (preferred)
        try:
            from .executor import DynamicEnvironmentFactory, ConfigurationExecutor
            from .instantiate import ConfigurationInstantiator
            self.DynamicEnvironmentFactory = DynamicEnvironmentFactory
            self.ConfigurationInstantiator = ConfigurationInstantiator
            self.ConfigurationExecutor = ConfigurationExecutor
        except ImportError:
            pass
        
        # Strategy 2: Try concrete_layer.falsifier.* (when running from project root)
        if self.DynamicEnvironmentFactory is None:
            try:
                from concrete_layer.falsifier.executor import DynamicEnvironmentFactory, ConfigurationExecutor
                from concrete_layer.falsifier.instantiate import ConfigurationInstantiator
                self.DynamicEnvironmentFactory = DynamicEnvironmentFactory
                self.ConfigurationInstantiator = ConfigurationInstantiator
                self.ConfigurationExecutor = ConfigurationExecutor
            except ImportError:
                pass
        
        # Strategy 3: Try falsifier.* (backward compatibility)
        if self.DynamicEnvironmentFactory is None:
            try:
                from falsifier.executor import DynamicEnvironmentFactory, ConfigurationExecutor
                from falsifier.instantiate import ConfigurationInstantiator
                self.DynamicEnvironmentFactory = DynamicEnvironmentFactory
                self.ConfigurationInstantiator = ConfigurationInstantiator
                self.ConfigurationExecutor = ConfigurationExecutor
            except ImportError:
                pass
        
        # Strategy 4: Add to path and try concrete_layer.falsifier.*
        if self.DynamicEnvironmentFactory is None:
            falsifier_dir = Path(__file__).parent
            project_root = falsifier_dir.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            try:
                from concrete_layer.falsifier.executor import DynamicEnvironmentFactory, ConfigurationExecutor
                from concrete_layer.falsifier.instantiate import ConfigurationInstantiator
                self.DynamicEnvironmentFactory = DynamicEnvironmentFactory
                self.ConfigurationInstantiator = ConfigurationInstantiator
                self.ConfigurationExecutor = ConfigurationExecutor
            except ImportError:
                pass
    
    def _create_env_with_params(self, decoded_params: Dict[str, Any], trial_idx: int = 1):
        """
        Create an environment instance with specific parameters.
        
        Uses executor.py's DynamicEnvironmentFactory and registration logic.
        
        Args:
            decoded_params: Decoded parameter values (positions, rotations, etc.)
            trial_idx: Trial index (1-based). First trial (trial_idx=1) uses default center positions without perturbation.
            
        Returns:
            Gymnasium-wrapped environment with GrootRoboCasaEnv
        """
        import gymnasium as gym
        import time
        
        # Instantiate base configuration FIRST to get actual positions
        # Pass config_path to use the same configuration file as CTConfigurationLoader
        instantiator = self.ConfigurationInstantiator(config_path=self.config_path)
        config = instantiator.instantiate(self.config_number, seed=self.seed)
        
        # Identify put/place target objects from task_plan
        # These should use default handedness positions unless there's overlap
        task_plan = config.get("task_plan", [])
        place_targets = set()
        for action in task_plan:
            if action.get("action") == "place":
                target_ref = action.get("target_ref")
                if target_ref:
                    place_targets.add(target_ref)
        
        # Get handedness from config (default to "right" if not specified)
        # Check multiple possible locations for handedness
        handedness = "right"  # Default
        if "handedness" in config:
            handedness = config["handedness"]
        elif "fixture_plan" in config:
            fixture_plan = config["fixture_plan"]
            if isinstance(fixture_plan, dict) and "handedness" in fixture_plan:
                handedness = fixture_plan["handedness"]
        
        # Ensure handedness is a string
        if not isinstance(handedness, str):
            handedness = "right"
        
        # Default positions based on handedness (from executor.py and tabletop_pnp.py)
        # For containers: right: (0.9, -0.3), left: (-0.9, -0.3)
        # For other objects: right: (0.5, -0.8), left: (-0.5, -0.8)
        # NOTE: Target objects (basket, plate) always at (0, 0) regardless of handedness
        DEFAULT_POSITIONS = {
            "right": {
                "container": (0.9, -0.3),  # Containers (basket, bowl, plate) - NOT used for place targets
                "default": (0.5, -0.8),     # Other objects
            },
            "left": {
                "container": (-0.9, -0.3),
                "default": (-0.5, -0.8),
            },
        }
        
        # Target position for place targets (basket, plate) - always at center (0, 0)
        TARGET_POSITION = (0.0, 0.0)
        
        # Check if task is "put to cabinet" to determine fruit position
        task_expression = config.get("task_expression", "")
        is_put_to_cabinet = "put" in task_expression.lower() and "cabinet" in task_expression.lower() and ("put" in task_expression.lower().split() and "cabinet" in task_expression.lower().split())
        # More precise check: parse task to see if there's a place/put action targeting cabinet
        try:
            from .instantiate import AbstractConfigParser
            parsed_task = AbstractConfigParser.parse_task_expression(task_expression)
            all_actions = []
            if parsed_task["type"] == "conditional":
                all_actions.extend(parsed_task.get("then_actions", []))
                all_actions.extend(parsed_task.get("else_actions", []))
            else:
                all_actions.extend(parsed_task.get("sequence_actions", []))
            for action in all_actions:
                action_type = action.get("action", "")
                target = action.get("target", "")
                if action_type in ("place", "put") and target == "cabinet":
                    is_put_to_cabinet = True
                    break
        except Exception:
            pass  # Fallback to simple string check
        
        # For "put to cabinet" tasks, fruit should be at left side (-0.5, -0.8) to match tabletop_cabinet_pnp.py
        # Otherwise, fruit (cup) at (0.5, -0.8)
        if is_put_to_cabinet:
            FRUIT_POSITION = (-0.5, -0.8)  # Left side for cabinet tasks
        else:
            FRUIT_POSITION = (0.5, -0.8)  # Default right side
        
        
        # Track placed objects for collision detection
        placed_objects = []
        
        # Initialize perturbation flag (used for both object and fixture perturbations)
        # First trial (trial_idx=1) uses default center positions without perturbation
        # Subsequent trials apply perturbation from parameter space
        use_perturbation = (trial_idx > 1)  # Skip perturbation for first trial
        
        # Apply small perturbations to object positions
        # decoded_params contains absolute values from param_space ranges
        # We use the DIFFERENCE from range center as the offset to apply
        for obj_cfg in config.get("object_plan", []):
            obj_name = obj_cfg.get("name", "")
            
            # Check if this is a place target without explicit position
            is_place_target = obj_name in place_targets
            is_container = not obj_cfg.get("graspable", True) or obj_name in ["basket", "bowl", "plate"]
            
            x_key = f"{obj_name}_x"
            y_key = f"{obj_name}_y"
            rot_key = f"{obj_name}_rot"
            
            # Set default position for place targets
            # For basket/plate, always override to center (0, 0) regardless of instantiate.py's zone-based placement
            # For fruit (cup), always override to (0.5, -0.8) regardless of instantiate.py's zone-based placement
            if "placement" in obj_cfg:
                placement = obj_cfg["placement"]
                if is_place_target:
                    # Target objects (basket, plate) always at center (0, 0) regardless of handedness
                    if obj_name in ("basket", "plate"):
                        old_pos = placement.get("pos", "not set")
                        placement["pos"] = TARGET_POSITION
                        print(f"[POSITION] {obj_name} (place target): {old_pos} -> {TARGET_POSITION} (trial {trial_idx}, use_perturbation={use_perturbation})")
                    elif "pos" not in placement or placement.get("pos") is None:
                        # Use default handedness position for other place targets
                        pos_type = "container" if is_container else "default"
                        default_pos = DEFAULT_POSITIONS[handedness][pos_type]
                        placement["pos"] = default_pos
                        print(f"[DEBUG] Set default position for place target '{obj_name}': {default_pos}")
                elif obj_name == "fruit":
                    # Fruit (cup) position depends on task type
                    # For "put to cabinet" tasks: (-0.5, -0.8) (left side)
                    # Otherwise: (0.5, -0.8) (right side)
                    old_pos = placement.get("pos", "not set")
                    placement["pos"] = FRUIT_POSITION
                    task_type = "put to cabinet" if is_put_to_cabinet else "default"
                    print(f"[POSITION] fruit (cup): {old_pos} -> {FRUIT_POSITION} (trial {trial_idx}, task={task_type}, use_perturbation={use_perturbation})")
            
            # Only apply perturbation if the object has position parameters in param_space
            if x_key in decoded_params and y_key in decoded_params and use_perturbation:
                x_val = decoded_params[x_key]
                y_val = decoded_params[y_key]
                
                # Get center of range to compute offset
                if obj_name in self.param_space.object_positions:
                    x_range, y_range = self.param_space.object_positions[obj_name]
                    x_center = (x_range.low + x_range.high) / 2
                    y_center = (y_range.low + y_range.high) / 2
                    # Offset = decoded value - center
                    offset_x = x_val - x_center
                    offset_y = y_val - y_center
                    
                    # Clamp offsets to small values (±0.05m max)
                    offset_x = max(-0.05, min(0.05, offset_x))
                    offset_y = max(-0.05, min(0.05, offset_y))
                    
                    # Apply to actual position from config
                    if "placement" in obj_cfg:
                        placement = obj_cfg["placement"]
                        
                        # Determine base position
                        # For basket/plate, always use center (0, 0) as base, overriding instantiate.py's zone-based placement
                        # For fruit (cup), use FRUIT_POSITION which depends on task type (put to cabinet -> left, otherwise -> right)
                        if is_place_target:
                            if obj_name in ("basket", "plate"):
                                current_pos = TARGET_POSITION
                            elif "pos" not in placement or placement.get("pos") is None:
                                # Use default handedness position for other place targets
                                pos_type = "container" if is_container else "default"
                                current_pos = DEFAULT_POSITIONS[handedness][pos_type]
                            else:
                                current_pos = placement["pos"]
                        elif obj_name == "fruit":
                            # Fruit (cup) uses FRUIT_POSITION which depends on task type
                            # For "put to cabinet" tasks: (-0.5, -0.8) (left side)
                            # Otherwise: (0.5, -0.8) (right side)
                            current_pos = FRUIT_POSITION
                        elif "pos" in placement:
                            current_pos = placement["pos"]
                        else:
                            # No position specified, skip perturbation
                            current_pos = None
                        
                        if current_pos is not None:
                            new_pos = (
                                current_pos[0] + offset_x,
                                current_pos[1] + offset_y
                            )
                        
                            # Check for overlap with already placed objects
                            # Only adjust if there's overlap (collision detection)
                            from .instantiate import PositionDomain
                            position_domain = PositionDomain()
                            
                            # Get object size for collision detection
                            from .instantiate import ObjectModelDomain
                            object_domain = ObjectModelDomain()
                            category = obj_cfg.get("category", obj_name)
                            # Handle fruit -> cup mapping
                            if category == "fruit":
                                category = "cup"
                            object_size = object_domain.get_object_size(category)
                            
                            # Check collision
                            has_collision = position_domain.check_collision(new_pos, object_size, placed_objects)
                            
                            if has_collision:
                                # Try to find a non-colliding position near the default
                                # Use default position as base if this is a place target
                                if is_place_target:
                                    # Target objects (basket, plate) always at center (0, 0) regardless of handedness
                                    if obj_name in ("basket", "plate"):
                                        base_pos = TARGET_POSITION
                                    else:
                                        # Use default handedness position for other place targets
                                        pos_type = "container" if is_container else "default"
                                        base_pos = DEFAULT_POSITIONS[handedness][pos_type]
                                    # Try small offsets around base position
                                    import math
                                    for attempt in range(10):
                                        offset_scale = (attempt + 1) * 0.02  # 2cm per attempt
                                        angle = attempt * 0.628  # ~36 degrees per attempt
                                        candidate_pos = (
                                            base_pos[0] + offset_scale * math.cos(angle),
                                            base_pos[1] + offset_scale * math.sin(angle)
                                        )
                                        if not position_domain.check_collision(candidate_pos, object_size, placed_objects):
                                            new_pos = candidate_pos
                                            print(f"[DEBUG] Adjusted position for '{obj_name}' to avoid collision: {new_pos}")
                                            break
                                
                                # If still colliding, use the perturbed position anyway
                                # (falsification should explore edge cases)
                            
                            placement["pos"] = new_pos
                            print(f"[POSITION] {obj_name}: final position = {new_pos} (base: {current_pos}, offset: ({offset_x:.4f}, {offset_y:.4f}), trial {trial_idx})")
                            
                            # Add to placed_objects for collision detection
                            placed_objects.append((new_pos, object_size))
                        elif obj_name in ("fruit", "plate", "basket") and "placement" in obj_cfg:
                            # Output final position for important objects even without perturbation
                            placement = obj_cfg["placement"]
                            final_pos = placement.get("pos")
                            if final_pos:
                                print(f"[POSITION] {obj_name}: final position = {final_pos} (no perturbation, trial {trial_idx})")
            
            # Apply rotation perturbation (skip for first trial)
            if rot_key in decoded_params and obj_name in self.param_space.object_rotations and use_perturbation:
                rot_val = decoded_params[rot_key]
                rot_range = self.param_space.object_rotations[obj_name]
                rot_center = (rot_range.low + rot_range.high) / 2
                offset_rot = rot_val - rot_center
                
                # Clamp offset
                offset_rot = max(-0.25, min(0.25, offset_rot))
                
                if "placement" in obj_cfg:
                    current_rot = obj_cfg["placement"].get("rotation", 0.0)
                    obj_cfg["placement"]["rotation"] = current_rot + offset_rot
        
        # Apply fixture state perturbations (drawer_door, cabinet_door, etc.)
        # Similar to object positions, use offset from original state instead of absolute values
        # First trial (trial_idx=1) uses default states without perturbation
        fixture_plan = config.get("fixture_plan", {})
        fixture_requirements = fixture_plan.get("_fixture_requirements", {})
        
        for fixture_name, states in self.param_space.fixture_states.items():
            for state_name, state_range in states.items():
                param_key = f"{fixture_name}_{state_name}"
                if param_key in decoded_params and use_perturbation:
                    perturbed_value = decoded_params[param_key]
                    
                    # Get original state from fixture_requirements (if exists)
                    if fixture_name in fixture_requirements:
                        if state_name == "door":
                            # Handle door state with offset logic
                            # Clamp to valid range [0, 1] for continuous door states
                            perturbed_value = max(0.0, min(1.0, float(perturbed_value)))
                            
                            original_min = fixture_requirements[fixture_name].get("door_state_min", 0.0)
                            original_max = fixture_requirements[fixture_name].get("door_state_max", 0.0)
                            original_center = (original_min + original_max) / 2.0
                            
                            # Calculate offset from parameter space center
                            param_center = (state_range.low + state_range.high) / 2.0
                            offset = perturbed_value - param_center
                            
                            # Apply offset to original state (clamped to valid range)
                            new_min = max(0.0, min(1.0, original_min + offset))
                            new_max = max(0.0, min(1.0, original_max + offset))
                            
                            # Ensure min <= max
                            if new_min > new_max:
                                new_min, new_max = new_max, new_min
                            
                            # Update fixture_requirements with perturbed state
                            fixture_requirements[fixture_name]["door_state_min"] = new_min
                            fixture_requirements[fixture_name]["door_state_max"] = new_max
                        elif state_name == "running":
                            # Handle running state (discrete: True/False)
                            # For discrete states, decoded value is already boolean
                            fixture_requirements[fixture_name]["running_state"] = bool(perturbed_value)
                    else:
                        # No original state, use perturbed value directly
                        if state_name == "door":
                            # Clamp to valid range [0, 1]
                            perturbed_value = max(0.0, min(1.0, float(perturbed_value)))
                            fixture_requirements[fixture_name] = {
                                "type": fixture_name.upper(),
                                "door_state_min": perturbed_value,
                                "door_state_max": perturbed_value,
                            }
                        elif state_name == "running":
                            fixture_requirements[fixture_name] = {
                                "type": fixture_name.upper(),
                                "running_state": bool(perturbed_value),
                            }
        
        # Store back the updated fixture_requirements
        fixture_plan["_fixture_requirements"] = fixture_requirements
        config["fixture_plan"] = fixture_plan
        
        # Generate unique class name
        self._trial_counter += 1
        trial_id = int(time.time() * 1000) % 100000
        class_name = f"Falsify{self.config_number}_{trial_id}_{self._trial_counter}"
        
        # Create environment class using executor's factory
        # Verify language instruction is in config before creating environment
        lang_instruction = config.get("lang", "complete the task")
        if lang_instruction == "complete the task":
            print(f"  [WARNING] Language instruction not found in config, using default")
        else:
            print(f"  [DEBUG] Language instruction from config: '{lang_instruction}'")
        
        EnvClass = self.DynamicEnvironmentFactory.create_environment_class(
            config,
            class_name=class_name
        )
        
        # Use executor's registration method
        executor = self.ConfigurationExecutor()
        executor._register_environment(EnvClass, class_name)
        
        # Create environment using gymnasium.make()
        # Use high resolution cameras for video recording
        # Note: Must use 800x1280 (16:10) because process_img_cotrain expects this resolution
        env_id = f"gr1_unified/{class_name}_GR1ArmsAndWaistFourierHands_Env"
        
        # Enable render for video recording (enable_render=True enables offscreen renderer)
        # create_env_robosuite sets has_offscreen_renderer=enable_render internally
        enable_render = self.render or self.save_video  # Enable if rendering OR saving video
        
        env = gym.make(
            env_id,
            enable_render=enable_render,  # This enables offscreen renderer for video capture
            camera_widths=1280 if self.save_video else None,  # High res for video
            camera_heights=800 if self.save_video else None,  # 800x1280 = 16:10 aspect ratio (required by process_img_cotrain)
        )
        
        return env
    
    def _build_stl_spec(self, env) -> STLFormula:
        """
        Build STL specification from the task expression and success plan.
        """
        # Use generic converter based on task type
        converter = get_task_converter(env)
        return converter.get_stl_spec(env, self.horizon)
    
    def _get_underlying_robocasa_env(self, env):
        """
        Get the underlying robocasa environment from a gymnasium wrapper.
        
        The gym wrapper hierarchy is:
        - gymnasium wrappers (AutoResetWrapper, PassiveEnvChecker, etc.)
        - GrootRoboCasaEnv (self.env -> RoboCasaEnv)
        - RoboCasaEnv (self.env -> actual robocasa TabletopGive/etc with sim)
        
        Returns the environment that has the 'sim' attribute.
        """
        # First try to get to GrootRoboCasaEnv/RoboCasaEnv level
        current = env
        
        # Unwrap gymnasium wrappers
        while hasattr(current, 'env') and not hasattr(current, 'sim'):
            # Check if this is a RoboCasaEnv-like wrapper that has self.env as robocasa env
            inner = getattr(current, 'env', None)
            if inner is None:
                break
            
            # Check if inner has sim (we found the robocasa env)
            if hasattr(inner, 'sim'):
                return inner
            
            current = inner
        
        # If current has sim, return it
        if hasattr(current, 'sim'):
            return current
        
        # Fallback: try unwrapped
        if hasattr(env, 'unwrapped'):
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'sim'):
                return unwrapped
            # Check if unwrapped has .env
            if hasattr(unwrapped, 'env') and hasattr(unwrapped.env, 'sim'):
                return unwrapped.env
        
        # Last resort: return original env (may cause errors downstream)
        print(f"Warning: Could not find underlying robocasa env with 'sim' attribute")
        return env
    
    def _rollout(
        self,
        normalized_params: np.ndarray,
        trial_idx: int = 0,
    ) -> Tuple[float, List[Dict], Dict[str, Any]]:
        """
        Execute rollout with given parameters.
        
        Uses gymnasium-wrapped environment which handles action format conversion.
        
        Args:
            normalized_params: Normalized parameter values [0, 1]
            trial_idx: Trial index for video naming
        
        Returns:
            Tuple of (final_robustness, trace, decoded_params)
        """
        # Decode parameters
        decoded_params = self.param_space.decode_params(normalized_params)
        
        # Create environment with parameters (using gymnasium wrapper)
        try:
            env = self._create_env_with_params(decoded_params, trial_idx=trial_idx)
        except Exception as e:
            print(f"Failed to create environment: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), [], decoded_params
        
        # Setup video recording - get original resolution frames from underlying env
        video_recorder = None
        video_file_path = None
        if self.save_video:
            try:
                from gr00t.eval.wrappers.video_recording_wrapper import VideoRecorder
                
                # Create high-quality video recorder (same as run_give.py)
                video_recorder = VideoRecorder.create_h264(
                    fps=20,
                    codec="h264",
                    input_pix_fmt="rgb24",
                    crf=18,  # High quality (lower = better quality, 18 is good)
                )
                
                # Create video filename
                success_str = "fail"  # Will update after rollout
                rob_str = f"rob{0.0:.4f}".replace("-", "neg").replace(".", "p")  # Placeholder
                video_filename = f"trial_{trial_idx}_{success_str}_{rob_str}.mp4"
                video_file_path = self.videos_dir / video_filename
                
                # Start recording
                video_recorder.start(str(video_file_path))
                
            except ImportError:
                print(f"  Warning: VideoRecorder not available, using fallback method")
                video_recorder = None
        
        # Setup recording (msgpack format for offline replay)
        recording_path = None
        recorder = None
        try:
            from gr00t.eval.simulation import SimulationRecorder, SimulationConfig
            
            # Create recording path
            success_str = "fail"  # Will update after rollout
            rob_str = f"rob{0.0:.4f}".replace("-", "neg").replace(".", "p")  # Placeholder
            recording_filename = f"trial_{trial_idx}_{success_str}_{rob_str}.msgpack"
            recording_path = self.recordings_dir / recording_filename
            
            # Create a minimal SimulationConfig for the recorder
            # (we don't need all fields, just for metadata)
            sim_config = SimulationConfig(
                env_name=f"Falsify{self.config_number}_{trial_idx}",
                n_episodes=1,
                n_envs=1,
                recording_metadata={
                    "config_number": self.config_number,
                    "trial_idx": trial_idx,
                    "task_expression": self.task_expression,
                    "decoded_params": decoded_params,
                },
            )
            recorder = SimulationRecorder(sim_config)
            
        except ImportError as e:
            print(f"  Warning: SimulationRecorder not available, skipping recording: {e}")
            recorder = None
        except Exception as e:
            print(f"  Warning: Failed to create SimulationRecorder: {e}")
            import traceback
            traceback.print_exc()
            recorder = None
        
        try:
            # Build STL spec - need to access the underlying robocasa env
            # GrootRoboCasaEnv -> RoboCasaEnv -> self.env (actual robocasa environment with sim)
            # The gym wrapper stores the robocasa env in self.env, not accessible via unwrapped
            underlying_env = self._get_underlying_robocasa_env(env)
            formula = self._build_stl_spec(underlying_env)
            monitor = STLMonitor(formula)
            
            # Gymnasium reset returns (obs, info)
            obs, info = env.reset()
            
            # Debug: Verify language instruction is in observation (only on first reset)
            lang_keys = ['annotation.human.coarse_action', 'annotation.human.action.task_description', 'language']
            found_lang = None
            for key in lang_keys:
                if key in obs:
                    found_lang = obs[key]
                    print(f"  [DEBUG] Found language instruction in obs['{key}']: {found_lang}")
                    break
            if found_lang is None:
                print(f"  [WARNING] Language instruction not found in observation!")
                print(f"    Available keys: {list(obs.keys())[:20]}...")
            else:
                # Also check underlying env's get_ep_meta
                try:
                    ep_meta = underlying_env.get_ep_meta()
                    ep_lang = ep_meta.get("lang", "NOT FOUND")
                    print(f"  [DEBUG] Underlying env get_ep_meta() lang: {ep_lang}")
                except Exception as e:
                    print(f"  [WARNING] Could not get ep_meta from underlying env: {e}")
            
            # Set initial observation for recorder
            if recorder is not None:
                recorder.set_initial_observation(obs)
            
            trace = []
            final_success = False
            
            for t in range(self.horizon):
                # Get action from policy - should return dict for gymnasium env
                action = self.policy(env, obs)
                
                # Gymnasium step returns (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Record step for offline replay
                if recorder is not None:
                    # Convert action to dict format if needed
                    if isinstance(action, dict):
                        action_dict = action
                    else:
                        # Convert numpy array to dict format
                        action_dict = {"action": action}
                    
                    # Record step (convert to numpy arrays for compatibility)
                    terminations = np.array([terminated], dtype=bool)
                    truncations = np.array([truncated], dtype=bool)
                    recorder.record_step(
                        actions=action_dict,
                        observations=obs,
                        terminations=terminations,
                        truncations=truncations,
                        env_infos=info,
                    )
                
                # Capture original resolution frame from underlying environment for video
                if video_recorder is not None and video_recorder.is_ready():
                    frame_captured = False
                    # Method 1: Get raw camera image from underlying robosuite environment (most reliable)
                    try:
                        # Force update observations to get latest camera image
                        raw_obs = underlying_env._get_observations(force_update=True)
                        camera_name = "egoview"
                        camera_key = f"{camera_name}_image"
                        if camera_key in raw_obs:
                            # Get original resolution frame (H, W, C) uint8
                            frame = raw_obs[camera_key].copy()
                            # Flip vertically (robosuite convention: images are upside down)
                            frame = frame[::-1, :, :]
                            # Ensure RGB format (remove alpha channel if present)
                            if len(frame.shape) == 3:
                                if frame.shape[2] == 4:
                                    frame = frame[:, :, :3]
                            # Ensure uint8 dtype
                            if frame.dtype != np.uint8:
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                            # Write frame to video
                            video_recorder.write_frame(frame)
                            frame_captured = True
                        else:
                            if t == 0:
                                print(f"  Warning: Camera key '{camera_key}' not found in observations")
                                print(f"    Available keys: {list(raw_obs.keys())[:10]}...")
                    except Exception as e:
                        if t == 0:  # Only print warning on first frame
                            print(f"  Warning: Could not get frame from underlying env: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Method 2: Use sim.render() directly (fallback)
                    if not frame_captured:
                        try:
                            # Use underlying robosuite sim.render() method
                            if hasattr(underlying_env, 'sim') and hasattr(underlying_env.sim, 'render'):
                                # Get camera dimensions from env
                                camera_width = 1280 if self.save_video else 128
                                camera_height = 800 if self.save_video else 128
                                rendered_frame = underlying_env.sim.render(
                                    width=camera_width,
                                    height=camera_height,
                                    camera_name="egoview",
                                )
                                if rendered_frame is not None:
                                    # Flip vertically (robosuite convention)
                                    rendered_frame = rendered_frame[::-1, :, :]
                                    # Ensure RGB format
                                    if len(rendered_frame.shape) == 3 and rendered_frame.shape[2] == 4:
                                        rendered_frame = rendered_frame[:, :, :3]
                                    # Ensure uint8 dtype
                                    if rendered_frame.dtype != np.uint8:
                                        if rendered_frame.max() <= 1.0:
                                            rendered_frame = (rendered_frame * 255).astype(np.uint8)
                                        else:
                                            rendered_frame = rendered_frame.astype(np.uint8)
                                    video_recorder.write_frame(rendered_frame)
                                    frame_captured = True
                        except Exception as e:
                            if t == 0:  # Only print warning on first frame
                                print(f"  Warning: Could not get frame from sim.render: {e}")
                    
                    # Method 3: Try env.render() without mode parameter (fallback)
                    if not frame_captured:
                        try:
                            # Some gymnasium wrappers don't accept mode parameter
                            rendered_frame = env.render()
                            if rendered_frame is not None:
                                if isinstance(rendered_frame, np.ndarray):
                                    # Ensure uint8 dtype
                                    if rendered_frame.dtype != np.uint8:
                                        if rendered_frame.max() <= 1.0:
                                            rendered_frame = (rendered_frame * 255).astype(np.uint8)
                                        else:
                                            rendered_frame = rendered_frame.astype(np.uint8)
                                    # Ensure RGB format (remove alpha channel if present)
                                    if len(rendered_frame.shape) == 3:
                                        if rendered_frame.shape[2] == 4:
                                            rendered_frame = rendered_frame[:, :, :3]
                                    # Write frame to video
                                    video_recorder.write_frame(rendered_frame)
                                    frame_captured = True
                        except Exception as e:
                            if t == 0:  # Only print warning on first frame
                                print(f"  Warning: Could not get frame from env.render(): {e}")
                    
                    # Method 4: Fallback to observation dict
                    if not frame_captured:
                        try:
                            frame = self._get_video_frame(obs)
                            if frame is not None:
                                video_recorder.write_frame(frame)
                                frame_captured = True
                        except Exception as e:
                            if t == 0:  # Only print warning on first frame
                                print(f"  Warning: Could not get frame from obs: {e}")
                    
                    if not frame_captured and t == 0:
                        print(f"  Warning: Could not capture video frame - video may be black")
                        print(f"    Try checking if offscreen renderer is enabled")
                
                # Render if enabled
                if self.render:
                    env.render()
                
                # Update monitor with underlying environment (which has sim attribute)
                robustness = monitor.update(underlying_env)
                
                # Check success
                if isinstance(info.get('success'), list):
                    step_success = info.get('success', [False])[0]
                else:
                    step_success = info.get('success', False)
                
                if step_success:
                    final_success = True
                
                trace.append({
                    'time': t,
                    'robustness': robustness,
                    'done': done,
                    'success': step_success
                })
                
                if done:
                    break
            
            final_robustness = monitor.get_final_robustness()
            
            # Stop video recording and rename file with trial info
            if video_recorder is not None and video_recorder.is_ready():
                video_recorder.stop()
                
                # Rename file with trial info, robustness, and success status
                if video_file_path and video_file_path.exists():
                    success_str = "success" if final_success else "fail"
                    rob_str = f"rob{final_robustness:.4f}".replace("-", "neg").replace(".", "p")
                    new_video_filename = f"trial_{trial_idx}_{success_str}_{rob_str}.mp4"
                    new_video_path = self.videos_dir / new_video_filename
                    
                    video_file_path.rename(new_video_path)
                    print(f"  Saved video: {new_video_path.name}")
                else:
                    print(f"  Warning: Video file not found: {video_file_path}")
            
            # Save recording (msgpack format)
            if recorder is not None and recording_path is not None:
                try:
                    # Update recording filename with final results
                    success_str = "success" if final_success else "fail"
                    rob_str = f"rob{final_robustness:.4f}".replace("-", "neg").replace(".", "p")
                    final_recording_filename = f"trial_{trial_idx}_{success_str}_{rob_str}.msgpack"
                    final_recording_path = self.recordings_dir / final_recording_filename
                    
                    # Update metadata with final results
                    recorder.metadata.update({
                        "final_success": final_success,
                        "final_robustness": float(final_robustness),
                    })
                    
                    recorder.save(final_recording_path)
                    if final_recording_path.exists():
                        file_size = final_recording_path.stat().st_size
                        print(f"  Saved recording: {final_recording_path.name} ({file_size / 1024 / 1024:.2f} MB)")
                    else:
                        print(f"  Warning: Recording file was not created: {final_recording_path}")
                except Exception as e:
                    print(f"  Warning: Could not save recording: {e}")
                    import traceback
                    traceback.print_exc()
            
            return final_robustness, trace, decoded_params
            
        except Exception as e:
            # Save recording even if there's an error (if recorder exists)
            if recorder is not None and recording_path is not None:
                try:
                    error_recording_filename = f"trial_{trial_idx}_error.msgpack"
                    error_recording_path = self.recordings_dir / error_recording_filename
                    recorder.metadata.update({
                        "error": str(e),
                        "error_type": type(e).__name__,
                    })
                    recorder.save(error_recording_path)
                    print(f"  Saved error recording: {error_recording_path.name}")
                except Exception as save_error:
                    print(f"  Warning: Could not save error recording: {save_error}")
            raise  # Re-raise the original exception
            
        finally:
            # Ensure video recorder is stopped even if there's an error
            if video_recorder is not None and video_recorder.is_ready():
                video_recorder.stop()
            env.close()
    
    def _get_video_frame(self, obs: Dict) -> Optional[np.ndarray]:
        """Extract video frame from observation."""
        from PIL import Image
        
        video_keys = [
            'video.ego_view_bg_crop_pad_res256_freq20',
            'video.ego_view_pad_res256_freq20',
            'video.ego_view',
        ]
        
        for key in video_keys:
            if key in obs:
                frame = obs[key]
                
                # Handle PIL Image
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                
                # Handle numpy array
                if isinstance(frame, np.ndarray):
                    # Remove batch/time dimensions if present: (T, H, W, C) -> (H, W, C)
                    while frame.ndim > 3:
                        frame = frame[0]
                    
                    # Ensure uint8 format
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    
                    return frame
        
        # Debug: print available keys if no video found
        obs_keys = [k for k in obs.keys() if 'video' in k.lower() or 'image' in k.lower()]
        if obs_keys:
            print(f"  Debug: Found video-related keys: {obs_keys}")
        
        return None
    
    def _save_video(
        self,
        frames: List[np.ndarray],
        trial_idx: int,
        robustness: float,
        success: bool
    ):
        """
        Save video frames to file using PyAV (same method as run_give.py).
        
        Uses high-quality H.264 encoding with proper resolution and aspect ratio.
        """
        if not frames:
            print(f"  Warning: No frames to save for trial {trial_idx}")
            return
        
        # Create filename with trial info
        success_str = "success" if success else "fail"
        rob_str = f"rob{robustness:.4f}".replace("-", "neg").replace(".", "p")
        video_path = self.videos_dir / f"trial_{trial_idx}_{success_str}_{rob_str}.mp4"
        
        # Ensure all frames are uint8 and same shape
        video_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Ensure frame is (H, W, 3) RGB format
            if frame.ndim == 2:
                # Grayscale -> RGB
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                # Single channel -> RGB
                frame = np.repeat(frame, 3, axis=2)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                # RGBA -> RGB
                frame = frame[:, :, :3]
            
            video_frames.append(frame)
        
        if not video_frames:
            print(f"  Warning: No valid frames to save for trial {trial_idx}")
            return
        
        # Use PyAV (same as VideoRecorder in run_give.py) for high-quality encoding
        try:
            import av
            
            # Get frame dimensions from first frame
            first_frame = video_frames[0]
            h, w = first_frame.shape[:2]
            
            # Create container and stream
            container = av.open(str(video_path), mode='w')
            stream = container.add_stream('h264', rate=20)  # 20 fps
            
            # Configure codec for high quality (same as VideoRecorder.create_h264)
            codec_context = stream.codec_context
            codec_context.width = w
            codec_context.height = h
            codec_context.pix_fmt = 'yuv420p'  # Standard format for compatibility
            codec_context.options = {'crf': '18', 'profile:v': 'high'}  # CRF 18 = high quality
            
            # Write frames
            for frame_array in video_frames:
                # Ensure frame matches expected dimensions
                if frame_array.shape[:2] != (h, w):
                    print(f"  Warning: Frame size mismatch, skipping frame")
                    continue
                
                # Convert numpy array to AV VideoFrame (RGB format)
                av_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                av_frame.pts = None  # Let encoder handle timestamps
                
                # Encode and mux
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            
            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
            
            print(f"  Saved video: {video_path.name} ({len(video_frames)} frames, {w}x{h})")
            
        except ImportError:
            print(f"  Warning: PyAV (av) not available, falling back to imageio")
            # Fallback to imageio if PyAV not available
            try:
                import imageio
                imageio.mimsave(
                    str(video_path),
                    video_frames,
                    fps=20,
                    codec='h264',
                )
                print(f"  Saved video (imageio): {video_path.name} ({len(video_frames)} frames)")
            except Exception as e:
                print(f"  Warning: Could not save video: {e}")
        except Exception as e:
            print(f"  Warning: Could not save video with PyAV: {e}")
            import traceback
            traceback.print_exc()
    
    def falsify(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run falsification to find specification violations.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with falsification results
        """
        import nevergrad as ng
        import time
        
        # Set global random seeds for deterministic behavior
        if verbose:
            print(f"Setting random seeds to {self.seed} for deterministic execution...")
        random.seed(self.seed)
        np.random.seed(self.seed)
        if TORCH_AVAILABLE and torch is not None:  # type: ignore
            torch.manual_seed(self.seed)  # type: ignore
            if torch.cuda.is_available():  # type: ignore
                torch.cuda.manual_seed(self.seed)  # type: ignore
                torch.cuda.manual_seed_all(self.seed)  # type: ignore
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
        
        start_time = time.time()
        
        dim = self.param_space.get_dimension()
        param_names = self.param_space.get_parameter_names()
        
        if verbose:
            print(f"=" * 60)
            print(f"CT-Based Falsification")
            print(f"=" * 60)
            print(f"Configuration: {self.config_number}")
            print(f"Task: {self.task_expression}")
            print(f"Parameter space dimension: {dim}")
            print(f"Parameters: {param_names}")
            print(f"Budget: {self.budget}")
            print(f"Output directory: {self.output_dir}")
            print(f"=" * 60)
        
        # Create optimizer
        optimizer = ng.optimizers.NGOpt(
            parametrization=ng.p.Array(shape=(dim,)).set_bounds(0, 1),
            budget=self.budget
        )
        
        iteration = [0]
        violation_found = [False]  # Flag to track if violation found
        
        def objective(params: np.ndarray) -> float:
            """Objective: minimize robustness (find violations)"""
            iteration[0] += 1
            trial_idx = iteration[0]
            
            if verbose:
                print(f"\n--- Trial {trial_idx}/{self.budget} ---")
            
            robustness, trace, decoded = self._rollout(params, trial_idx=trial_idx)
            
            # Record trial
            trial_record = {
                'trial': trial_idx,
                'robustness': float(robustness),
                'params_normalized': params.tolist(),
                'params_decoded': decoded,
                'trace_length': len(trace),
                'final_success': trace[-1]['success'] if trace else False,
            }
            self.all_trials.append(trial_record)
            
            if verbose:
                print(f"  Robustness: {robustness:.4f}")
            
            # Track best (most violating) result
            if robustness < self.best_robustness:
                self.best_robustness = robustness
                self.best_params = params.copy()
                self.best_decoded_params = decoded
                self.falsifying_trace = trace
                
                if verbose:
                    print(f"  ✓ New best robustness!")
            
            # Check if violation found (robustness < 0 means specification violated)
            if robustness < 0:
                violation_found[0] = True
                if verbose:
                    print(f"\n  ⚠️  SPECIFICATION VIOLATION FOUND! (robustness = {robustness:.4f})")
                    print(f"  Stopping early after {trial_idx} trials.")
            
            return robustness
        
        # Run optimization with early stopping
        # Use manual loop to support early stopping when violation found
        try:
            for i in range(self.budget):
                # Check if violation already found
                if violation_found[0]:
                    if verbose:
                        print(f"\nEarly stopping: violation found in trial {iteration[0]}")
                    break
                
                # Ask optimizer for next candidate
                candidate = optimizer.ask()
                
                # Evaluate objective
                loss = objective(candidate.value)
                
                # Tell optimizer the result
                optimizer.tell(candidate, loss)
            
            # Get recommendation
            recommendation = optimizer.provide_recommendation()
            
        except KeyboardInterrupt:
            if verbose:
                print("\n\nOptimization interrupted by user.")
            recommendation = optimizer.provide_recommendation()
        
        elapsed_time = time.time() - start_time
        
        # Compute statistics
        robustness_values = [t['robustness'] for t in self.all_trials]
        success_count = sum(1 for t in self.all_trials if t['final_success'])
        
        result = {
            'config_number': self.config_number,
            'task_expression': self.task_expression,
            'falsified': self.best_robustness < 0,
            'best_robustness': float(self.best_robustness),
            'best_params_normalized': self.best_params.tolist() if self.best_params is not None else None,
            'best_params_decoded': self.best_decoded_params,
            'param_names': param_names,
            'budget': self.budget,
            'iterations': iteration[0],
            'early_stopped': violation_found[0],  # Whether stopped early due to violation
            'elapsed_time': elapsed_time,
            'statistics': {
                'mean_robustness': float(np.mean(robustness_values)) if robustness_values else 0.0,
                'min_robustness': float(np.min(robustness_values)) if robustness_values else 0.0,
                'max_robustness': float(np.max(robustness_values)) if robustness_values else 0.0,
                'success_rate': success_count / len(self.all_trials) if self.all_trials else 0.0,
                'success_count': success_count,
                'total_trials': len(self.all_trials),
            },
            'output_dir': str(self.output_dir),
            'videos_dir': str(self.videos_dir),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save result to file
        result_file = self.output_dir / "result.json"
        with open(result_file, 'w') as f:
            # Don't save trace in main result (too large)
            json.dump(result, f, indent=2, default=str)
        
        # Save all trials to separate file
        trials_file = self.output_dir / "all_trials.json"
        with open(trials_file, 'w') as f:
            json.dump(self.all_trials, f, indent=2, default=str)
        
        if verbose:
            print(f"\n" + "=" * 60)
            print(f"Results")
            print(f"=" * 60)
            print(f"Falsified: {result['falsified']}")
            print(f"Best robustness: {result['best_robustness']:.4f}")
            print(f"Trials run: {result['iterations']}/{result['budget']}")
            if result['early_stopped']:
                print(f"Early stopped: ✓ (violation found)")
            print(f"Success rate: {result['statistics']['success_rate']:.2%}")
            print(f"Elapsed time: {elapsed_time:.2f}s")
            if result['best_params_decoded']:
                print(f"Best parameters:")
                for name, value in result['best_params_decoded'].items():
                    print(f"  {name}: {value}")
            print(f"\nResults saved to: {result_file}")
            print(f"All trials saved to: {trials_file}")
            print(f"Videos saved to: {self.videos_dir}")
        
        return result


def create_policy_function(
    host: str = "localhost",
    port: int = 5555,
    use_random: bool = False,
) -> Callable[[Any, Any], Any]:
    """
    Create a policy function for use with the falsifier.
    
    The policy function takes (env, obs) and returns single-step dictionary actions
    compatible with GrootRoboCasaEnv's step() method.
    
    Args:
        host: Inference server host
        port: Inference server port
        use_random: If True, return random actions instead of using server
        
    Returns:
        Policy function: action_dict = policy(env, obs)
    """
    if use_random:
        def random_policy(env, obs):
            """Random policy returning single-step dictionary actions."""
            return {
                'action.left_arm': np.random.uniform(-1, 1, size=7),
                'action.left_hand': np.random.uniform(-1, 1, size=6),
                'action.right_arm': np.random.uniform(-1, 1, size=7),
                'action.right_hand': np.random.uniform(-1, 1, size=6),
                'action.waist': np.random.uniform(-1, 1, size=3),
            }
        return random_policy
    
    # Use inference server
    from gr00t.eval.service import ExternalRobotInferenceClient
    policy_client = ExternalRobotInferenceClient(host=host, port=port)
    
    # Track if language instruction has been printed (using closure variable)
    lang_printed = [False]  # Use list to allow modification in nested function
    
    def server_policy(env, obs):
        """
        Policy that queries inference server and returns single-step actions.
        
        Handles observation format conversion:
        - Renames video keys to match server expectation (video.ego_view)
        - Adds time dimension: (H, W, C) -> (1, H, W, C)
        - Ensures dtype is uint8
        - Extracts single timestep from action chunks
        """
        try:
            if not isinstance(obs, dict):
                raise ValueError(f"Unexpected observation type: {type(obs)}")
            
            obs_copy = {}
            
            # Process video: rename to video.ego_view and add time dimension
            # Server expects shape [T, H, W, C] with dtype uint8
            video_key_mappings = [
                'video.ego_view_bg_crop_pad_res256_freq20',
                'video.ego_view_pad_res256_freq20',
            ]
            video_found = False
            for old_key in video_key_mappings:
                if old_key in obs:
                    val = obs[old_key]
                    # Convert PIL Image to numpy if needed
                    from PIL import Image
                    if isinstance(val, Image.Image):
                        val = np.array(val)
                    # Ensure uint8
                    if isinstance(val, np.ndarray):
                        if val.dtype != np.uint8:
                            if val.max() <= 1.0:
                                val = (val * 255).astype(np.uint8)
                            else:
                                val = val.astype(np.uint8)
                        # Add time dimension: (H, W, C) -> (1, H, W, C)
                        if val.ndim == 3:
                            val = np.expand_dims(val, axis=0)
                    obs_copy['video.ego_view'] = val
                    video_found = True
                    break
            
            if not video_found:
                raise ValueError(f"No video key found. Available keys: {list(obs.keys())}")
            
            # Copy state keys
            for key in obs:
                if key.startswith('state.'):
                    val = obs[key]
                    # Add time dimension for states too: (D,) -> (1, D)
                    if isinstance(val, np.ndarray) and val.ndim == 1:
                        val = np.expand_dims(val, axis=0)
                    obs_copy[key] = val
            
            # Copy language instruction
            # FourierGr1ArmsWaistDataConfig expects 'annotation.human.coarse_action'
            # Must be passed as a LIST with one string element - unsqueeze_dict_values will then 
            # convert it to np.array and add batch dimension: ["text"] -> shape (1, 1)
            # GrootRoboCasaEnv.get_groot_observation() should have already converted 'language' 
            # to 'annotation.human.coarse_action' with "unlocked_waist: " prefix
            lang_str = None
            if 'annotation.human.coarse_action' in obs:
                lang = obs['annotation.human.coarse_action']
                if isinstance(lang, list):
                    lang_str = lang[0] if lang else None
                else:
                    lang_str = str(lang)
                # Use the language instruction from observation (already has "unlocked_waist: " prefix)
                obs_copy['annotation.human.coarse_action'] = [lang_str]  # Wrap in list
            elif 'annotation.human.action.task_description' in obs:
                lang = obs['annotation.human.action.task_description']
                if isinstance(lang, list):
                    lang_str = lang[0] if lang else None
                else:
                    lang_str = str(lang)
                # Add "unlocked_waist: " prefix for GR1ArmsAndWaist robot
                obs_copy['annotation.human.coarse_action'] = [f"unlocked_waist: {lang_str}"]
            elif 'language' in obs:
                # Fallback: if 'language' key exists but not converted yet
                lang = obs['language']
                if isinstance(lang, list):
                    lang_str = lang[0] if lang else None
                else:
                    lang_str = str(lang)
                # Add "unlocked_waist: " prefix for GR1ArmsAndWaist robot
                obs_copy['annotation.human.coarse_action'] = [f"unlocked_waist: {lang_str}"]
            else:
                # Default instruction - as list (should not happen if language instruction is properly set)
                print(f"  [WARNING] Language instruction not found in observation, using default")
                print(f"    Available keys: {list(obs.keys())[:20]}...")
                obs_copy['annotation.human.coarse_action'] = ["unlocked_waist: complete the task"]
            
            # Debug: Print language instruction being used (only on first call)
            if not lang_printed[0]:
                if lang_str:
                    print(f"  [DEBUG] Using language instruction: '{lang_str}'")
                else:
                    print(f"  [WARNING] Language instruction is None or empty!")
                lang_printed[0] = True
            
            # Get action from inference server
            action_dict = policy_client.get_action(obs_copy)
            
            # Extract single timestep from action chunks: (T, D) -> (D,)
            if isinstance(action_dict, dict):
                single_step_action = {}
                for key, value in action_dict.items():
                    if key.startswith('action.'):
                        if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                            single_step_action[key] = value[0]
                        else:
                            single_step_action[key] = value
                    else:
                        single_step_action[key] = value
                if single_step_action:
                    return single_step_action
            
            return action_dict
            
        except Exception as e:
            print(f"Policy call failed: {e}")
            # Return zero actions on failure
            return {
                'action.left_arm': np.zeros(7),
                'action.left_hand': np.zeros(6),
                'action.right_arm': np.zeros(7),
                'action.right_hand': np.zeros(6),
                'action.waist': np.zeros(3),
            }
    
    return server_policy


def falsify_ct_configuration(
    config_number: int,
    policy: Optional[Callable[[Any, Any], Any]] = None,
    horizon: int = 300,
    budget: int = 50,
    config_path: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
    render: bool = False,
    policy_host: str = "localhost",
    policy_port: int = 5555,
    use_random_policy: bool = False,
    output_dir: Optional[str] = None,
    save_video: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to falsify a CT configuration.
    
    Args:
        config_number: Configuration number from CT JSON (1-based)
        policy: Policy function. If None, creates one using policy_host/port or random.
        horizon: Time horizon for rollouts
        budget: Optimization budget
        config_path: Optional path to CT configurations JSON
        seed: Random seed
        verbose: Whether to print progress
        render: Whether to enable rendering (show visual window)
        policy_host: Inference server host (if policy not provided)
        policy_port: Inference server port (if policy not provided)
        use_random_policy: Use random policy instead of inference server
        output_dir: Directory to store results (videos, recordings, logs)
        save_video: Whether to save videos of rollouts
        
    Returns:
        Dictionary with falsification results
    """
    # Create policy if not provided
    if policy is None:
        policy = create_policy_function(
            host=policy_host,
            port=policy_port,
            use_random=use_random_policy
        )
    
    falsifier = CTBasedFalsifier(
        config_number=config_number,
        policy=policy,
        horizon=horizon,
        budget=budget,
        config_path=config_path,
        seed=seed,
        render=render,
        output_dir=output_dir,
        save_video=save_video,
    )
    
    return falsifier.falsify(verbose=verbose)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI interface for falsification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="STL-based Falsification for Robot Tasks")
    parser.add_argument(
        "--config-number", "-n",
        type=int,
        help="CT configuration number to falsify (1-based)"
    )
    parser.add_argument(
        "--budget", "-b",
        type=int,
        default=50,
        help="Optimization budget (default: 50)"
    )
    parser.add_argument(
        "--horizon", "-H",
        type=int,
        default=300,
        help="Rollout horizon (default: 300)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--config-path", "-c",
        type=str,
        default=None,
        help="Path to CT configurations JSON"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available configurations"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run standalone demo (no environment required)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--run", "-r",
        action="store_true",
        help="Actually run falsification (requires policy or inference server)"
    )
    parser.add_argument(
        "--policy-host",
        type=str,
        default="localhost",
        help="Inference server host (default: localhost)"
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=5555,
        help="Inference server port (default: 5555)"
    )
    parser.add_argument(
        "--use-random-policy",
        action="store_true",
        help="Use random policy instead of inference server (for testing)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (show visual window)"
    )
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default=None,
        help="Directory to store results (default: ./falsify_results/config_N)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video saving"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        example_standalone()
        return
    
    if args.list:
        loader = CTConfigurationLoader(args.config_path)
        print(f"Total configurations: {loader.get_total_configurations()}")
        print("\nFirst 10 configurations:")
        for i in range(1, min(11, loader.get_total_configurations() + 1)):
            config = loader.get_configuration(i)
            print(f"  {i}: {config.get('task_expression', 'N/A')}")
        return
    
    if args.config_number is None:
        # Run standalone demo if no config specified
        example_standalone()
        return
    
    # Show parameter space info
    loader = CTConfigurationLoader(args.config_path)
    param_space = loader.extract_parameter_space(args.config_number)
    
    print(f"\nConfiguration {args.config_number}:")
    print(f"  Task: {loader.get_task_expression(args.config_number)}")
    print(f"  Parameter space dimension: {param_space.get_dimension()}")
    print(f"  Parameters: {param_space.get_parameter_names()}")
    
    # If --run flag is set, actually run falsification
    if args.run:
        import json
        
        # Create policy function using helper
        if args.use_random_policy:
            print(f"\nUsing random policy for testing...")
            policy_fn = create_policy_function(use_random=True)
        else:
            try:
                print(f"\nConnecting to inference server at {args.policy_host}:{args.policy_port}...")
                policy_fn = create_policy_function(
                    host=args.policy_host,
                    port=args.policy_port,
                    use_random=False
                )
                print("Connected successfully!")
            except Exception as e:
                print(f"\nError: Failed to connect to inference server: {e}")
                print("\nTo use a random policy for testing, use --use-random-policy flag.")
                print("Otherwise, please ensure the inference server is running:")
                print(f"  python scripts/inference_service.py --server --model_path <MODEL_PATH> --port {args.policy_port}")
                raise
        
        # Determine output directory
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = f"./falsify_results/config_{args.config_number}"
        
        # Run falsification using convenience function
        print(f"\nStarting falsification...")
        print(f"  Budget: {args.budget}")
        print(f"  Horizon: {args.horizon}")
        print(f"  Seed: {args.seed}")
        print(f"  Output directory: {output_dir}")
        print(f"  Save videos: {not args.no_video}")
        
        try:
            result = falsify_ct_configuration(
                config_number=args.config_number,
                policy=policy_fn,
                horizon=args.horizon,
                budget=args.budget,
                config_path=args.config_path,
                seed=args.seed,
                verbose=True,
                render=args.render,
                output_dir=output_dir,
                save_video=not args.no_video,
            )
            
            # Save results if additional output file specified (also saved automatically)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nAdditional results saved to {args.output}")
            
            # Print summary
            print(f"\n" + "=" * 60)
            print(f"Falsification Summary")
            print(f"=" * 60)
            print(f"Falsified: {result['falsified']}")
            print(f"Best robustness: {result['best_robustness']:.6f}")
            if result['falsified']:
                print(f"\n⚠️  Specification violation found!")
                print(f"\nBest parameters that violate the spec:")
                if result['best_params_decoded']:
                    for name, value in result['best_params_decoded'].items():
                        print(f"  {name}: {value}")
            else:
                print(f"\n✓ No specification violations found within budget.")
            
        except KeyboardInterrupt:
            print("\n\nFalsification interrupted by user.")
            return
        except Exception as e:
            import traceback
            print(f"\nError during falsification: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            return
    else:
        # Just show info without running
        print(f"\nTo run falsification, use --run flag:")
        print(f"  python falsifier/falsify.py --config-number {args.config_number} --run")
        print(f"\nOptions:")
        print(f"  --use-random-policy     Use random policy (for testing)")
        print(f"  --policy-host HOST     Inference server host (default: localhost)")
        print(f"  --policy-port PORT     Inference server port (default: 5555)")
        print(f"  --budget N             Optimization budget (default: 50)")
        print(f"  --horizon H            Rollout horizon (default: 300)")


if __name__ == "__main__":
    main()

