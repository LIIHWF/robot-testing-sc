"""
STL-based Falsification Framework (Simplified)

This module provides a clean falsification approach that:
1. Uses ConfigurationInstantiator for base configuration
2. Uses ConfigurationExecutor for environment creation
3. Applies small perturbations using nevergrad optimization
4. Uses STL robustness as the optimization objective

Usage:
    python new_falsify.py --config-number 1 --budget 50 --run
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import sys
import json
import random
import numpy as np

# Add parent directory to path for imports when running as script
_this_dir = Path(__file__).parent
_parent_dir = _this_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    import torch as _torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# STL SPECIFICATION TYPES
# =============================================================================

@dataclass
class AtomicPredicate:
    """
    Atomic predicate: f(s) > 0
    
    Robustness value is f(s):
    - f(s) > 0: satisfied
    - f(s) < 0: violated
    """
    name: str
    func: Callable[[Any], float]
    description: str = ""
    
    def evaluate(self, state: Any) -> float:
        return self.func(state)
    
    def to_rtamt_var(self) -> str:
        return self.name.replace("(", "_").replace(")", "").replace(",", "_").replace(" ", "")


@dataclass
class STLFormula:
    """STL Formula representation."""
    operator: Optional[str] = None
    operands: List[Any] = field(default_factory=list)  # STLFormula | AtomicPredicate
    time_bounds: Tuple[float, float] = (0, float('inf'))
    
    @classmethod
    def atomic(cls, predicate: AtomicPredicate) -> 'STLFormula':
        return cls(operator=None, operands=[predicate])
    
    @classmethod
    def eventually(cls, phi: 'STLFormula', time_bounds: Tuple[float, float]) -> 'STLFormula':
        return cls(operator='eventually', operands=[phi], time_bounds=time_bounds)
    
    @classmethod
    def globally(cls, phi: 'STLFormula', time_bounds: Tuple[float, float]) -> 'STLFormula':
        return cls(operator='globally', operands=[phi], time_bounds=time_bounds)
    
    @classmethod
    def and_(cls, *operands: 'STLFormula') -> 'STLFormula':
        return cls(operator='and', operands=list(operands))
    
    @classmethod
    def or_(cls, *operands: 'STLFormula') -> 'STLFormula':
        return cls(operator='or', operands=list(operands))
    
    @classmethod
    def not_(cls, phi: 'STLFormula') -> 'STLFormula':
        return cls(operator='not', operands=[phi])
    
    def get_atomic_predicates(self) -> List[AtomicPredicate]:
        predicates = []
        for op in self.operands:
            if isinstance(op, AtomicPredicate):
                predicates.append(op)
            elif isinstance(op, STLFormula):
                predicates.extend(op.get_atomic_predicates())
        return predicates
    
    def to_rtamt_spec(self) -> str:
        if self.operator is None:
            return f"({self.operands[0].to_rtamt_var()} >= 0)"
        elif self.operator == 'and':
            return "(" + " and ".join(op.to_rtamt_spec() for op in self.operands) + ")"
        elif self.operator == 'or':
            return "(" + " or ".join(op.to_rtamt_spec() for op in self.operands) + ")"
        elif self.operator == 'not':
            return f"(not {self.operands[0].to_rtamt_spec()})"
        elif self.operator == 'eventually':
            a, b = self.time_bounds
            b_str = str(int(b)) if b != float('inf') else "inf"
            return f"eventually[{int(a)},{b_str}] {self.operands[0].to_rtamt_spec()}"
        elif self.operator == 'globally':
            a, b = self.time_bounds
            b_str = str(int(b)) if b != float('inf') else "inf"
            return f"always[{int(a)},{b_str}] {self.operands[0].to_rtamt_spec()}"
        raise ValueError(f"Unknown operator: {self.operator}")


# =============================================================================
# STL MONITOR
# =============================================================================

class STLMonitor:
    """Monitors STL specifications using rtamt."""
    
    def __init__(self, formula: STLFormula):
        import rtamt
        
        self.formula = formula
        self.predicates = formula.get_atomic_predicates()
        self.spec = rtamt.StlDiscreteTimeSpecification()
        
        for pred in self.predicates:
            self.spec.declare_var(pred.to_rtamt_var(), 'float')
        
        self.spec.spec = formula.to_rtamt_spec()
        self.spec.parse()
        self.spec.pastify()
        
        self.time_step = 0
        self.robustness_trace = []
    
    def reset(self):
        self.__init__(self.formula)
    
    def update(self, state: Any) -> float:
        signal_values = [(pred.to_rtamt_var(), pred.evaluate(state)) for pred in self.predicates]
        robustness = self.spec.update(self.time_step, signal_values)
        self.robustness_trace.append(robustness)
        self.time_step += 1
        return float(robustness) if robustness is not None else 0.0
    
    def get_final_robustness(self) -> float:
        return self.robustness_trace[-1] if self.robustness_trace else float('-inf')


# =============================================================================
# ROBUSTNESS EXTRACTORS
# =============================================================================

class RobustnessExtractor:
    """Extracts robustness values from environment state."""
    
    @staticmethod
    def get_object_position(env, obj_name: str) -> np.ndarray:
        """Get position of object or fixture."""
        # Check fixtures
        if obj_name in ("drawer", "cabinet", "microwave", "counter"):
            fixture = RobustnessExtractor._get_fixture(env, obj_name)
            if fixture is not None:
                return np.array(fixture.pos)
            return np.array([0.0, 0.0, 0.92])
        
        # Check obj_body_id
        if hasattr(env, 'obj_body_id') and obj_name in env.obj_body_id:
            return np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        
        # Check env.objects
        if hasattr(env, 'objects') and obj_name in env.objects:
            obj = env.objects[obj_name]
            if hasattr(obj, 'root_body'):
                body_id = env.sim.model.body_name2id(obj.root_body)
                return np.array(env.sim.data.body_xpos[body_id])
        
        return np.array([0.0, 0.0, 0.92])
    
    @staticmethod
    def _get_fixture(env, fixture_name: str):
        """Get fixture from environment."""
        if hasattr(env, 'fixtures'):
            if fixture_name in env.fixtures:
                return env.fixtures[fixture_name]
            for name, fxtr in env.fixtures.items():
                if fixture_name.lower() in name.lower():
                    return fxtr
        return None
    
    @staticmethod
    def gripper_distance(env, obj_name: str, side: str = "right") -> float:
        """Distance from gripper to object."""
        obj_pos = RobustnessExtractor.get_object_position(env, obj_name)
        gripper_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id[side]]
        return float(np.linalg.norm(gripper_pos - obj_pos))
    
    @staticmethod
    def gripper_far(env, obj_name: str, threshold: float = 0.25) -> float:
        """Robustness: gripper is far from object."""
        left_dist = RobustnessExtractor.gripper_distance(env, obj_name, "left")
        right_dist = RobustnessExtractor.gripper_distance(env, obj_name, "right")
        return min(left_dist - threshold, right_dist - threshold)
    
    @staticmethod
    def obj_in_receptacle(env, obj_name: str, recep_name: str, threshold: float = 0.15) -> float:
        """Robustness: object is in receptacle."""
        obj_pos = RobustnessExtractor.get_object_position(env, obj_name)
        recep_pos = RobustnessExtractor.get_object_position(env, recep_name)
        horiz_dist = float(np.linalg.norm(obj_pos[:2] - recep_pos[:2]))
        return threshold - horiz_dist
    
    @staticmethod
    def door_state(env, fixture_name: str, target: str = "closed", threshold: float = 0.005) -> float:
        """Robustness: door is in target state."""
        fixture = RobustnessExtractor._get_fixture(env, fixture_name)
        if fixture is None:
            return -1.0
        try:
            state = fixture.get_door_state(env=env)["door"]
            if target == "closed":
                return threshold - state
            return state - (1.0 - threshold)
        except:
            return -1.0


# =============================================================================
# STL SPEC BUILDER (from success_plan)
# =============================================================================

class STLSpecBuilder:
    """Builds STL specification from task success criteria."""
    
    @staticmethod
    def build(env, success_plan: List[Dict], horizon: int) -> STLFormula:
        """Build STL formula from success_plan."""
        predicates = []
        
        for criterion in success_plan:
            crit_type = criterion.get('type', '')
            params = criterion.get('params', {})
            
            if crit_type == 'obj_in_receptacle':
                obj = params.get('obj_name', 'obj')
                recep = params.get('receptacle_name', 'container')
                predicates.append(AtomicPredicate(
                    name=f"in_{recep}",
                    func=lambda s, e=env, o=obj, r=recep: RobustnessExtractor.obj_in_receptacle(e, o, r),
                ))
            
            elif crit_type == 'gripper_far':
                obj = params.get('obj_name', 'obj')
                predicates.append(AtomicPredicate(
                    name="gripper_far",
                    func=lambda s, e=env, o=obj: RobustnessExtractor.gripper_far(e, o),
                ))
            
            elif crit_type == 'door_closed':
                fixture = params.get('fixture_name', 'drawer')
                predicates.append(AtomicPredicate(
                    name=f"{fixture}_closed",
                    func=lambda s, e=env, f=fixture: RobustnessExtractor.door_state(e, f, "closed"),
                ))
            
            elif crit_type == 'door_open':
                fixture = params.get('fixture_name', 'drawer')
                predicates.append(AtomicPredicate(
                    name=f"{fixture}_open",
                    func=lambda s, e=env, f=fixture: RobustnessExtractor.door_state(e, f, "open"),
                ))
        
        # Default predicate if none specified
        if not predicates:
            predicates.append(AtomicPredicate(
                name="default",
                func=lambda s: 0.0,
            ))
        
        # Combine: eventually(AND of all predicates)
        if len(predicates) == 1:
            success = STLFormula.atomic(predicates[0])
        else:
            success = STLFormula.and_(*[STLFormula.atomic(p) for p in predicates])
        
        return STLFormula.eventually(success, (0, horizon))


# =============================================================================
# PERTURBATION SPACE
# =============================================================================

@dataclass
class PerturbationSpace:
    """Defines perturbation ranges for falsification."""
    position_range: float = 0.05      # ±5cm (conservative to avoid placement issues)
    rotation_range: float = 0.3       # ±0.3 rad (~17°)
    door_state_range: float = 0.0     # Disabled - door state perturbation causes fixture placement issues
    
    def get_dimension(self, config: Dict[str, Any]) -> int:
        """Count perturbable parameters."""
        dim = 0
        for obj in config.get('object_plan', []):
            obj_name = obj.get('name', '')
            # Skip fixed objects (target containers)
            if obj_name in self.FIXED_OBJECTS:
                continue
            if 'placement' in obj:
                if 'pos' in obj['placement']:
                    dim += 2  # x, y
                if 'rotation' in obj['placement']:
                    dim += 1
        
        # Only count door state if perturbation is enabled
        if self.door_state_range > 0:
            fixture_req = config.get('fixture_plan', {}).get('_fixture_requirements', {})
            for req in fixture_req.values():
                if 'door_state_min' in req:
                    dim += 1
        
        return max(dim, 1)  # At least 1 dimension
    
    # Objects that should NOT be perturbed (target containers stay centered)
    FIXED_OBJECTS = {'basket', 'plate', 'bowl'}
    
    def apply(self, config: Dict[str, Any], params: np.ndarray) -> Dict[str, Any]:
        """Apply normalized params [0,1] as perturbations."""
        result = deepcopy(config)
        idx = 0
        
        # Perturb object positions/rotations
        for obj in result.get('object_plan', []):
            if 'placement' not in obj:
                continue
            
            obj_name = obj.get('name', '')
            
            # Skip perturbation for fixed objects (target containers)
            if obj_name in self.FIXED_OBJECTS:
                continue
            
            placement = obj['placement']
            
            if 'pos' in placement and idx + 1 < len(params):
                pos = placement['pos']
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    dx = (params[idx] - 0.5) * 2 * self.position_range
                    dy = (params[idx + 1] - 0.5) * 2 * self.position_range
                    placement['pos'] = (pos[0] + dx, pos[1] + dy)
                    idx += 2
            
            if 'rotation' in placement and idx < len(params):
                rot = placement.get('rotation', 0.0)
                drot = (params[idx] - 0.5) * 2 * self.rotation_range
                placement['rotation'] = rot + drot
                idx += 1
        
        # Perturb fixture states (only if enabled)
        if self.door_state_range > 0:
            fixture_req = result.get('fixture_plan', {}).get('_fixture_requirements', {})
            for req in fixture_req.values():
                if 'door_state_min' in req and idx < len(params):
                    base = (req['door_state_min'] + req['door_state_max']) / 2
                    delta = (params[idx] - 0.5) * 2 * self.door_state_range
                    new_state = max(0.0, min(1.0, base + delta))
                    req['door_state_min'] = new_state
                    req['door_state_max'] = new_state
                    idx += 1
        
        return result


# =============================================================================
# FALSIFIER
# =============================================================================

class Falsifier:
    """
    STL-based falsifier.
    
    Workflow:
    1. Load base config via ConfigurationInstantiator
    2. Apply perturbations
    3. Create environment via executor
    4. Run rollout, compute STL robustness
    5. Use nevergrad to minimize robustness (find violations)
    """
    
    def __init__(
        self,
        config_number: int,
        policy: Callable[[Any, Any], Any],
        horizon: int = 300,
        budget: int = 50,
        seed: int = 42,
        output_dir: Optional[str] = None,
        save_video: bool = True,
        config_path: Optional[str] = None,
    ):
        self.config_number = config_number
        self.policy = policy
        self.horizon = horizon
        self.budget = budget
        self.seed = seed
        self.save_video = save_video
        self.config_path = config_path
        
        # Output directory
        if output_dir is None:
            output_dir = f"./falsify_results/config_{config_number}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        
        # Load base configuration
        from .instantiate import ConfigurationInstantiator
        self.instantiator = ConfigurationInstantiator(config_path=config_path)
        self.base_config = self.instantiator.instantiate(config_number, seed=seed)
        self.task_expression = self.base_config.get('task_expression', '')
        
        # Perturbation space
        self.perturb_space = PerturbationSpace()
        
        # Results
        self.best_robustness = float('inf')
        self.best_params = None
        self.all_trials = []
        self._trial_id = 0
    
    def _set_seeds(self):
        """Set random seeds."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            if TORCH_AVAILABLE:
                import torch as th
                th.manual_seed(self.seed)
                if th.cuda.is_available():
                    th.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass
    
    def _create_env(self, config: Dict[str, Any]):
        """Create gymnasium environment from config."""
        import gymnasium as gym
        import time
        from .executor import DynamicEnvironmentFactory, ConfigurationExecutor
        
        self._trial_id += 1
        trial_id = int(time.time() * 1000) % 100000
        class_name = f"Falsify{self.config_number}_{trial_id}_{self._trial_id}"
        
        EnvClass = DynamicEnvironmentFactory.create_environment_class(config, class_name=class_name)
        
        executor = ConfigurationExecutor()
        executor._register_environment(EnvClass, class_name)
        
        env_id = f"gr1_unified/{class_name}_GR1ArmsAndWaistFourierHands_Env"
        return gym.make(
            env_id,
            enable_render=self.save_video,
            camera_widths=1280 if self.save_video else None,
            camera_heights=800 if self.save_video else None,
        )
    
    def _get_robocasa_env(self, env):
        """Unwrap to get underlying robocasa environment with sim."""
        current = env
        while hasattr(current, 'env'):
            inner = getattr(current, 'env', None)
            if inner is None:
                break
            if hasattr(inner, 'sim'):
                return inner
            current = inner
        if hasattr(current, 'sim'):
            return current
        if hasattr(env, 'unwrapped'):
            uw = env.unwrapped
            if hasattr(uw, 'sim'):
                return uw
            if hasattr(uw, 'env') and hasattr(uw.env, 'sim'):
                return uw.env
        return env
    
    def _rollout(self, params: np.ndarray, trial_idx: int) -> Tuple[float, bool, List[Dict]]:
        """Execute one rollout with perturbations."""
        # First trial uses base config without perturbation to verify it works
        if trial_idx == 1:
            print("  (Using base config without perturbation)")
            perturbed = deepcopy(self.base_config)
        else:
            perturbed = self.perturb_space.apply(self.base_config, params)
        
        # Create environment
        try:
            env = self._create_env(perturbed)
        except Exception as e:
            print(f"  Error creating environment: {e}")
            # Return high robustness (penalty) for invalid configurations
            return 1000.0, False, []
        
        try:
            robocasa_env = self._get_robocasa_env(env)
            
            # Build STL spec
            success_plan = perturbed.get('success_plan', [])
            formula = STLSpecBuilder.build(robocasa_env, success_plan, self.horizon)
            monitor = STLMonitor(formula)
            
            # Rollout
            obs, info = env.reset()
            trace = []
            final_success = False
            
            for t in range(self.horizon):
                action = self.policy(env, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                robustness = monitor.update(robocasa_env)
                
                success = info.get('success', [False])
                if isinstance(success, list):
                    success = success[0]
                if success:
                    final_success = True
                
                trace.append({
                    'time': t,
                    'robustness': robustness,
                    'success': success,
                })
                
                if terminated or truncated:
                    break
            
            return monitor.get_final_robustness(), final_success, trace
            
        finally:
            env.close()
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run falsification."""
        import nevergrad as ng
        import time
        
        self._set_seeds()
        start_time = time.time()
        
        dim = self.perturb_space.get_dimension(self.base_config)
        
        if verbose:
            print("=" * 60)
            print(f"Falsification: Config #{self.config_number}")
            print(f"Task: {self.task_expression}")
            print(f"Perturbation dim: {dim}, Budget: {self.budget}")
            print("=" * 60)
        
        # Validate base configuration first
        if verbose:
            print("\nValidating base configuration...")
        try:
            test_env = self._create_env(self.base_config)
            test_env.close()
            if verbose:
                print("  ✓ Base configuration valid")
        except Exception as e:
            if verbose:
                print(f"  ✗ Base configuration INVALID: {e}")
            return {
                'config_number': self.config_number,
                'task_expression': self.task_expression,
                'falsified': False,
                'best_robustness': float('inf'),
                'best_params': None,
                'trials_run': 0,
                'budget': self.budget,
                'elapsed_seconds': time.time() - start_time,
                'output_dir': str(self.output_dir),
                'error': f"Base configuration invalid: {e}",
                'timestamp': datetime.now().isoformat(),
            }
        
        optimizer = ng.optimizers.NGOpt(
            parametrization=ng.p.Array(shape=(dim,)).set_bounds(0, 1),
            budget=self.budget,
        )
        
        violation_found = False
        trial_count = 0
        
        for _ in range(self.budget):
            if violation_found:
                break
            
            trial_count += 1
            candidate = optimizer.ask()
            
            if verbose:
                print(f"\n--- Trial {trial_count}/{self.budget} ---")
            
            robustness, success, trace = self._rollout(candidate.value, trial_count)
            
            self.all_trials.append({
                'trial': trial_count,
                'robustness': float(robustness),
                'success': success,
                'params': candidate.value.tolist(),
            })
            
            if verbose:
                status = "✓" if success else "✗"
                print(f"  Robustness: {robustness:.4f}, Success: {status}")
            
            if robustness < self.best_robustness:
                self.best_robustness = robustness
                self.best_params = candidate.value.copy()
                if verbose:
                    print("  → New best!")
            
            if robustness < 0:
                violation_found = True
                if verbose:
                    print(f"\n  ⚠️ VIOLATION FOUND! (robustness={robustness:.4f})")
            
            optimizer.tell(candidate, robustness)
        
        elapsed = time.time() - start_time
        
        result = {
            'config_number': self.config_number,
            'task_expression': self.task_expression,
            'falsified': self.best_robustness < 0,
            'best_robustness': float(self.best_robustness),
            'best_params': self.best_params.tolist() if self.best_params is not None else None,
            'trials_run': trial_count,
            'budget': self.budget,
            'elapsed_seconds': elapsed,
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save results
        with open(self.output_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2)
        with open(self.output_dir / "all_trials.json", 'w') as f:
            json.dump(self.all_trials, f, indent=2)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Results")
            print("=" * 60)
            print(f"Falsified: {result['falsified']}")
            print(f"Best robustness: {result['best_robustness']:.4f}")
            print(f"Trials: {trial_count}/{self.budget}")
            print(f"Time: {elapsed:.1f}s")
            print(f"Output: {self.output_dir}")
        
        return result


# =============================================================================
# POLICY FACTORY
# =============================================================================

def create_policy(
    host: str = "localhost",
    port: int = 5555,
    use_random: bool = False,
) -> Callable[[Any, Any], Any]:
    """Create a policy function."""
    
    if use_random:
        def random_policy(env, obs):
            return {
                'action.left_arm': np.random.uniform(-1, 1, size=7),
                'action.left_hand': np.random.uniform(-1, 1, size=6),
                'action.right_arm': np.random.uniform(-1, 1, size=7),
                'action.right_hand': np.random.uniform(-1, 1, size=6),
                'action.waist': np.random.uniform(-1, 1, size=3),
            }
        return random_policy
    
    # Server-based policy
    from gr00t.eval.service import ExternalRobotInferenceClient
    client = ExternalRobotInferenceClient(host=host, port=port)
    lang_printed = [False]
    
    def server_policy(env, obs):
        try:
            from PIL import Image
            obs_copy = {}
            
            # Process video
            video_keys = [
                'video.ego_view_bg_crop_pad_res256_freq20',
                'video.ego_view_pad_res256_freq20',
            ]
            for key in video_keys:
                if key in obs:
                    val = obs[key]
                    if isinstance(val, Image.Image):
                        val = np.array(val)
                    if isinstance(val, np.ndarray):
                        if val.dtype != np.uint8:
                            val = (val * 255).astype(np.uint8) if val.max() <= 1.0 else val.astype(np.uint8)
                        if val.ndim == 3:
                            val = np.expand_dims(val, axis=0)
                    obs_copy['video.ego_view'] = val
                    break
            
            # Copy state keys
            for key in obs:
                if key.startswith('state.'):
                    val = obs[key]
                    if isinstance(val, np.ndarray) and val.ndim == 1:
                        val = np.expand_dims(val, axis=0)
                    obs_copy[key] = val
            
            # Language instruction
            lang_str = None
            for lang_key in ['annotation.human.coarse_action', 'annotation.human.action.task_description', 'language']:
                if lang_key in obs:
                    lang = obs[lang_key]
                    lang_str = lang[0] if isinstance(lang, list) else str(lang)
                    break
            
            if lang_str and not lang_str.startswith('unlocked_waist:'):
                lang_str = f"unlocked_waist: {lang_str}"
            obs_copy['annotation.human.coarse_action'] = [lang_str or "unlocked_waist: complete the task"]
            
            if not lang_printed[0]:
                print(f"  [Policy] Language: {lang_str}")
                lang_printed[0] = True
            
            # Get action
            action_dict = client.get_action(obs_copy)
            
            # Extract single timestep
            if isinstance(action_dict, dict):
                single_action = {}
                for key, value in action_dict.items():
                    if key.startswith('action.') and isinstance(value, np.ndarray) and len(value.shape) >= 2:
                        single_action[key] = value[0]
                    else:
                        single_action[key] = value
                if single_action:
                    return single_action
            
            return action_dict
            
        except Exception as e:
            print(f"  [Policy] Error: {e}")
            return {
                'action.left_arm': np.zeros(7),
                'action.left_hand': np.zeros(6),
                'action.right_arm': np.zeros(7),
                'action.right_hand': np.zeros(6),
                'action.waist': np.zeros(3),
            }
    
    return server_policy


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def falsify(
    config_number: int,
    policy: Optional[Callable] = None,
    horizon: int = 300,
    budget: int = 50,
    seed: int = 42,
    output_dir: Optional[str] = None,
    save_video: bool = True,
    config_path: Optional[str] = None,
    use_random_policy: bool = False,
    policy_host: str = "localhost",
    policy_port: int = 5555,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run falsification.
    
    Args:
        config_number: Configuration number (1-based)
        policy: Policy function. If None, creates one.
        horizon: Rollout horizon
        budget: Optimization budget
        seed: Random seed
        output_dir: Output directory
        save_video: Whether to save videos
        config_path: Path to configuration JSON
        use_random_policy: Use random policy
        policy_host: Inference server host
        policy_port: Inference server port
        verbose: Print progress
        
    Returns:
        Result dictionary
    """
    if policy is None:
        policy = create_policy(
            host=policy_host,
            port=policy_port,
            use_random=use_random_policy,
        )
    
    falsifier = Falsifier(
        config_number=config_number,
        policy=policy,
        horizon=horizon,
        budget=budget,
        seed=seed,
        output_dir=output_dir,
        save_video=save_video,
        config_path=config_path,
    )
    
    return falsifier.run(verbose=verbose)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="STL-based Falsification")
    parser.add_argument("--config-number", "-n", type=int, help="Configuration number")
    parser.add_argument("--budget", "-b", type=int, default=50, help="Optimization budget")
    parser.add_argument("--horizon", "-H", type=int, default=300, help="Rollout horizon")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--config-path", "-c", type=str, default=None, help="Config JSON path")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--list", "-l", action="store_true", help="List configurations")
    parser.add_argument("--run", "-r", action="store_true", help="Run falsification")
    parser.add_argument("--use-random-policy", action="store_true", help="Use random policy")
    parser.add_argument("--policy-host", type=str, default="localhost")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--no-video", action="store_true", help="Disable video saving")
    
    args = parser.parse_args()
    
    if args.list:
        from .instantiate import ConfigurationInstantiator
        instantiator = ConfigurationInstantiator(config_path=args.config_path)
        total = instantiator.get_total_configurations()
        print(f"Total configurations: {total}")
        print("\nFirst 10:")
        for i in range(1, min(11, total + 1)):
            config = instantiator.get_abstract_config(i)
            print(f"  {i}: {config.get('task_expression', 'N/A')}")
        return
    
    if args.config_number is None:
        parser.print_help()
        return
    
    if not args.run:
        from .instantiate import ConfigurationInstantiator
        instantiator = ConfigurationInstantiator(config_path=args.config_path)
        config = instantiator.get_abstract_config(args.config_number)
        print(f"\nConfiguration #{args.config_number}:")
        print(f"  Task: {config.get('task_expression', 'N/A')}")
        
        concrete = instantiator.instantiate(args.config_number, seed=args.seed)
        space = PerturbationSpace()
        dim = space.get_dimension(concrete)
        print(f"  Perturbation dim: {dim}")
        print(f"\nUse --run to execute falsification")
        return
    
    result = falsify(
        config_number=args.config_number,
        horizon=args.horizon,
        budget=args.budget,
        seed=args.seed,
        output_dir=args.output_dir,
        save_video=not args.no_video,
        config_path=args.config_path,
        use_random_policy=args.use_random_policy,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
    )
    
    if result['falsified']:
        print(f"\n⚠️ Specification VIOLATED")
    else:
        print(f"\n✓ No violations found within budget")


if __name__ == "__main__":
    main()

