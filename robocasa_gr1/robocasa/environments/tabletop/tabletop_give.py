from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union
import os

import numpy as np
import robocasa.utils.object_utils as OU
import robocasa
from robocasa.environments.tabletop.tabletop import *
from robocasa.environments.tabletop.tabletop_pnp import DEFAULT_DISTRACTOR_CONFIG
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.object_utils import obj_inside_of
from robosuite.controllers import load_composite_controller_config


class TabletopGive(Tabletop, DexMGConfigHelper):
    """
    Deterministic task template that emphasizes fixed placements for a given configuration.

    Compared to ``TabletopAnyTask``, this class enforces deterministic placement by requiring
    explicit placement specifications for every spawnable object. It is intended for scenarios
    where a user wants to "give" (hand over / place / insert) an object to a particular fixture
    or another object using a scripted action plan.

    Args:
        object_plan (List[Dict]): Blueprint describing every interactable entity. Each entry expects:
            - name (str): Unique identifier of the entity referenced in task_plan / success_plan.
            - kind (str, optional): ``"object"`` (default) to spawn MJCF object,
              ``"fixture"`` to alias an existing fixture without spawning.
            - obj_groups / exclude_obj_groups / obj_registries / object_scale: same semantics as other tasks.
            - placement (Dict, required for ``kind == "object"``):
                - fixture (str | Fixture | FixtureType): Anchor surface.
                - pos / xy (tuple[float, float]): XY coordinates relative to fixture.
                - yaw / rotation (float | tuple[float, float]): Deterministic rotation bounds.
                - size (tuple[float, float], optional): Region extent; defaults to a very small region.
                - try_to_place_in, ensure_object_boundary_in_range, etc. if needed.
            - fixture (str, optional for ``kind == "fixture"``): fixture alias to bind to.
            - geom_suffix / obj_type / obj_joint: optional overrides when referencing fixture geoms.

        task_plan (List[Dict]): Ordered sequence of subtasks. Each entry supports:
            - action (str): e.g., ``"pick"``, ``"place"``, ``"manipulate"``, ``"wait"``.
            - object_ref (str): Name from object_plan.
            - target_ref (str, optional): Target object / fixture.
            - gripper_name (str, optional): ``"right"`` (default) or ``"left"``.
            - subtask_term_signal / subtask_term_offset_range / selection_strategy / action_noise /
              num_interpolation_steps / num_fixed_steps / apply_noise_during_interpolation.

        success_plan (Union[Callable, Dict, List[Dict]]): Success evaluation configuration identical
            to ``TabletopAnyTask`` but evaluated deterministically.

        fixture_plan (Dict, optional): Declarative fixture definitions. Keys become attributes on env.
            Special key ``"init_robot_base_pos"`` may specify name of fixture to align robot base with.

        enforce_determinism (bool): If True, validates that placement regions are near-zero sized.
        handedness (str): "right" or "left" used for optional handed offsets.
        layout_id (int, optional): If provided, restricts layouts to this id.
        distractor_config (Dict, optional): Defaults to ``DEFAULT_DISTRACTOR_CONFIG`` but typically
            unused for deterministic scripts.
        use_distractors (bool): Whether to spawn distractors (default False for determinism).
    """

    REGION_EPSILON = 1e-3

    def __init__(
        self,
        object_plan: List[Dict[str, Any]],
        task_plan: List[Dict[str, Any]],
        success_plan: Union[
            Callable[[Tabletop], bool],
            Dict[str, Any],
            List[Dict[str, Any]],
        ],
        fixture_plan: Optional[Dict[str, Any]] = None,
        enforce_determinism: bool = True,
        handedness: str = "right",
        layout_id: Optional[int] = None,
        style_ids: Optional[List[int]] = None,
        distractor_config: Optional[Dict[str, Any]] = None,
        use_distractors: bool = False,
        *args,
        **kwargs,
    ):
        print(f"[DEBUG TabletopGive.__init__] Called with style_ids={style_ids}, layout_id={layout_id}")
        
        if handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")

        self.object_plan = object_plan
        self.task_plan = task_plan
        self.success_plan = success_plan
        self.fixture_plan = fixture_plan or {}
        self.enforce_determinism = enforce_determinism
        self.handedness = handedness
        self._fixture_aliases: Dict[str, Any] = {}

        if layout_id is not None:
            self.VALID_LAYOUTS = [layout_id]
            self._deterministic_layout_id = layout_id
            # Use the first style from style_ids if provided, otherwise default to 0
            if style_ids is not None and len(style_ids) > 0:
                self._deterministic_style_id = style_ids[0]
                print(f"[DEBUG TabletopGive] Set _deterministic_style_id={self._deterministic_style_id} from style_ids={style_ids}")
            else:
                self._deterministic_style_id = kwargs.get('style_id', 0)
                print(f"[DEBUG TabletopGive] Set _deterministic_style_id={self._deterministic_style_id} from kwargs or default")
        else:
            self._deterministic_layout_id = None
            self._deterministic_style_id = None
            print(f"[DEBUG TabletopGive] layout_id is None, so _deterministic_style_id=None")

        # Ensure distractor_config is None if use_distractors is False
        # (to satisfy assertion in parent class)
        if not use_distractors:
            distractor_config = None
        elif distractor_config is None:
            distractor_config = DEFAULT_DISTRACTOR_CONFIG

        # Handle controller config for GR1 robots
        # Check if robots parameter is provided in kwargs
        robots = kwargs.get("robots", None)
        if robots is None and len(args) > 0:
            # robots might be the first positional argument in Tabletop.__init__
            # But in TabletopGive, robots is passed via kwargs, so check kwargs first
            pass
        
        # If GR1FixedLowerBody or similar, set controller to BASIC
        if robots is not None:
            robot_name = robots[0] if isinstance(robots, list) else robots
            if robot_name in ("GR1FixedLowerBody", "GR1ArmsOnly", "GR1ArmsAndWaist", "GR1FixedLowerBodyInspireHands", "GR1FixedLowerBodyFourierHands"):
                # Load default controller config and modify it
                if "controller_configs" not in kwargs or kwargs.get("controller_configs") is None:
                    controller_configs = load_composite_controller_config(
                        controller=None,
                        robot="gr1",  # Use lowercase base name
                    )
                    if controller_configs:
                        controller_configs["type"] = "BASIC"
                        if "composite_controller_specific_configs" not in controller_configs:
                            controller_configs["composite_controller_specific_configs"] = {}
                        controller_configs["control_delta"] = False
                        kwargs["controller_configs"] = controller_configs

        # Extract generative_textures from kwargs before passing to super
        generative_textures = kwargs.pop('generative_textures', False)
        
        # Debug: check what's being passed
        print(f"[DEBUG TabletopGive] Before super().__init__: style_ids={style_ids}, generative_textures={generative_textures}, kwargs keys={list(kwargs.keys())}")
        
        super().__init__(
            distractor_config=distractor_config,
            use_distractors=use_distractors,
            style_ids=style_ids,
            generative_textures=generative_textures,  # Explicitly pass generative_textures
            *args,
            **kwargs,
        )
        
        # Set deterministic layout and style after parent initialization
        if self._deterministic_layout_id is not None:
            if not hasattr(self, '_ep_meta'):
                self._ep_meta = {}
            self._ep_meta['layout_id'] = self._deterministic_layout_id
            self._ep_meta['style_id'] = self._deterministic_style_id

    # --------------------------------------------------------------------- #
    # Fixture helpers
    # --------------------------------------------------------------------- #
    def _setup_table_references(self):
        super()._setup_table_references()

        counter_cfg = self.fixture_plan.get(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.counter = self._register_fixture("counter", counter_cfg)

        for name, cfg in self.fixture_plan.items():
            if name in {"counter", "init_robot_base_pos"}:
                continue
            self._register_fixture(name, cfg)

        init_base = self.fixture_plan.get("init_robot_base_pos")
        if init_base:
            self.init_robot_base_pos = self._resolve_fixture_reference(init_base)
        else:
            self.init_robot_base_pos = self.counter

    def _register_fixture(self, name: str, cfg: Any):
        if isinstance(cfg, dict):
            ref = self.register_fixture_ref(name, cfg)
        else:
            ref = cfg
        setattr(self, name, ref)
        self._fixture_aliases[name] = ref
        return ref

    def _resolve_fixture_reference(self, ref: Any):
        if isinstance(ref, str):
            if hasattr(self, ref):
                return getattr(self, ref)
            if ref in self._fixture_aliases:
                return self._fixture_aliases[ref]
            try:
                fxtr_type = getattr(FixtureType, ref.upper())
                return self.get_fixture(fxtr_type)
            except AttributeError:
                # Not a valid FixtureType enum, try as fixture name string
                try:
                    return self.get_fixture(ref)
                except (AssertionError, KeyError):
                    # Not a valid fixture name either, raise AttributeError
                    raise AttributeError(f"'{ref}' is not a valid fixture reference (not a FixtureType enum, fixture name, or object attribute)")
        return ref

    # --------------------------------------------------------------------- #
    # Object placement
    # --------------------------------------------------------------------- #
    def _get_obj_cfgs(self):
        cfgs = []
        for entry in self.object_plan:
            if entry.get("kind", "object") != "object":
                continue

            placement_cfg = self._build_placement_cfg(entry)
            obj_cfg = {
                "name": entry["name"],
                "placement": placement_cfg,
            }

            # Handle deterministic model specification
            if "model" in entry:
                # If model is specified, use it directly (deterministic)
                model_path = entry["model"]
                # Normalize path
                if not os.path.isabs(model_path):
                    # Try relative to objects directory first
                    if not model_path.endswith(".xml"):
                        test_path = os.path.join(
                            robocasa.models.assets_root, "objects", model_path, "model.xml"
                        )
                        if os.path.exists(test_path):
                            model_path = test_path
                        else:
                            # Try as direct path
                            model_path = os.path.join(
                                robocasa.models.assets_root, "objects", model_path
                            )
                    else:
                        model_path = os.path.join(
                            robocasa.models.assets_root, "objects", model_path
                        )
                elif not model_path.endswith(".xml"):
                    # If absolute but no .xml, assume it's a directory
                    model_path = os.path.join(model_path, "model.xml")
                
                # Check if path exists
                if not os.path.exists(model_path):
                    if self.enforce_determinism:
                        raise ValueError(
                            f"Model path for '{entry['name']}' does not exist: {model_path}\n"
                            "Please specify a valid model path or set enforce_determinism=False "
                            "to use obj_groups instead."
                        )
                    else:
                        # Fall back to obj_groups if available
                        if "obj_groups" in entry:
                            obj_cfg["obj_groups"] = entry["obj_groups"]
                        else:
                            raise ValueError(
                                f"Model path '{model_path}' does not exist and no obj_groups specified."
                            )
                else:
                    # Set obj_groups to the exact model path for deterministic loading
                    obj_cfg["obj_groups"] = model_path
            elif "obj_groups" in entry:
                # If obj_groups is specified but not a model path, require model for determinism
                if self.enforce_determinism:
                    raise ValueError(
                        f"Object '{entry['name']}' must specify 'model' (exact path) "
                        f"for deterministic mode. obj_groups='{entry['obj_groups']}' "
                        "allows random sampling."
                    )
                obj_cfg["obj_groups"] = entry["obj_groups"]
            else:
                if self.enforce_determinism:
                    raise ValueError(
                        f"Object '{entry['name']}' must specify 'model' (exact path) "
                        "for deterministic mode."
                    )
                obj_cfg["obj_groups"] = "all"

            optional_keys = [
                "exclude_obj_groups",
                "graspable",
                "object_scale",
                "obj_registries",
            ]
            for key in optional_keys:
                if key in entry:
                    obj_cfg[key] = entry[key]

            cfgs.append(obj_cfg)
        return cfgs

    def _build_placement_cfg(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if "placement" not in entry:
            raise ValueError(f"Object plan for '{entry['name']}' missing placement.")

        placement = deepcopy(entry["placement"])
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
            placement["size"] = placement.get("size", default_size)
            
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

    # --------------------------------------------------------------------- #
    # Success logic
    # --------------------------------------------------------------------- #
    def _check_success(self):  # type: ignore[override]
        if callable(self.success_plan):
            return bool(self.success_plan(self))

        criteria = (
            self.success_plan
            if isinstance(self.success_plan, list)
            else [self.success_plan]
        )
        for criterion in criteria:
            if not self._evaluate_criterion(criterion):
                return False
        return True

    def _evaluate_criterion(self, criterion: Dict[str, Any]) -> bool:
        ctype = criterion.get("type", "custom")
        params = criterion.get("params", {})
        negate = criterion.get("negate", False)
        result: Union[bool, np.bool_, Any] = False  # type: ignore[name-defined]

        if ctype == "obj_in_receptacle":
            obj_name = params.get("obj_name")
            receptacle_name = params.get("receptacle_name")
            if obj_name and receptacle_name:
                result = OU.check_obj_in_receptacle(
                    self, obj_name, receptacle_name, **params.get("kwargs", {})
                )

        elif ctype == "obj_fixture_contact":
            obj_name = params.get("obj_name")
            fixture_name = params.get("fixture_name")
            if obj_name and fixture_name:
                fixture = self._resolve_fixture_reference(fixture_name)
                result = OU.check_obj_fixture_contact(self, obj_name, fixture)

        elif ctype == "obj_upright":
            obj_name = params.get("obj_name")
            threshold = params.get("threshold", 0.8)
            if obj_name:
                result = OU.check_obj_upright(self, obj_name, threshold=threshold)

        elif ctype == "gripper_far":
            obj_name = params.get("obj_name")
            if obj_name:
                result = OU.any_gripper_obj_far(self, obj_name=obj_name)

        elif ctype == "fixture_state":
            fixture_name = params.get("fixture_name")
            state_key = params.get("state_key")
            comparison = params.get("comparison", "eq")
            expected_value = params.get("expected_value")

            if fixture_name and state_key is not None:
                fixture = self._resolve_fixture_reference(fixture_name)
                if hasattr(fixture, "get_door_state"):
                    state = fixture.get_door_state(env=self)
                    actual_value = state.get(state_key)
                    if actual_value is not None:
                        if comparison == "eq":
                            result = abs(actual_value - expected_value) < 1e-3
                        elif comparison == "ge":
                            result = actual_value >= expected_value
                        elif comparison == "le":
                            result = actual_value <= expected_value

        elif ctype == "obj_inside_fixture":
            obj_name = params.get("obj_name")
            fixture_name = params.get("fixture_name")
            partial_check = params.get("partial_check", False)
            if obj_name and fixture_name:
                fixture = self._resolve_fixture_reference(fixture_name)
                result = obj_inside_of(
                    env=self,
                    obj_name=self.objects[obj_name].name,
                    fixture_id=fixture,
                    partial_check=partial_check,
                )

        elif ctype == "door_open":
            fixture_name = params.get("fixture_name")
            if fixture_name:
                fixture = self._resolve_fixture_reference(fixture_name)
                if hasattr(fixture, "get_door_state"):
                    state = fixture.get_door_state(env=self)
                    door_value = state.get("door", 0.0)
                    # Door is open if door value is greater than a threshold (typically > 0.1)
                    result = door_value > 0.1

        elif ctype == "door_closed":
            fixture_name = params.get("fixture_name")
            if fixture_name:
                fixture = self._resolve_fixture_reference(fixture_name)
                if hasattr(fixture, "get_door_state"):
                    state = fixture.get_door_state(env=self)
                    door_value = state.get("door", 0.0)
                    # Door is closed if door value is less than a threshold (typically < 0.1)
                    result = door_value < 0.1

        elif ctype == "custom":
            func = params.get("func")
            if callable(func):
                result = func(self, **params.get("kwargs", {}))

        result = bool(result) if result is not False else False
        return not result if negate else result

    # --------------------------------------------------------------------- #
    # DexMG helpers
    # --------------------------------------------------------------------- #
    def get_object(self):
        objects = {}
        for entry in self.object_plan:
            name = entry["name"]
            if entry.get("kind", "object") == "fixture":
                fixture_ref = entry.get("fixture", name)
                fixture = self._resolve_fixture_reference(fixture_ref)
                geom_suffix = entry.get("geom_suffix", "")
                obj_type = entry.get("obj_type", "geom")
                objects[name] = dict(
                    obj_name=fixture.name + geom_suffix,
                    obj_type=obj_type,
                    obj_joint=entry.get("obj_joint"),
                )
            elif name in self.objects:
                objects[name] = dict(
                    obj_name=self.objects[name].root_body,
                    obj_type="body",
                    obj_joint=None,
                )
        return objects

    def get_subtask_term_signals(self):
        signals = {}
        for step in self.task_plan:
            signal_name = step.get("subtask_term_signal")
            if not signal_name:
                continue

            action = step.get("action", "pick")
            obj_ref = step.get("object_ref")
            if action == "pick" and obj_ref in self.objects:
                gripper_name = step.get("gripper_name", "right")
                gripper = self.robots[0].gripper[gripper_name]
                signals[signal_name] = int(
                    self._check_grasp(gripper=gripper, object_geoms=self.objects[obj_ref])
                )
            elif action == "place" and obj_ref:
                target_ref = step.get("target_ref")
                if target_ref and target_ref in self.objects:
                    signals[signal_name] = int(
                        OU.check_obj_in_receptacle(self, obj_ref, target_ref)
                    )
                elif target_ref:
                    fixture = self._resolve_fixture_reference(target_ref)
                    signals[signal_name] = int(
                        OU.check_obj_fixture_contact(self, obj_ref, fixture)
                    )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()

    def get_task_config(self):
        task = DexMGConfigHelper.AttrDict()
        for idx, step in enumerate(self.task_plan, start=1):
            gripper_name = step.get("gripper_name", "right")
            task_spec_key = f"task_spec_{0 if gripper_name == 'right' else 1}"
            subtask_key = f"subtask_{idx}"

            subtask_cfg = {
                "object_ref": step.get("object_ref"),
                "subtask_term_signal": step.get("subtask_term_signal"),
                "subtask_term_offset_range": step.get("subtask_term_offset_range"),
                "selection_strategy": step.get("selection_strategy", "random"),
                "selection_strategy_kwargs": step.get("selection_strategy_kwargs"),
                "action_noise": step.get("action_noise", 0.0),
                "num_interpolation_steps": step.get("num_interpolation_steps", 5),
                "num_fixed_steps": step.get("num_fixed_steps", 0),
                "apply_noise_during_interpolation": step.get(
                    "apply_noise_during_interpolation", False
                ),
            }
            if step.get("action") == "pick":
                subtask_cfg["selection_object_ref"] = step.get("object_ref")

            getattr(task, task_spec_key)[subtask_key] = subtask_cfg
        return task.to_dict()

    # --------------------------------------------------------------------- #
    # Episode metadata
    # --------------------------------------------------------------------- #
    def _get_obj_lang_name(self, obj_ref: str) -> str:
        """
        Get the language name for an object reference.
        
        Priority:
        1. display_name or lang_name from object_plan
        2. get_obj_lang() if object exists in self.objects
        3. Fall back to obj_ref
        
        Args:
            obj_ref: Object reference name from task_plan
            
        Returns:
            Language description string for the object
        """
        # First, check if display_name or lang_name is specified in object_plan
        for entry in self.object_plan:
            if entry.get("name") == obj_ref:
                display_name = entry.get("display_name") or entry.get("lang_name")
                if display_name:
                    return display_name
        
        # Try to use get_obj_lang if object exists
        if obj_ref in self.objects:
            try:
                lang_result = self.get_obj_lang(obj_name=obj_ref)
                # get_obj_lang returns a tuple if get_preposition=True, otherwise a string
                if isinstance(lang_result, tuple):
                    return lang_result[0]
                return lang_result
            except (AssertionError, KeyError, AttributeError):
                pass
        
        # Fall back to obj_ref
        return obj_ref
    
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        lang_bits = []
        for step in self.task_plan:
            action = step.get("action", "manipulate")
            obj_ref = step.get("object_ref")
            target_ref = step.get("target_ref")

            if action == "pick" and obj_ref:
                obj_lang = self._get_obj_lang_name(obj_ref)
                lang_bits.append(f"pick the {obj_lang}")
            elif action == "place" and obj_ref and target_ref:
                obj_lang = self._get_obj_lang_name(obj_ref)
                target_lang = self._get_obj_lang_name(target_ref)
                # Determine preposition based on target type
                preposition = "on"  # default
                if target_ref in self.objects:
                    try:
                        lang_result = self.get_obj_lang(obj_name=target_ref, get_preposition=True)
                        if isinstance(lang_result, tuple) and len(lang_result) == 2:
                            _, preposition = lang_result
                    except (AssertionError, KeyError, AttributeError, ValueError):
                        pass
                lang_bits.append(f"place {obj_lang} {preposition} {target_lang}")
            elif action == "manipulate" and obj_ref:
                obj_lang = self._get_obj_lang_name(obj_ref)
                lang_bits.append(f"manipulate the {obj_lang}")

        if lang_bits:
            ep_meta["lang"] = " and ".join(lang_bits)
        return ep_meta

    # --------------------------------------------------------------------- #
    # Deterministic object creation override
    # --------------------------------------------------------------------- #
    def _create_obj(self, cfg):
        """
        Override to ensure deterministic object creation when model path is specified.
        """
        # If model path is specified (ends with .xml), use it directly
        obj_groups = cfg.get("obj_groups", "all")
        
        if isinstance(obj_groups, str) and obj_groups.endswith(".xml"):
            # Direct model path - use it deterministically
            # Pass it as obj_groups to sample_object, which handles .xml paths deterministically
            # But first, ensure we have the info structure to avoid re-sampling
            mjcf_path = obj_groups
            # Normalize path
            if not os.path.isabs(mjcf_path):
                # Try to resolve relative to assets root
                from robosuite.utils.mjcf_utils import xml_path_completion
                try:
                    mjcf_path = xml_path_completion(mjcf_path, root=robocasa.models.assets_root)
                except:
                    # If that fails, try relative to objects directory
                    mjcf_path = os.path.join(
                        robocasa.models.assets_root, "objects", 
                        mjcf_path.split("/objects/")[-1] if "/objects/" in mjcf_path else mjcf_path
                    )
            
            # Use sample_object with the exact path - it handles .xml paths deterministically
            object_kwargs, object_info = self.sample_object(
                mjcf_path,  # Pass exact path
                exclude_groups=cfg.get("exclude_obj_groups", None),
                exclude_cat=cfg.get("exclude_cat", None),
                graspable=cfg.get("graspable", None),
                washable=cfg.get("washable", None),
                microwavable=cfg.get("microwavable", None),
                cookable=cfg.get("cookable", None),
                freezable=cfg.get("freezable", None),
                max_size=cfg.get("max_size", (None, None, None)),
                object_scale=cfg.get("object_scale", None),
            )
            
            object = MJCFObject(name=cfg["name"], **object_kwargs)
            return object, object_info
        else:
            # Fall back to parent implementation
            return super()._create_obj(cfg)

