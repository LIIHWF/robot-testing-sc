from robocasa.environments.tabletop.tabletop import *
from robocasa.environments.tabletop.tabletop_pnp import DEFAULT_DISTRACTOR_CONFIG
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.object_utils import obj_inside_of
import robocasa.utils.object_utils as OU
from typing import List, Dict, Callable, Optional, Any, Union


class TabletopPnP(Tabletop, DexMGConfigHelper):
    """
    Class encapsulating the atomic counter to container pick and place task.

    Args:
        container_type (str): Type of container to place the object in.

        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.

        distractor_config (dict): Configuration for distractor objects.

        use_distractors (bool): Whether to use distractor objects.

        handedness (Optional[str]): Which hand to optimize object spawning for ("right" or "left").

        source_container (str): Source container to sample the target object from.

        target_container (str): Target container to place the object in.
    """

    VALID_LAYOUTS = [0]

    def __init__(
        self,
        obj_groups="all",
        exclude_obj_groups=None,
        source_container=None,
        source_container_size=(0.5, 0.5),
        target_container=None,
        target_container_size=(0.5, 0.5),
        distractor_config=DEFAULT_DISTRACTOR_CONFIG,
        use_distractors=True,
        handedness="right",
        *args,
        **kwargs,
    ):
        if handedness is not None and handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        self.target_container = target_container
        self.source_container = source_container
        self.source_container_size = source_container_size
        self.target_container_size = target_container_size
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        self.handedness = handedness

        super().__init__(
            distractor_config=distractor_config,
            use_distractors=use_distractors,
            *args,
            **kwargs,
        )

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        source_container_lang = (
            self.get_obj_lang(obj_name="obj_container")
            if self.source_container
            else "counter"
        )
        target_container_lang = (
            self.get_obj_lang(obj_name="container")
            if self.target_container
            else "counter"
        )
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the {source_container_lang} and place it in the {target_container_lang}"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        if self.target_container:
            cfgs.append(
                dict(
                    name="container",
                    obj_groups=self.target_container,
                    placement=dict(
                        fixture=self.counter,
                        size=self.target_container_size,
                        pos=(0.9, -0.3) if self.handedness == "right" else (-0.9, -0.3),
                    ),
                )
            )

        # Randomize handedness per invoke rather than storing it
        handedness = (
            self.handedness if self.handedness else self.rng.choice(["left", "right"])
        )
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(
                        self.source_container_size
                        if self.source_container
                        else (0.3, 0.3)
                    ),
                    pos=(0.5, -0.8) if handedness == "right" else (-0.5, -0.8),
                    try_to_place_in=self.source_container,
                ),
            )
        )
        return cfgs

    def _check_success(self):
        if self.target_container:
            gripper_container_far = OU.any_gripper_obj_far(self, obj_name="container")
            gripper_obj_far = OU.any_gripper_obj_far(self, obj_name="obj")
            obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container")
            obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
            container_upright = OU.check_obj_upright(self, "container", threshold=0.8)
            return (
                gripper_container_far
                and gripper_obj_far
                and obj_in_container
                and not obj_on_counter
                and container_upright
            )
        else:
            gripper_obj_far = OU.any_gripper_obj_far(self, obj_name="obj")
            obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
            return gripper_obj_far and obj_on_counter

    def get_object(self):
        objects = dict()
        objects["obj"] = dict(
            obj_name=self.objects["obj"].root_body, obj_type="body", obj_joint=None
        )

        objects["container"] = dict(
            obj_name=self.objects["container"].root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_object"] = int(
            self._check_grasp(
                gripper=self.robots[0].gripper["right"],
                object_geoms=self.objects["obj"],
            )
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            selection_object_ref="container",
            object_ref="obj",
            subtask_term_signal="grasp_object",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="container",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        return task.to_dict()

    def visualize_spawn_region(self, obj_name="container"):
        """
        Draws a single site representing the full spawn region as a 3D box, cylinder, or sphere.
        """
        import robocasa.utils.object_utils as OU
        import numpy as np
        import mujoco

        highest_spawn_region = OU.get_highest_spawn_region(self, self.objects[obj_name])

        for spawn in self.objects[obj_name].spawns:
            if spawn != highest_spawn_region:
                continue

            # Get spawn region parameters
            region_points = OU.calculate_spawn_region(self, spawn)
            spawn_type = spawn.get("type", "box")

            # Create or update geom
            self.viewer.update()
            current_geom = None
            for geom_idx in range(self.viewer.viewer.user_scn.ngeom):
                if self.viewer.viewer.user_scn.geoms[geom_idx].label == spawn.get(
                    "name"
                ):
                    current_geom = self.viewer.viewer.user_scn.geoms[geom_idx]
                    break
            if current_geom is None:
                self.viewer.viewer.user_scn.ngeom += 1
                current_geom = self.viewer.viewer.user_scn.geoms[
                    self.viewer.viewer.user_scn.ngeom - 1
                ]

            if spawn_type == "box":
                p0, px, py, pz = region_points
                v_x = px - p0
                v_y = py - p0
                v_z = pz - p0
                center = p0 + 0.5 * (v_x + v_y + v_z)
                half_size = np.array(
                    [
                        np.linalg.norm(v_x) / 2.0,
                        np.linalg.norm(v_y) / 2.0,
                        np.linalg.norm(v_z) / 2.0,
                    ]
                )
                R = np.column_stack(
                    (
                        v_x / np.linalg.norm(v_x),
                        v_y / np.linalg.norm(v_y),
                        v_z / np.linalg.norm(v_z),
                    )
                )
                geom_type = mujoco.mjtGeom.mjGEOM_BOX

            elif spawn_type == "cylinder":
                p0, axis_vector, radius = region_points
                height = np.linalg.norm(axis_vector)
                center = p0 + 0.5 * axis_vector
                half_size = np.array([radius, height / 2, radius])
                z_axis = axis_vector / height
                x_axis = (
                    np.array([1, 0, 0]) if abs(z_axis[1]) < 0.9 else np.array([0, 1, 0])
                )
                y_axis = np.cross(z_axis, x_axis)
                x_axis = np.cross(y_axis, z_axis)
                R = np.column_stack((x_axis, y_axis, z_axis))
                geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER

            elif spawn_type == "sphere":
                center, radius = region_points
                half_size = np.array([radius, radius, radius])
                R = np.eye(3)
                geom_type = mujoco.mjtGeom.mjGEOM_SPHERE

            else:
                raise ValueError(f"Invalid spawn type: {spawn_type}")

            mujoco.mjv_initGeom(
                current_geom,
                type=geom_type,
                size=half_size,
                pos=center,
                mat=R.reshape(9, 1),
                rgba=np.array([0, 1, 0, 0.8]),
            )
            current_geom.label = spawn.get("name")

    def visualize_bounding_box(self, obj_name="obj"):
        """
        Draws a single site representing the full container spawn region as a 3D box.
        """
        import numpy as np
        import robocasa.utils.transform_utils as T

        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])
        obj_quat = T.convert_quat(
            np.array(self.sim.data.body_xquat[self.obj_body_id[obj_name]]), to="xyzw"
        )

        bbox_points = self.objects[obj_name].get_bbox_points(
            trans=obj_pos, rot=obj_quat
        )

        p0, px, py, pz = bbox_points[:4]

        v_x = px - p0
        v_y = py - p0
        v_z = pz - p0
        center = p0 + 0.5 * (v_x + v_y + v_z)
        half_extent_x = np.linalg.norm(v_x) / 2.0
        half_extent_y = np.linalg.norm(v_y) / 2.0
        half_extent_z = np.linalg.norm(v_z) / 2.0
        half_size = np.array([half_extent_x, half_extent_y, half_extent_z])

        x_axis = v_x / np.linalg.norm(v_x)
        y_axis = v_y / np.linalg.norm(v_y)
        z_axis = v_z / np.linalg.norm(v_z)

        R = np.column_stack((x_axis, y_axis, z_axis))

        import mujoco

        self.viewer.update()
        current_geom = None
        for geom_idx in range(self.viewer.viewer.user_scn.ngeom):
            if self.viewer.viewer.user_scn.geoms[geom_idx].label == obj_name:
                current_geom = self.viewer.viewer.user_scn.geoms[geom_idx]
                break
        if current_geom is None:
            self.viewer.viewer.user_scn.ngeom += 1
            current_geom = self.viewer.viewer.user_scn.geoms[
                self.viewer.viewer.user_scn.ngeom - 1
            ]
        mujoco.mjv_initGeom(
            current_geom,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_size,
            pos=center,
            mat=R.reshape(9, 1),
            rgba=np.array([0, 0, 1, 0.8]),
        )
        current_geom.label = obj_name



class TabletopMicrowavePnPClose(Tabletop, DexMGConfigHelper):
    """
    Class encapsulating the microwave-based pick and place task.

    Args:
        obj_groups (str): Object groups to sample the target object from.
        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
        handedness (Optional[str]): Which hand to optimize object spawning for ("right" or "left").
        obj_scale (float): Scaling factor for the object.
        behavior (str): "open" or "close" for microwave door manipulation behavior.
    """

    VALID_LAYOUTS = [2]
    NUM_OBJECTS = 1

    def __init__(
        self,
        obj_groups="all",
        exclude_obj_groups=None,
        handedness="right",
        obj_scale=1.0,
        behavior="close",
        distractor_config=DEFAULT_DISTRACTOR_CONFIG,
        use_distractors=True,
        *args,
        **kwargs,
    ):
        if handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        assert behavior in ["open", "close"], "Invalid behavior"

        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        self.handedness = handedness
        self.obj_scale = obj_scale
        self.behavior = behavior

        super().__init__(
            *args,
            **kwargs,
            distractor_config=distractor_config,
            use_distractors=use_distractors,
        )

    def _setup_table_references(self):
        """
        Setup references for the microwave and workspace.
        """
        super()._setup_table_references()
        self.microwave = self.get_fixture(FixtureType.MICROWAVE)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.microwave

    def get_ep_meta(self):
        """
        Get the episode metadata for the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="obj")
        ep_meta[
            "lang"
        ] = f"pick up the {obj_lang}, place it into the microwave and close the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Define object configurations for pick and place.
        """
        cfgs = []
        handedness = (
            self.handedness if self.handedness else self.rng.choice(["left", "right"])
        )
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                object_scale=self.obj_scale,
                placement=dict(
                    fixture=self.counter,
                    size=(0.2, 0.2),
                    pos=((0.5, -0.8) if handedness == "right" else (-0.5, -0.2)),
                ),
                obj_registries=["objaverse"],
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the object is successfully placed inside the microwave.
        """
        inside_of_microwave = obj_inside_of(
            env=self,
            obj_name=self.objects["obj"].name,
            fixture_id=self.microwave,
            partial_check=True,
        )
        door_state = self.microwave.get_door_state(env=self)["door"]
        door_state_correct = (
            door_state >= 0.5 if self.behavior == "open" else door_state <= 0.005
        )
        return inside_of_microwave and door_state_correct

    def get_object(self):
        """
        Return object references for the task.
        """
        objects = {
            "obj": dict(
                obj_name=self.objects["obj"].root_body,
                obj_type="body",
                obj_joint=None,
            ),
            "microwave": dict(
                obj_name=self.microwave.name + "_door_handle",
                obj_type="geom",
                obj_joint=None,
            ),
        }
        return objects

    def get_subtask_term_signals(self):
        """
        Define subtask termination signals.
        """
        signals = {
            "grasp_object": int(
                self._check_grasp(
                    gripper=self.robots[0].gripper["right"],
                    object_geoms=self.objects["obj"],
                )
            ),
            "obj_in_microwave": int(
                OU.check_obj_fixture_contact(self, "obj", self.microwave)
            ),
        }
        return signals

    def _reset_internal(self):
        """
        Reset the environment for the microwave PnP task.
        """
        super()._reset_internal()
        if self.behavior == "open":
            self.microwave.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    @staticmethod
    def task_config():
        return {
            "task_spec_0": {
                "subtask_1": dict(
                    object_ref="obj",
                    subtask_term_signal="grasp_object",
                    subtask_term_offset_range=(5, 10),
                    selection_strategy="random",
                    selection_strategy_kwargs=None,
                    action_noise=0.0,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=True,
                ),
                "subtask_2": dict(
                    object_ref="microwave",
                    subtask_term_signal=None,
                    subtask_term_offset_range=None,
                    selection_strategy="random",
                    selection_strategy_kwargs=None,
                    action_noise=0.0,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=True,
                ),
            },
            "task_spec_1": {
                "subtask_1": dict(
                    object_ref="microwave",
                    subtask_term_signal=None,
                    subtask_term_offset_range=None,
                    selection_strategy="random",
                    selection_strategy_kwargs=None,
                    action_noise=0.0,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=True,
                ),
            },
        }


class TabletopAnyTask(Tabletop, DexMGConfigHelper):
    """
    A general-purpose task framework that allows flexible task definition through configuration.
    
    This class enables users to define tasks by specifying:
    1. Item configurations with coordinates and properties
    2. Task sequence templates
    3. Custom success criteria
    
    Args:
        item_configs (List[Dict]): List of item configurations. Each dict should contain:
            - name (str): Unique name for the item
            - obj_groups (str or List[str]): Object groups to sample from
            - exclude_obj_groups (str or List[str], optional): Object groups to exclude
            - placement (Dict): Placement configuration with:
                - fixture (str or Fixture): Fixture reference or fixture type
                - pos (tuple): Position coordinates (x, y) relative to fixture
                - size (tuple): Size of placement region (width, height)
                - rotation (tuple, optional): Rotation range in radians
                - try_to_place_in (str, optional): Container name to place inside
            - graspable (bool, optional): Whether the object can be grasped
            - object_scale (float, optional): Scaling factor for the object
            - obj_registries (List[str], optional): Object registries to use
        
        task_sequence (List[Dict]): Sequence of subtasks. Each dict should contain:
            - action (str): Action type ("pick", "place", "manipulate", "wait")
            - object_ref (str): Reference to item name from item_configs
            - target_ref (str, optional): Target object/fixture for place/manipulate actions
            - subtask_term_signal (str, optional): Signal name for termination
            - subtask_term_offset_range (tuple, optional): Offset range for termination
            - selection_strategy (str, optional): Strategy for object selection
            - action_noise (float, optional): Noise level for actions
            - num_interpolation_steps (int, optional): Number of interpolation steps
            - num_fixed_steps (int, optional): Number of fixed steps
            - apply_noise_during_interpolation (bool, optional): Whether to apply noise
        
        success_criteria (Union[Callable, Dict, List[Dict]]): Success checking criteria:
            - If Callable: Direct function that takes (env) and returns bool
            - If Dict: Single criterion with type and parameters
            - If List[Dict]: Multiple criteria (all must pass)
            Criteria dict format:
                - type (str): Type of check ("obj_in_receptacle", "obj_fixture_contact", 
                  "obj_upright", "gripper_far", "fixture_state", "custom")
                - params (Dict): Parameters for the check
                - negate (bool, optional): Whether to negate the result
        
        fixture_configs (Dict, optional): Additional fixture configurations
        handedness (str, optional): "right" or "left" for hand preference
        layout_id (int, optional): Layout ID to use (default: 0)
        distractor_config (Dict, optional): Distractor configuration
        use_distractors (bool, optional): Whether to use distractors
    """
    
    VALID_LAYOUTS = [0]  # Can be overridden
    
    def __init__(
        self,
        item_configs: List[Dict],
        task_sequence: List[Dict],
        success_criteria: Union[Callable, Dict, List[Dict]],
        fixture_configs: Optional[Dict] = None,
        handedness: str = "right",
        layout_id: int = 0,
        distractor_config: Dict = DEFAULT_DISTRACTOR_CONFIG,
        use_distractors: bool = True,
        *args,
        **kwargs,
    ):
        if handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        
        self.item_configs = item_configs
        self.task_sequence = task_sequence
        self.success_criteria = success_criteria
        self.fixture_configs = fixture_configs or {}
        self.handedness = handedness
        self.layout_id = layout_id
        
        # Set VALID_LAYOUTS if layout_id is specified
        if layout_id is not None:
            self.VALID_LAYOUTS = [layout_id]
        
        super().__init__(
            distractor_config=distractor_config,
            use_distractors=use_distractors,
            *args,
            **kwargs,
        )
    
    def _setup_table_references(self):
        """Setup table and fixture references based on configuration."""
        super()._setup_table_references()
        
        # Setup default counter if not specified
        if "counter" not in self.fixture_configs:
            self.counter = self.register_fixture_ref(
                "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
            )
        else:
            counter_cfg = self.fixture_configs["counter"]
            if isinstance(counter_cfg, dict):
                self.counter = self.register_fixture_ref("counter", counter_cfg)
            else:
                self.counter = counter_cfg
        
        # Setup other fixtures from fixture_configs
        for name, cfg in self.fixture_configs.items():
            if name == "counter":
                continue
            if isinstance(cfg, dict):
                setattr(self, name, self.register_fixture_ref(name, cfg))
            else:
                setattr(self, name, cfg)
        
        # Set initial robot base position
        if "init_robot_base_pos" in self.fixture_configs:
            self.init_robot_base_pos = getattr(
                self, self.fixture_configs["init_robot_base_pos"]
            )
        else:
            self.init_robot_base_pos = self.counter
    
    def _resolve_fixture_reference(self, fixture_ref):
        """Resolve fixture reference to actual fixture object."""
        if isinstance(fixture_ref, str):
            # Try to get from attributes first
            if hasattr(self, fixture_ref):
                return getattr(self, fixture_ref)
            # Try to get by fixture type
            try:
                fixture_type = getattr(FixtureType, fixture_ref.upper())
                return self.get_fixture(fixture_type)
            except (AttributeError, KeyError):
                # Default to counter
                return self.counter
        return fixture_ref
    
    def _get_obj_cfgs(self):
        """Generate object configurations from item_configs."""
        cfgs = []
        
        for item_cfg in self.item_configs:
            cfg = {
                "name": item_cfg["name"],
                "obj_groups": item_cfg.get("obj_groups", "all"),
            }
            
            if "exclude_obj_groups" in item_cfg:
                cfg["exclude_obj_groups"] = item_cfg["exclude_obj_groups"]
            
            if "graspable" in item_cfg:
                cfg["graspable"] = item_cfg["graspable"]
            
            if "object_scale" in item_cfg:
                cfg["object_scale"] = item_cfg["object_scale"]
            
            if "obj_registries" in item_cfg:
                cfg["obj_registries"] = item_cfg["obj_registries"]
            
            # Process placement configuration
            if "placement" in item_cfg:
                placement = item_cfg["placement"].copy()
                
                # Resolve fixture reference
                if "fixture" in placement:
                    placement["fixture"] = self._resolve_fixture_reference(
                        placement["fixture"]
                    )
                
                # Adjust position based on handedness if needed
                if "pos" in placement and isinstance(placement["pos"], tuple):
                    pos = placement["pos"]
                    # If position is a function of handedness, apply it
                    if callable(pos):
                        placement["pos"] = pos(self.handedness)
                
                cfg["placement"] = placement
            
            cfgs.append(cfg)
        
        return cfgs
    
    def _check_success(self):
        """Check success based on success_criteria."""
        if callable(self.success_criteria):
            result = self.success_criteria(self)
            return bool(result)
        
        criteria_list = (
            self.success_criteria
            if isinstance(self.success_criteria, list)
            else [self.success_criteria]
        )
        
        for criterion in criteria_list:
            if not self._evaluate_criterion(criterion):
                return False
        
        return True
    
    def _evaluate_criterion(self, criterion: Dict) -> bool:
        """Evaluate a single success criterion."""
        criterion_type = criterion.get("type", "custom")
        params = criterion.get("params", {})
        negate = criterion.get("negate", False)
        
        result = False
        
        if criterion_type == "obj_in_receptacle":
            obj_name = params.get("obj_name")
            receptacle_name = params.get("receptacle_name")
            if obj_name and receptacle_name:
                result = OU.check_obj_in_receptacle(
                    self, obj_name, receptacle_name, **params.get("kwargs", {})
                )
        
        elif criterion_type == "obj_fixture_contact":
            obj_name = params.get("obj_name")
            fixture_name = params.get("fixture_name")
            if obj_name and fixture_name:
                fixture = self._resolve_fixture_reference(fixture_name)
                result = OU.check_obj_fixture_contact(self, obj_name, fixture)
        
        elif criterion_type == "obj_upright":
            obj_name = params.get("obj_name")
            threshold = params.get("threshold", 0.8)
            if obj_name:
                result = OU.check_obj_upright(self, obj_name, threshold=threshold)
        
        elif criterion_type == "gripper_far":
            obj_name = params.get("obj_name")
            gripper_name = params.get("gripper_name", "right")
            if obj_name:
                gripper = self.robots[0].gripper[gripper_name]
                result = OU.any_gripper_obj_far(self, obj_name=obj_name)
        
        elif criterion_type == "fixture_state":
            fixture_name = params.get("fixture_name")
            state_key = params.get("state_key")
            expected_value = params.get("expected_value")
            comparison = params.get("comparison", "eq")  # eq, ge, le
            
            if fixture_name and state_key is not None:
                fixture = self._resolve_fixture_reference(fixture_name)
                if hasattr(fixture, "get_door_state"):
                    state = fixture.get_door_state(env=self)
                    actual_value = state.get(state_key)
                    
                    if comparison == "eq":
                        result = abs(actual_value - expected_value) < 0.01
                    elif comparison == "ge":
                        result = actual_value >= expected_value
                    elif comparison == "le":
                        result = actual_value <= expected_value
        
        elif criterion_type == "obj_inside_fixture":
            obj_name = params.get("obj_name")
            fixture_name = params.get("fixture_name")
            partial_check = params.get("partial_check", False)
            
            if obj_name and fixture_name:
                fixture = self._resolve_fixture_reference(fixture_name)
                if hasattr(fixture, "id"):
                    result = obj_inside_of(
                        env=self,
                        obj_name=self.objects[obj_name].name,
                        fixture_id=fixture.id,
                        partial_check=partial_check,
                    )
        
        elif criterion_type == "custom":
            func = params.get("func")
            if callable(func):
                result = func(self, **params.get("kwargs", {}))
        
        # Ensure result is a Python bool, not numpy bool_
        result = bool(result) if result is not False else False
        return not result if negate else result
    
    def get_object(self):
        """Return object references for all items."""
        objects = {}
        
        for item_cfg in self.item_configs:
            item_name = item_cfg["name"]
            if item_name in self.objects:
                obj = self.objects[item_name]
                objects[item_name] = dict(
                    obj_name=obj.root_body,
                    obj_type="body",
                    obj_joint=None,
                )
        
        # Add fixture references if needed
        for task_step in self.task_sequence:
            target_ref = task_step.get("target_ref")
            if target_ref and target_ref not in objects:
                fixture = self._resolve_fixture_reference(target_ref)
                if hasattr(fixture, "name"):
                    # Try to get a handle or specific geom
                    handle_name = task_step.get("handle_name", "_door_handle")
                    geom_name = fixture.name + handle_name
                    objects[target_ref] = dict(
                        obj_name=geom_name,
                        obj_type="geom",
                        obj_joint=None,
                    )
        
        return objects
    
    def get_subtask_term_signals(self):
        """Generate subtask termination signals from task sequence."""
        signals = {}
        
        for task_step in self.task_sequence:
            signal_name = task_step.get("subtask_term_signal")
            if signal_name:
                action = task_step.get("action", "pick")
                object_ref = task_step.get("object_ref")
                
                if action == "pick" and object_ref and object_ref in self.objects:
                    gripper_name = task_step.get("gripper_name", "right")
                    gripper = self.robots[0].gripper[gripper_name]
                    signals[signal_name] = int(
                        self._check_grasp(
                            gripper=gripper,
                            object_geoms=self.objects[object_ref],
                        )
                    )
                elif action == "place" and object_ref:
                    # Check if object is in target
                    target_ref = task_step.get("target_ref")
                    if target_ref:
                        if target_ref in self.objects:
                            signals[signal_name] = int(
                                OU.check_obj_in_receptacle(
                                    self, object_ref, target_ref
                                )
                            )
                        else:
                            # Target is a fixture
                            fixture = self._resolve_fixture_reference(target_ref)
                            signals[signal_name] = int(
                                OU.check_obj_fixture_contact(self, object_ref, fixture)
                            )
        
        return signals
    
    @staticmethod
    def task_config():
        """This should be overridden or called with task_sequence."""
        # Default implementation - should be customized per instance
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        return task.to_dict()
    
    def get_task_config(self):
        """Generate task configuration from task_sequence."""
        task = DexMGConfigHelper.AttrDict()
        
        # Group subtasks by task_spec (hand)
        task_specs = {}
        current_task_spec = 0
        current_subtask = 1
        
        for step_idx, task_step in enumerate(self.task_sequence):
            # Determine which hand/task_spec to use
            gripper_name = task_step.get("gripper_name", "right")
            task_spec_key = f"task_spec_{0 if gripper_name == 'right' else 1}"
            
            if task_spec_key not in task_specs:
                task_specs[task_spec_key] = {}
            
            subtask_key = f"subtask_{current_subtask}"
            
            # Build subtask config
            subtask_config = {
                "object_ref": task_step.get("object_ref"),
                "subtask_term_signal": task_step.get("subtask_term_signal"),
                "subtask_term_offset_range": task_step.get("subtask_term_offset_range"),
                "selection_strategy": task_step.get("selection_strategy", "random"),
                "selection_strategy_kwargs": task_step.get("selection_strategy_kwargs"),
                "action_noise": task_step.get("action_noise", 0.05),
                "num_interpolation_steps": task_step.get("num_interpolation_steps", 5),
                "num_fixed_steps": task_step.get("num_fixed_steps", 0),
                "apply_noise_during_interpolation": task_step.get(
                    "apply_noise_during_interpolation", True
                ),
            }
            
            # Add selection_object_ref for pick actions
            if task_step.get("action") == "pick":
                subtask_config["selection_object_ref"] = task_step.get("object_ref")
            
            task_specs[task_spec_key][subtask_key] = subtask_config
            
            # Increment subtask counter
            current_subtask += 1
        
        # Convert to AttrDict structure
        for task_spec_key, subtasks in task_specs.items():
            task_spec_num = int(task_spec_key.split("_")[-1])
            for subtask_key, subtask_config in subtasks.items():
                subtask_num = int(subtask_key.split("_")[-1])
                setattr(
                    getattr(task, f"task_spec_{task_spec_num}"),
                    f"subtask_{subtask_num}",
                    subtask_config,
                )
        
        return task.to_dict()
    
    def get_ep_meta(self):
        """Generate episode metadata."""
        ep_meta = super().get_ep_meta()
        
        # Generate language description from task sequence
        lang_parts = []
        for step in self.task_sequence:
            action = step.get("action", "manipulate")
            obj_ref = step.get("object_ref", "object")
            target_ref = step.get("target_ref")
            
            if action == "pick":
                obj_lang = self.get_obj_lang(obj_name=obj_ref) if obj_ref in self.objects else obj_ref
                lang_parts.append(f"pick the {obj_lang}")
            elif action == "place" and target_ref:
                target_lang = (
                    self.get_obj_lang(obj_name=target_ref)
                    if target_ref in self.objects
                    else target_ref
                )
                lang_parts.append(f"place it in the {target_lang}")
            elif action == "manipulate":
                lang_parts.append(f"{action} the {obj_ref}")
        
        ep_meta["lang"] = " and ".join(lang_parts) if lang_parts else "complete the task"
        
        return ep_meta
