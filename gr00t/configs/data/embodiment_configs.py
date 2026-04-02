from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


MODALITY_CONFIGS = {
    ##### Pre-registered posttrain configurations #####
    "unitree_g1": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(30)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base_height_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # navigate_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "libero_panda": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "oxe_widowx": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image_0"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "pad",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 8)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
            mean_std_embedding_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "oxe_google": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
                "rw",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 8)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
            mean_std_embedding_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "behavior_r1_pro": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "observation.images.rgb.head_256_256",
                "observation.images.rgb.left_wrist_256_256",
                "observation.images.rgb.right_wrist_256_256",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "robot_pos",  # 3
                "robot_ori_cos",  # 3
                "robot_ori_sin",  # 3
                "robot_2d_ori",  # 1
                "robot_2d_ori_cos",  # 1
                "robot_2d_ori_sin",  # 1
                "robot_lin_vel",  # 3
                "robot_ang_vel",  # 3
                "arm_left_qpos",  # 7
                "arm_left_qpos_sin",  # 7
                "arm_left_qpos_cos",  # 7
                "eef_left_pos",  # 3
                "eef_left_quat",  # 4
                "gripper_left_qpos",  # 2
                "arm_right_qpos",  # 7
                "arm_right_qpos_sin",  # 7
                "arm_right_qpos_cos",  # 7
                "eef_right_pos",  # 3
                "eef_right_quat",  # 4
                "gripper_right_qpos",  # 2
                "trunk_qpos",  # 4
            ],
        ),  # dim = 82
        "action": ModalityConfig(
            delta_indices=list(range(0, 32)),
            modality_keys=[
                "base",
                "torso",
                "left_arm",
                "left_gripper",
                "right_arm",
                "right_gripper",
            ],
            action_configs=[
                # base
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # torso
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="trunk_qpos",
                ),
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="arm_left_qpos",
                ),
                # left_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="arm_right_qpos",
                ),
                # right_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "r1lite": {
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                # "left_arm",
                # "right_arm",
                "left_gripper",
                "right_gripper",
                "torso_state_joint",
                "left_ee_pose_gripper_base",
                "right_ee_pose_gripper_base",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(15)),
            modality_keys=[
                # "left_arm",
                # "right_arm",
                "left_gripper",
                "right_gripper",
                "torso_action_joint",
                "left_ee_pose_gripper_base",
                "right_ee_pose_gripper_base",
            ],
            action_configs=[
                # left_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # torso.positions (state_key="torso")
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="torso_state_joint",
                ),
                # left_ee_pose_gripper_base
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_QUAT_XYZW,
                ),
                # right_ee_pose_gripper_base
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_QUAT_XYZW,
                ),
            ],
        ),
    },
    "oxe_droid": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "exterior_image_1_left",
                "wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 32)),
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
            action_configs=[
                # joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
            ],
        ),
    },
    "robochallenge_single_arm": {
        # Unified RoboChallenge manipulation config for ARX5, UR5, FRANKA, ALOHA.
        # State: unified 8D per arm (x, y, z, qx, qy, qz, qw, gripper).
        # Single-arm robots populate `state` / `action`, dual-arm ALOHA uses left/right fields.
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ee_pose_gripper_base",  # (8,) for single-arm robots, zero for ALOHA
            ],
        ),
        "action": ModalityConfig(
            # Action chunks are 8 steps long by default.
            delta_indices=list(range(30)),
            modality_keys=[
                "ee_pose_gripper_base",  # (8,) single-arm target action
            ],
            action_configs=[
                # action
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_QUAT_XYZW,
                ),
            ],
        ),
    },
    "robochallenge_dual_arm": {
        # Unified RoboChallenge manipulation config for ARX5, UR5, FRANKA, ALOHA.
        # State: unified 8D per arm (x, y, z, qx, qy, qz, qw, gripper).
        # Single-arm robots populate `state` / `action`, dual-arm ALOHA uses left/right fields.
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_ee_pose_gripper_base",  # (8,) for ALOHA left arm, zero for single-arm robots
                "right_ee_pose_gripper_base",  # (8,) for ALOHA right arm, zero for single-arm robots
            ],
        ),
        "action": ModalityConfig(
            # Action chunks are 8 steps long by default.
            delta_indices=list(range(30)),
            modality_keys=[
                "left_ee_pose_gripper_base",  # (8,) ALOHA left arm target action
                "right_ee_pose_gripper_base",  # (8,) ALOHA right arm target action
            ],
            action_configs=[
                # action_left
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_QUAT_XYZW,
                ),
                # action_right
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_QUAT_XYZW,
                ),
            ],
        ),
    },
    "unitree_g1_ee": {
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_ee_pose_gripper_base",
                "right_ee_pose_gripper_base",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(30)),
            modality_keys=[
                "left_ee_pose_gripper_base",
                "right_ee_pose_gripper_base",
            ],
            action_configs=[
                # action_ee_left
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_RPY,
                    state_key="left_ee_pose_gripper_base",
                ),
                # action_ee_right
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_RPY,
                    state_key="right_ee_pose_gripper_base",
                ),
            ],
        ),
    },
    "oxe_new": {
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ee_pose_gripper_base",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(3)),
            modality_keys=[
                "ee_pose_gripper_base",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_RPY,
                    state_key="ee_pose_gripper_base",
                ),
            ],
        ),
    },
}


def register_modality_config(
    config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
):
    assert embodiment_tag.value not in MODALITY_CONFIGS, (
        f"Embodiment tag {embodiment_tag} already registered"
    )
    MODALITY_CONFIGS[embodiment_tag.value] = config
