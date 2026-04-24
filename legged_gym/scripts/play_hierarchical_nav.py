import os

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.task_registry import get_args, task_registry


def _load_jit_policy(policy_path, device, label):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"{label} policy not found: {policy_path}")

    policy = torch.jit.load(policy_path, map_location=device).to(device)
    policy.eval()
    return policy


def _compose_low_level_actions(env, locomotion_policy, recovery_policy):
    student_obs = env.get_student_obs().detach()
    locomotion_actions = locomotion_policy(student_obs)
    protective_mask, recovery_mask = env.get_low_level_masks()

    actions = locomotion_actions
    if protective_mask.any() or recovery_mask.any():
        actions = actions.clone()

    if protective_mask.any():
        protective_actions = env.get_protective_actions()
        actions[protective_mask] = protective_actions[protective_mask]

    if recovery_mask.any():
        recovery_actions = recovery_policy(student_obs)
        actions[recovery_mask] = recovery_actions[recovery_mask]

    return actions


def play(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.is_train = True
    env_cfg.env.episode_length_s = 60
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

    nav_obs = env.get_nav_observations()
    max_steps = int(env.max_episode_length.item()) if hasattr(env.max_episode_length, "item") else int(env.max_episode_length)

    for step in range(max_steps):
        with torch.inference_mode():
            nav_actions = nav_policy.act(nav_obs)
            env.apply_navigation_actions(nav_actions)
            low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
            nav_obs, _, nav_rewards, nav_dones, _, _ = env.step(low_level_actions)

        if step % 50 == 0:
            status = env.get_navigation_status(0)
            command = env.nav_command_buffer[0]
            print(
                f"step={step:04d} "
                f"x={status['local_x']:.2f} "
                f"y={status['local_y']:.2f} "
                f"goal_dist={status['goal_distance']:.2f} "
                f"track_err={status['track_error']:.2f} "
                f"clearance={status['min_clearance']:.2f} "
                f"cmd=({command[0].item():.2f}, {command[2].item():.2f})"
            )

        if bool(nav_dones[0].item()):
            status = env.get_navigation_status(0)
            print(
                "episode reset | "
                f"goal={status['goal_reached']} "
                f"collision={status['collision']} "
                f"out_of_bounds={status['out_of_bounds']} "
                f"reward={nav_rewards[0].item():.2f}"
            )


if __name__ == "__main__":
    args = get_args()
    args.task = "roll_robot_r_hierarchical_nav"
    args.headless = False
    play(args)
