"""Surgery-specific inference helpers for interactive world model simulation."""

from __future__ import annotations

from typing import Dict, List, Optional

# Latent action indices are learned unsupervised; these labels describe the
# intended semantic space for interactive surgery simulation.
LAPAROSCOPIC_ACTION_LEGEND: Dict[int, str] = {
    0: "idle / hold position",
    1: "grasp / close jaws",
    2: "release / open jaws",
    3: "move instrument left",
    4: "move instrument right",
    5: "move instrument up",
    6: "move instrument down",
    7: "camera pan / retract",
}

ROBOTIC_ACTION_LEGEND: Dict[int, str] = {
    0: "hold / no motion",
    1: "left arm: grasp",
    2: "left arm: release",
    3: "right arm: grasp",
    4: "right arm: release",
    5: "bilateral retraction",
    6: "camera arm adjustment",
    7: "cautery / cut motion",
}


def get_action_legend(surgery_type: str, n_actions: int) -> Dict[int, str]:
    base = ROBOTIC_ACTION_LEGEND if surgery_type == "robotic" else LAPAROSCOPIC_ACTION_LEGEND
    legend = {}
    for i in range(n_actions):
        legend[i] = base.get(i, f"latent motion {i}")
    return legend


def print_action_legend(legend: Dict[int, str]) -> None:
    print("\n=== Surgical Action Controls ===")
    print("Latent actions are inferred from video during training.")
    print("At inference, select an action index to steer the simulated procedure:\n")
    for idx, label in sorted(legend.items()):
        print(f"  [{idx}] {label}")
    print()


def prompt_surgical_action(
    step: int,
    n_actions: int,
    legend: Dict[int, str],
    default_action: Optional[int] = None,
) -> int:
    print_action_legend(legend)
    prompt = f"Step {step + 1}: enter action [0..{n_actions - 1}]"
    if default_action is not None:
        prompt += f" (Enter = {default_action})"
    prompt += ": "

    user_input = input(prompt).strip()
    if not user_input and default_action is not None:
        return default_action
    if not user_input.isdigit() or not (0 <= int(user_input) < n_actions):
        raise ValueError(f"Invalid action. Enter an integer in [0, {n_actions - 1}]")
    return int(user_input)


def describe_simulation_mode(
    surgery_type: str,
    n_actions: int,
    generation_steps: int,
    context_window: int,
) -> None:
    print("\n=== Surgical World Model Simulation ===")
    print(f"Surgery type: {surgery_type}")
    print(f"Context window: {context_window} frames")
    print(f"Generation steps: {generation_steps}")
    print(f"Latent action vocabulary: {n_actions}")
    print(
        "\nThis simulator autoregressively predicts laparoscopic frames "
        "conditioned on inferred instrument-motion tokens (Genie-style latent actions)."
    )
