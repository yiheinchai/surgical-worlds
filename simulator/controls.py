"""Keyboard and UI control mappings for surgical simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LAPAROSCOPIC_CONTROLS: Dict[str, int] = {
    "idle": 0, "grasp": 1, "release": 2, "left": 3, "right": 4,
    "up": 5, "down": 6, "camera": 7,
}

ROBOTIC_CONTROLS: Dict[str, int] = {
    "idle": 0, "left_grasp": 1, "left_release": 2, "right_grasp": 3,
    "right_release": 4, "retract": 5, "camera": 6, "cautery": 7,
}

KEYBOARD_BINDINGS_LAPAROSCOPIC: Dict[str, str] = {
    " ": "idle", "g": "grasp", "r": "release", "a": "left", "d": "right",
    "w": "up", "s": "down", "c": "camera",
}

KEYBOARD_BINDINGS_ROBOTIC: Dict[str, str] = {
    " ": "idle", "q": "left_grasp", "e": "right_grasp", "z": "left_release",
    "x": "right_release", "w": "retract", "c": "camera", "f": "cautery",
}

@dataclass
class ControlScheme:
    name: str
    actions: Dict[str, int]
    keyboard: Dict[str, str]

    def action_id(self, control_name: str) -> int:
        if control_name not in self.actions:
            raise KeyError(f"Unknown control '{control_name}'.")
        return self.actions[control_name]

    def control_buttons(self) -> List[Tuple[str, str, int]]:
        labels = {
            "idle": "Hold", "grasp": "Grasp", "release": "Release",
            "left": "Left", "right": "Right", "up": "Up", "down": "Down", "camera": "Camera",
            "left_grasp": "L Grasp", "left_release": "L Release",
            "right_grasp": "R Grasp", "right_release": "R Release",
            "retract": "Retract", "cautery": "Cautery",
        }
        return [(name, labels.get(name, name), aid) for name, aid in self.actions.items()]

    def help_text(self) -> str:
        lines = [f"**{self.name} controls**\n"]
        for key, control in sorted(self.keyboard.items(), key=lambda x: x[1]):
            lines.append(f"- `{key}` -> {control.replace('_', ' ').title()}")
        return "\n".join(lines)


def get_control_scheme(surgery_type: str) -> ControlScheme:
    if surgery_type == "robotic":
        return ControlScheme("Robotic Laparoscopic", ROBOTIC_CONTROLS, KEYBOARD_BINDINGS_ROBOTIC)
    return ControlScheme("Manual Laparoscopic", LAPAROSCOPIC_CONTROLS, KEYBOARD_BINDINGS_LAPAROSCOPIC)
