#!/usr/bin/env python3
"""Launch the SurgicalWorlds playable simulator (Genie 3 for surgery)."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    subprocess.run([sys.executable, os.path.join(ROOT, "app", "surgery_simulator.py")] + sys.argv[1:])
