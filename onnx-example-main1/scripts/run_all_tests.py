import subprocess
from pathlib import Path

script_dir = Path(__file__).parent

print("Running partition tests...")
subprocess.run(["python", str(script_dir / "partition"
".py")])

print("\nRunning metamorphic tests...")
subprocess.run(["python", str(script_dir / "metamorhpic.py")])