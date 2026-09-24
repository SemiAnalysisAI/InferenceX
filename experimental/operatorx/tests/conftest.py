import sys
from pathlib import Path

# Make `operatorx` importable however pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
