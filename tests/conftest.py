import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent
# make scripts/ and tests/ importable from test modules
sys.path.insert(0, str(_root.parent / "scripts"))
sys.path.insert(0, str(_root))
