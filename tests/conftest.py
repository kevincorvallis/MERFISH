import pathlib
import sys

# make scripts/ importable from tests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
