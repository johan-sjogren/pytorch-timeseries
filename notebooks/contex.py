from pathlib import Path, PurePath
import sys

lib_path = Path(PurePath(__file__).parent, "..", "src")
sys.path.append(str(lib_path.resolve()))
