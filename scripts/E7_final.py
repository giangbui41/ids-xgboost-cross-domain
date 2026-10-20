"""E7: one evaluation on IoT FINAL using a configuration fixed before this run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ids_xgb.pipeline import main


if __name__ == "__main__":
    main(["run", "--stage", "E7", *sys.argv[1:]])
