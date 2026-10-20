"""Feature measurement drift stress test for few-shot methods; not FGSM/PGD"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids_xgb.pipeline import main

if __name__ == "__main__":
    main(["run", "--stage", "E6", *sys.argv[1:]])
