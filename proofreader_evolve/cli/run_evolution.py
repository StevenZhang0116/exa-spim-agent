"""Entry point for fixed-pool candidate precision evolution."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    from proofreader_evolve.cli.run_precision_evolution import main as run
    return run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
