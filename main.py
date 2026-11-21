"""Project entry point.

python main.py             Original experiment and original results/plots.
python main.py --enhanced  Enhanced fixed-vs-adaptive comparison + five-fold CV.
python main.py --quick     Enhanced comparison without cross-validation.
python main.py --all       Both original and enhanced experiments.
"""

import argparse
from src.experiment import run_experiment, print_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bivariate decision tree experiments")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--enhanced", action="store_true", help="run adaptive enhancement and CV")
    mode.add_argument("--quick", action="store_true", help="run adaptive experiment without CV")
    mode.add_argument("--all", action="store_true", help="run both original and enhanced experiments")
    args = parser.parse_args()

    if not args.enhanced and not args.quick:
        results = run_experiment()
        print_summary(results)

    if args.enhanced or args.quick or args.all:
        from src.enhancement_demo import run_demo
        run_demo(include_cv=not args.quick)
