import argparse

import polars as pl
from rich import print as printr

from optimization.optimization import OptimizationModel
from optimization.utils import (
    plot_solution,
    print_model_summary,
    print_params,
    print_solution,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True, help="path to data file")
    parser.add_argument("--energy_price", "-ep", type=str, default="data/energy_price.csv", help="energy price file")
    parser.add_argument(
        "--verbosity", "-v", choices=[0, 1, 2], type=int, default=1, help="0 = only output, 1 = verbose, 2 = debug"
    )
    parser.add_argument("--ce_function", "-cef", type=str, choices=["constant", "quadratic", "one"], default="one")
    args = parser.parse_args()

    data = pl.read_csv(args.data)
    energy_price = pl.read_csv(args.energy_price)

    # print params
    if args.verbosity >= 1:
        # print_params(data)
        printr(data.head())

    # optimization
    opt_model = OptimizationModel(data, energy_price, "optimization")
    opt_model.set_variables()
    opt_model.set_constraints(ce_function_type=args.ce_function)
    opt_model.set_objective()

    # display model
    print_model_summary(opt_model, verbosity=args.verbosity)

    # solve
    opt_model.optimize()

    # print solution
    print_solution(opt_model, verbosity=args.verbosity)
    plot_solution(opt_model)


if __name__ == "__main__":
    main()
