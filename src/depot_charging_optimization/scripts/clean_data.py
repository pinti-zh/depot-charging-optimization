import os
from typing import Optional

import polars as pl
import click
from rich import print as printr


def is_comma_float(string):
    if "," in string:
        try:
            float(string.replace(",", "."))
            return True
        except ValueError:
            return False
    else:
        return False


def add_decimal_point_to_int(string):
    try:
        x = int(string)
        return f"{x}.0"
    except ValueError:
        return string


@click.command()
@click.argument("source", type=str)
@click.argument("target", type=str)
@click.option("--sep", "-s", type=str, default=";", help="csv file seperator")
def clean_data(source, target, sep):
    data = pl.read_csv(source, separator=sep, infer_schema=False)

    printr(f"loaded [magenta]{source}[/magenta]\n  {len(data.columns)} columns and {len(data)} rows")
    printr("  head", data.head())

    none_value_placeholders = ["NA"]
    for col in data.columns:
        found_placeholder = False
        found_comma_floats = False
        for none_value_placeholder in none_value_placeholders:
            if none_value_placeholder in data[col]:
                found_placeholder = True
                data = data.with_columns(pl.col(col).replace(none_value_placeholder, None))
        if any(data[col].map_elements(is_comma_float, return_dtype=pl.Boolean)):
            found_comma_floats = True
            data = data.with_columns(pl.col(col).str.replace(",", "."))
            data = data.with_columns(
                pl.col(col).map_elements(add_decimal_point_to_int, return_dtype=Optional[pl.String])
            )
        printr(f"    column [cyan]{col}[/cyan]  ", end="")
        if not (found_placeholder or found_comma_floats):
            printr("[bold green]ok")
        elif found_placeholder and not found_comma_floats:
            printr("[gold1]replaced none value placeholders")
        elif not found_placeholder and found_comma_floats:
            printr("[orange1]replaced floats that had comma decimal point")
        elif found_placeholder and found_comma_floats:
            printr(
                "[gold1]replaced none value placeholders [italic white]and [orange1]replaced floats that had comma decimal point"
            )
    output_dir = os.path.dirname(target)
    os.makedirs(output_dir, exist_ok=True)
    data.write_csv(target)

    printr(f"saved data as [green]{target}")
