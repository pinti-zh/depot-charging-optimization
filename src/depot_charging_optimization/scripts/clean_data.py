import logging
import os

import click
import pandas as pd
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("optimize")


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
    df = pd.read_csv(source, sep=sep, dtype=str, keep_default_na=False)  # read data as string

    logger.info(f"loaded [magenta]{source}[/magenta]\n  {len(df.columns)} columns and {len(df)} rows")

    none_value_placeholders = ["NA"]
    for col in df.columns:
        found_placeholder = False
        found_comma_floats = False
        for none_value_placeholder in none_value_placeholders:
            if none_value_placeholder in df[col]:
                found_placeholder = True
                df[col] = df[col].replace(none_value_placeholder, None)
        if any(df[col].map(is_comma_float)):
            found_comma_floats = True
            df[col] = df[col].map(lambda x: x.replace(",", "."))
            df[col] = df[col].map(add_decimal_point_to_int)
        if not (found_placeholder or found_comma_floats):
            logger.info(f"  column [cyan]{col}[/cyan] [bold green]ok")
        elif found_placeholder and not found_comma_floats:
            logger.info(f"  column [cyan]{col}[/cyan] [gold1]replaced none value placeholders")
        elif not found_placeholder and found_comma_floats:
            logger.info(f"  column [cyan]{col}[/cyan] [orange1]replaced floats that had comma decimal point")
        elif found_placeholder and found_comma_floats:
            logger.info(
                f"  column [cyan]{col}[/cyan] [gold1]replaced none value placeholders [italic white]and [orange1]replaced floats that had comma decimal point"
            )
    output_dir = os.path.dirname(target)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(target, index=False)

    logger.info(f"saved data as [green]{target}")
