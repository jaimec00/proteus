from __future__ import annotations

from pathlib import Path
import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

PLUGIN_PATH = Path(__file__).parent

def stable_hash(expr: IntoExpr, seed: int = 0) -> pl.Expr:
	return register_plugin_function(
		plugin_path=PLUGIN_PATH,
		function_name="stable_hash",
		args=[expr],
		kwargs={"seed": seed},
		is_elementwise=True,
	)
