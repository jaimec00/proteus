use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;
use xxhash_rust::xxh64::xxh64;

#[derive(serde::Deserialize)]
struct HashKwargs {
	seed: u64,
}

#[polars_expr(output_type = UInt64)]
fn stable_hash(inputs: &[Series], kwargs: HashKwargs) -> PolarsResult<Series> {
	let ca = inputs[0].str()?;
	let out: ChunkedArray<UInt64Type> = ca.apply_nonnull_values_generic(
		DataType::UInt64,
		|s| xxh64(s.as_bytes(), kwargs.seed),
	);
	Ok(out.into_series())
}
