
import enum
import torch
from proteus.types import T, Float, Int, Bool
from proteus.utils.struct_utils import get_CA_raw_and_CB_unit, get_backbone

CONSTRUCT_FUNCTION = None

class ConstructFunctionNames(enum.StrEnum):
	PROTEUS = "proteus"

class ConstructRegistry:

	@staticmethod
	def set_construct_function(construct_function: str):
		global CONSTRUCT_FUNCTION
		ConstructRegistry._assert_not_set()
		assert construct_function in ConstructFunctionNames
		CONSTRUCT_FUNCTION = construct_function

	@staticmethod
	def needs_pair_cuseqlens():
		ConstructRegistry._assert_set()
		return CONSTRUCT_FUNCTION in [ConstructFunctionNames.PAIRFORMER]

	@staticmethod
	def construct(*args):
		ConstructRegistry._assert_set()
		return getattr(ConstructFunctions, CONSTRUCT_FUNCTION)(*args)

	@staticmethod
	def _assert_set():
		assert CONSTRUCT_FUNCTION is not None, f"never called ConstructRegistry.set_construct_function, CONSTRUCT_FUNCTION=None"

	@staticmethod
	def _assert_not_set():
		assert CONSTRUCT_FUNCTION is None, f"already called ConstructRegistry.set_construct_function, {CONSTRUCT_FUNCTION=}"


class ConstructFunctions:

	@staticmethod
	@torch.no_grad()
	def proteus(
		coords: Float[T, "L 14 3"], 
		labels: Int[T, "L"], 
		seq_idx: Int[T, "L"], 
		chain_idx: Int[T, "L"], 
		trgt_mask: Bool[T, "L"], 
		homo_mask: Bool[T, "L"],
		caa_mask: Bool[T, "L"],
		atom_mask: Bool[T, "L"]
	):

		coords_ca, coords_cb_unit = get_CA_raw_and_CB_unit(coords)
		seq_mask = homo_mask & ~trgt_mask
		loss_mask = caa_mask & trgt_mask

		return {
			"coords_ca": coords_ca,
			"coords_cb_unit": coords_cb_unit,
			"labels": labels,
			"seq_mask": seq_mask,
			"loss_mask": loss_mask,
		}