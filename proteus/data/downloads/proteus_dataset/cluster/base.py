import abc

from pathlib import Path
from typing import ClassVar

from proteus.types import Dict
from proteus.data.data_constants import ClusteringMethod, ClusterInputType


class ClusterMethodBase(abc.ABC):
	method: ClassVar[ClusteringMethod]
	required_inputs: ClassVar[set[ClusterInputType]]

	@property
	@abc.abstractmethod
	def thresholds(self) -> list[float]: ...

	@abc.abstractmethod
	def create_db(self) -> None: ...

	@abc.abstractmethod
	def has_raw_db(self) -> bool: ...

	@abc.abstractmethod
	def has_cluster_db(self, threshold: float) -> bool: ...

	@abc.abstractmethod
	def run_cluster(self, threshold: float) -> None: ...

	@abc.abstractmethod
	def parse_clusters(self, threshold: float) -> Dict[str, str]: ...

	@abc.abstractmethod
	def load_clusters(self, threshold: float) -> Dict[str, str]: ...

	@abc.abstractmethod
	def cluster_tsv_path(self, threshold: float) -> Path: ...

	@abc.abstractmethod
	def cleanup_raw_db(self) -> None: ...

	@abc.abstractmethod
	def cleanup_tsvs(self) -> None: ...
