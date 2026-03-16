import abc

from pathlib import Path

from proteus.types import List, Dict


class DownloadMethodBase(abc.ABC):
	checkpoint_path: Path

	@abc.abstractmethod
	def download(self, profiler=None) -> List[Dict]: ...
