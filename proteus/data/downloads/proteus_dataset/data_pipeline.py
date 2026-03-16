'''
entry point for the experimental structure data collection and cleaning pipeline
'''

import importlib
import logging
import shutil

from pathlib import Path
from cloudpathlib import S3Path
import pyarrow as pa
import pyarrow.parquet as pq

from proteus.utils.s3_utils import upload_bytes_to_s3_sync
from proteus.data.data_constants import DataPath, IndexCol, ClusterInputType, cluster_col_name
from proteus.data.downloads.proteus_dataset.conf.pipeline import DataPipelineBaseCfg
from proteus.data.downloads.proteus_dataset.download.base import DownloadMethodBase
from proteus.data.downloads.proteus_dataset.cluster.base import ClusterMethodBase


def _load_impl_cls(cfg):
	"""resolve _impl_cls dotted string to the actual class"""
	module_path, cls_name = cfg._impl_cls.rsplit(".", 1)
	module = importlib.import_module(module_path)
	return getattr(module, cls_name)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class DataPipeline:
	def __init__(self, cfg: DataPipelineBaseCfg):
		# instantiate clustering methods from configs
		self.cluster_methods: list[ClusterMethodBase] = [
			_load_impl_cls(method_cfg)(method_cfg) for method_cfg in cfg.cluster_methods
		]
		assert len(self.cluster_methods) >= 1, "at least one clustering method required"

		# union of required input types across all methods
		self.required_inputs: set[ClusterInputType] = set()
		for m in self.cluster_methods:
			self.required_inputs |= m.required_inputs

		# instantiate download methods from configs
		self.download_methods: list[DownloadMethodBase] = [
			_load_impl_cls(method_cfg)(method_cfg, self.required_inputs) for method_cfg in cfg.download_methods
		]

		self.s3_path = S3Path(cfg.s3_path)
		self.local_path = Path(cfg.local_path)

	def download(self, profiler=None):
		index_rows = []
		for method in self.download_methods:
			index_rows.extend(method.download(profiler=profiler))
		return index_rows

	def run(self, profiler=None):
		index_rows = self.download(profiler=profiler)
		index_path = self.local_path / DataPath.INDEX
		index_path.parent.mkdir(parents=True, exist_ok=True)

		# build columnar dict from list of row dicts
		columns = {}
		for key in index_rows[0]:
			columns[key] = [row[key] for row in index_rows]

		table = pa.table(columns)
		pq.write_table(table, index_path)
		logger.info(f"saved index with {len(index_rows)} rows to {index_path}")

		# phase 1: create all raw DBs that need building
		for method in self.cluster_methods:
			need_raw_db = any(not method.cluster_tsv_path(t).exists() for t in method.thresholds)
			if need_raw_db and not method.has_raw_db():
				method.create_db()

		# phase 2: delete raw input files (all methods have built their DBs)
		for input_type in self.required_inputs:
			input_dir = self.local_path / f"raw_{input_type}"
			if input_dir.exists():
				shutil.rmtree(input_dir)

		# phase 3: cluster per method, per threshold
		for method in self.cluster_methods:
			need_raw_db = any(not method.cluster_tsv_path(t).exists() for t in method.thresholds)

			for threshold in method.thresholds:
				col_name = cluster_col_name(method.method, threshold)

				if method.cluster_tsv_path(threshold).exists():
					logger.info(f"found existing {method.cluster_tsv_path(threshold).name}, skipping {method.method} for threshold {threshold}")
					clusters = method.load_clusters(threshold)
				else:
					if not method.has_cluster_db(threshold):
						method.run_cluster(threshold)
					clusters = method.parse_clusters(threshold)

				cluster_ids = [
					clusters.get(f"{pdb}_{chain}", "")
					for pdb, chain in zip(columns[IndexCol.PDB], columns[IndexCol.CHAIN])
				]
				table = table.append_column(col_name, pa.array(cluster_ids))

			# raw db no longer needed after all thresholds for this method
			if need_raw_db:
				method.cleanup_raw_db()

		pq.write_table(table, index_path)

		# upload index to s3
		s3_index_path = self.s3_path / DataPath.INDEX
		upload_bytes_to_s3_sync(index_path.read_bytes(), s3_index_path)
		logger.info(f"uploaded index to {s3_index_path}")

		# cleanup only after S3 upload succeeds
		for method in self.cluster_methods:
			method.cleanup_tsvs()
		for method in self.download_methods:
			if method.checkpoint_path.exists():
				method.checkpoint_path.unlink()
				logger.info(f"removed checkpoint file {method.checkpoint_path}")
