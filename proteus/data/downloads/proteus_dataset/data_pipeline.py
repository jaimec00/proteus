'''
entry point for the experimental structure data collection and cleaning pipeline
'''

import logging

from pathlib import Path
from cloudpathlib import S3Path
import pyarrow as pa
import pyarrow.parquet as pq

from proteus.utils.s3_utils import upload_bytes_to_s3_sync
from proteus.data.data_constants import DataPath, IndexCol, ClusteringMethod, cluster_col_name
from proteus.data.downloads.proteus_dataset.conf.pipeline import DataPipelineBaseCfg
from proteus.data.downloads.proteus_dataset.download_experimental import ExperimentalDataDownload
from proteus.data.downloads.proteus_dataset.struct_clust import FoldSeek

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class DataPipeline:
	def __init__(self, cfg: DataPipelineBaseCfg):
		self.experimental_dl = ExperimentalDataDownload(cfg.experimental_dl)
		self.foldseek = FoldSeek(cfg.foldseek)
		self.s3_path = S3Path(cfg.s3_path)
		self.local_path = Path(cfg.local_path)

	def download(self):
		# TODO: add other download sources
		return self.experimental_dl.download()

	def run(self):
		index_rows = self.download()
		index_path = self.local_path / DataPath.INDEX
		index_path.parent.mkdir(parents=True, exist_ok=True)

		# build columnar dict from list of row dicts
		columns = {}
		for key in index_rows[0]:
			columns[key] = [row[key] for row in index_rows]

		table = pa.table(columns)
		pq.write_table(table, index_path)
		logger.info(f"saved index with {len(index_rows)} rows to {index_path}")

		# clustering: loop over thresholds, reusing the raw db across all
		thresholds = self.foldseek.tmscore_thresholds
		need_raw_db = any(
			not self.foldseek.cluster_tsv_path(t).exists() for t in thresholds
		)

		if need_raw_db:
			has_raw_db = any(self.foldseek.db_path.glob(self.foldseek.raw_db_path.name + "*"))
			if not has_raw_db:
				self.foldseek.create_db()

		for threshold in thresholds:
			col_name = cluster_col_name(ClusteringMethod.FOLDSEEK, threshold)

			if self.foldseek.cluster_tsv_path(threshold).exists():
				logger.info(f"found existing {self.foldseek.cluster_tsv_path(threshold).name}, skipping foldseek for threshold {threshold}")
				clusters = self.foldseek.load_clusters(threshold)
			else:
				has_cluster_db = any(self.foldseek.db_path.glob(self.foldseek.cluster_db_path(threshold).name + "*"))
				if has_cluster_db:
					logger.info(f"found existing cluster db for threshold {threshold}, skipping cluster step")
				else:
					self.foldseek.run_cluster(threshold)
				clusters = self.foldseek.parse_clusters(threshold)

			cluster_ids = [
				clusters.get(f"{pdb}_{chain}", "")
				for pdb, chain in zip(columns[IndexCol.PDB], columns[IndexCol.CHAIN])
			]
			table = table.append_column(col_name, pa.array(cluster_ids))

		# raw db no longer needed after all thresholds
		if need_raw_db:
			self.foldseek.cleanup_raw_db()

		pq.write_table(table, index_path)

		# upload index to s3
		s3_index_path = self.s3_path / DataPath.INDEX
		upload_bytes_to_s3_sync(index_path.read_bytes(), s3_index_path)
		logger.info(f"uploaded index to {s3_index_path}")

		# cleanup only after S3 upload succeeds
		self.foldseek.cleanup_tsvs()
		if self.experimental_dl.checkpoint_path.exists():
			self.experimental_dl.checkpoint_path.unlink()
			logger.info("removed checkpoint file after successful completion")
