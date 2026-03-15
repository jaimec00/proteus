from proteus.data.downloads.proteus_dataset.conf.download import register_download
from proteus.data.downloads.proteus_dataset.conf.cluster import register_cluster
from proteus.data.downloads.proteus_dataset.conf.hydra import register_hydra

def register_download_configs():
	register_download()
	register_cluster()
	register_hydra()
