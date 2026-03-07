from proteus.data.downloads.proteus_dataset.conf.experimental_download import register_experimental_download
from proteus.data.downloads.proteus_dataset.conf.foldseek import register_foldseek
from proteus.data.downloads.proteus_dataset.conf.hydra import register_hydra

def register_download_configs():
    register_experimental_download()
    register_foldseek()
    register_hydra()
