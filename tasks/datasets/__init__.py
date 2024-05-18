from .base_dataset import MetaDataset

# import the dataset class here
from .r2r import R2RDataset
from .cvdn import CVDNDataset
from .soon import SOONDataset
from .eqa import EQADataset
from .reverie import REVERIEDataset
from .r2r_aug import R2RAugDataset
from .reverie_aug import REVERIEAugDataset
from .llava import LLaVADataset
from .scanqa import ScanQADataset
from .roomtour import RoomTourDataset
from .tour3d_dataset import Tour3DDataset

def load_dataset(name, *args, **kwargs):
    cls = MetaDataset.registry[name]
    return cls(*args, **kwargs)