REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .ExpoComm_controller import ExpoCommMAC
from .ExpoComm_bvme_controller import ExpoCommBMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["ExpoComm_mac"] = ExpoCommMAC
REGISTRY["ExpoComm_bvme_mac"] = ExpoCommBMAC
