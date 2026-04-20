from .coma import COMACritic
from .centralV import CentralVCritic
from .coma_ns import COMACriticNS
from .centralV_ns import CentralVCriticNS
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .pac_ac_ns import PACCriticNS

# PAC-DCG requires torch_scatter which may not be available
try:
    from .pac_dcg_ns import DCGCriticNS
    _has_dcg = True
except ImportError:
    _has_dcg = False

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["coma_critic_ns"] = COMACriticNS
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["pac_critic_ns"] = PACCriticNS
if _has_dcg:
    REGISTRY["pac_dcg_critic_ns"] = DCGCriticNS
