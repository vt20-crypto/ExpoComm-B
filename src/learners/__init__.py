from .q_learner import QLearner, AuxQLearner, ContQLearner
from .bvme_q_learner import BVMEQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner

# PAC-DCG requires torch_scatter which may not be available
try:
    from .actor_critic_pac_dcg_learner import PACDCGLearner
    _has_pac_dcg = True
except ImportError:
    _has_pac_dcg = False

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["aux_q_learner"] = AuxQLearner  # customized
REGISTRY["cont_q_learner"] = ContQLearner  # customized
REGISTRY["bvme_q_learner"] = BVMEQLearner  # ExpoComm-B
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
if _has_pac_dcg:
    REGISTRY["pac_dcg_learner"] = PACDCGLearner

