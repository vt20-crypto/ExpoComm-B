"""
ExpoComm-B Controller: Multi-Agent Controller with BVME support.

Extends ExpoCommMAC to pass test_mode to the agent (needed for BVME to
switch between stochastic sampling during training and deterministic
mean during evaluation), and to collect KL loss from the BVME module.
"""

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from .ExpoComm_controller import ExpoCommMAC, get_exp_neighbors


class ExpoCommBMAC(ExpoCommMAC):
    """
    Multi-Agent Controller for ExpoComm-B.

    Same as ExpoCommMAC but:
    1. Passes test_mode flag to agent for BVME (eval uses mean, train uses sample)
    2. Collects and accumulates KL loss across timesteps
    """

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        # KL loss accumulated across timesteps within an episode
        self.kl_loss_accum = th.tensor(0.0)
        self.kl_step_count = 0

    def reset_kl_loss(self):
        """Reset KL accumulator at the start of each training batch."""
        self.kl_loss_accum = th.tensor(0.0)
        self.kl_step_count = 0

    def get_avg_kl_loss(self):
        """Get average KL loss across all timesteps."""
        if self.kl_step_count == 0:
            return th.tensor(0.0)
        return self.kl_loss_accum / self.kl_step_count

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs, topk_indices = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # ExpoComm-B agent returns 4 values (not 3): q, hidden, z, kl_loss
        agent_outs, self.hidden_states, z, kl_loss = self.agent(
            agent_inputs, self.hidden_states, topk_indices, test_mode=test_mode
        )

        # Accumulate KL loss (only during training)
        if not test_mode:
            self.kl_loss_accum = self.kl_loss_accum.to(kl_loss.device) + kl_loss
            self.kl_step_count += 1

        # Auxiliary forward: predict state from compressed message z
        states_predicted = self.agent.aux_forward(z)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(
            ep_batch.batch_size, self.n_agents, -1
        ), states_predicted.view(ep_batch.batch_size, self.n_agents, -1)
