"""
ExpoComm-B Agent: ExpoComm with BVME (Bandwidth-constrained Variational Message Encoding).

This agent extends the ExpoComm one-peer agent by adding a BVME variational
bottleneck on the communication channel. The exponential topology from ExpoComm
decides WHO to communicate with; BVME decides HOW MUCH information to transmit.

Architecture flow:
    obs → fc1(ReLU) → GRU → h (hidden state)
                             ↓
                   _communicate(neighbor, prev_msgs, h) → raw_msg
                             ↓
                   BVME: raw_msg → (μ, σ) → sample z  ← bandwidth bottleneck
                             ↓
                   [h ; z] → fc2 → Q-values
                             ↓
                   z → predict_net → predicted_state (auxiliary loss)

Design decisions:
    - The BVME bottleneck is placed AFTER message aggregation (after _communicate),
      BEFORE the Q-network. This is the "on-path" coupling from the BVME paper,
      ensuring the KL penalty constrains the representations that drive decisions.
    - The full raw_msg (not compressed z) is stored in hidden_info for the next
      timestep's communication, so the communication channel operates normally.
    - During evaluation, z = μ (deterministic, no sampling) for stability.
"""

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.bvme import BVMEModule


class ExpoCommBAgent(nn.Module):
    """
    ExpoComm-B agent with one-peer communication and BVME compression.

    Based on ExpoCommOAgent but with a BVME variational bottleneck on messages.
    """

    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim

        # --- Observation encoder ---
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        assert self.args.use_rnn

        # --- Message processing (same as ExpoCommOAgent) ---
        self.msg_processor = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
        )
        self.msg_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)

        # --- BVME compression module ---
        # compressed_dim: if bvme_compressed_dim is set, use it; otherwise use hidden_dim
        # (information-theoretic compression only, no dimensional reduction)
        self.compressed_dim = getattr(args, "bvme_compressed_dim", args.hidden_dim)
        self.bvme = BVMEModule(
            input_dim=args.hidden_dim,
            compressed_dim=self.compressed_dim,
            sigma_0=getattr(args, "bvme_sigma_0", 0.01),
            log_var_min=getattr(args, "bvme_log_var_min", -5.0),
            log_var_max=getattr(args, "bvme_log_var_max", 3.0),
        )

        # --- Recurrent hidden state ---
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)

        # --- Q-value head ---
        # Input: [h (hidden_dim) ; z (compressed_dim)]
        self.fc2 = nn.Sequential(
            nn.Linear(args.hidden_dim + self.compressed_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions),
        )

        # --- Auxiliary task: predict global state from compressed message ---
        self.state_dim = int(np.prod(args.state_shape))
        self.predict_net = nn.Sequential(
            nn.Linear(self.compressed_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, self.state_dim),
        )

    def init_hidden(self):
        """Initialize hidden state and message memory."""
        h = self.fc1.weight.new(1, self.args.hidden_dim).zero_()
        msg = th.zeros_like(h)
        hidden_info = (h, msg)
        return hidden_info

    def forward(self, inputs, hidden_info, topk_indices, test_mode=False):
        """
        Forward pass for one timestep.

        Args:
            inputs: Agent observations, shape (bs * n_agents, input_dim).
            hidden_info: Tuple of (h, msg) from previous timestep.
                h: GRU hidden state, shape (bs * n_agents, hidden_dim).
                msg: Previous message, shape (bs * n_agents, hidden_dim).
            topk_indices: Neighbor indices from exponential topology,
                shape (bs, n_agents) for one-peer mode.
            test_mode: If True, use deterministic BVME (z = μ).

        Returns:
            q: Q-values, shape (bs * n_agents, n_actions).
            hidden_info: Updated (h, msg) for next timestep.
            z: Compressed message for auxiliary loss, shape (bs * n_agents, compressed_dim).
            kl_loss: Scalar KL divergence from BVME (0.0 in test mode).
        """
        hidden_state, msg = hidden_info
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)

        # Encode observation
        x = F.relu(self.fc1(inputs))

        # Update hidden state
        h = self.rnn(x, h_in)

        # Communicate with selected neighbor (ExpoComm exponential topology)
        raw_msg = self._communicate(topk_indices, msg, h)

        # === BVME COMPRESSION (our addition) ===
        # Apply variational bottleneck to message BEFORE it enters Q-network.
        # This is "on-path" coupling: the KL penalty directly constrains
        # the representations that the Q-network uses for decisions.
        z, kl_loss = self.bvme(raw_msg, test_mode=test_mode)

        # Q-value computation uses compressed message z
        out = th.cat([h, z], dim=-1)
        q = self.fc2(out)

        # Store the FULL (uncompressed) message in hidden_info.
        # This is what neighbors will receive at the next timestep.
        # The bandwidth constraint is applied at the decision level (Q-network),
        # not at the communication channel level.
        hidden_info = (h, raw_msg)

        return q, hidden_info, z, kl_loss

    def aux_forward(self, z):
        """Predict global state from compressed message (auxiliary task)."""
        return self.predict_net(z)

    def _communicate(self, topk_indices, other_msg, ego_h):
        """
        One-peer communication (same as ExpoCommOAgent._communicate).

        At each timestep, receives a message from exactly one neighbor
        (selected by round-robin over exponential topology), processes it
        through an MLP and GRU to produce the aggregated message.

        Args:
            topk_indices: Shape (bs, n_agents) — one neighbor per agent.
            other_msg: Previous messages from all agents, shape (bs * n_agents, hidden_dim).
            ego_h: Current hidden state, shape (bs * n_agents, hidden_dim).

        Returns:
            m_aggregated: Processed message, shape (bs * n_agents, hidden_dim).
        """
        bs, n_agents = topk_indices.shape
        assert n_agents == self.n_agents

        # Own previous message (used as GRU hidden state)
        msg_ego = other_msg.reshape(bs * n_agents, self.args.hidden_dim)

        # Expand to gather neighbor messages
        other_msg = other_msg.reshape(bs, 1, n_agents, self.args.hidden_dim).expand(
            -1, n_agents, -1, -1
        )

        # Gather the selected neighbor's message
        topk_indices_expanded = topk_indices[:, :, None, None].expand(
            -1, -1, -1, self.args.hidden_dim
        )
        msg_received = other_msg.gather(dim=2, index=topk_indices_expanded)
        msg_received = msg_received[:, :, 0, :]

        # Process: [ego_h; received_msg] → MLP → GRU
        ego_h = ego_h.reshape(bs, n_agents, -1)
        msg_input = self.msg_processor(th.cat([ego_h, msg_received], dim=-1))
        msg_input = msg_input.reshape(bs * n_agents, -1)

        m_aggregated = self.msg_rnn(msg_input, msg_ego)

        return m_aggregated.reshape(bs * n_agents, -1)
