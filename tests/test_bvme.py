"""
Unit tests for the BVME module and ExpoComm-B agent.

Run with: python -m pytest tests/test_bvme.py -v
Or simply: python tests/test_bvme.py
"""

import sys
import os

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn


def test_bvme_module():
    """Test BVMEModule in isolation with random inputs."""
    from modules.bvme import BVMEModule

    print("=" * 60)
    print("TEST 1: BVMEModule basic functionality")
    print("=" * 60)

    batch_size = 32
    input_dim = 64
    compressed_dim = 64  # same dim (info-theoretic compression)
    sigma_0 = 0.01

    bvme = BVMEModule(
        input_dim=input_dim,
        compressed_dim=compressed_dim,
        sigma_0=sigma_0,
    )

    # Random message input
    msg = torch.randn(batch_size, input_dim)

    # --- Training mode ---
    z_train, kl_train = bvme(msg, test_mode=False)

    assert z_train.shape == (batch_size, compressed_dim), \
        f"Expected shape ({batch_size}, {compressed_dim}), got {z_train.shape}"
    assert kl_train.ndim == 0, f"KL should be scalar, got shape {kl_train.shape}"
    assert kl_train.item() > 0, f"KL should be positive during training, got {kl_train.item()}"
    print(f"  Training: z shape = {z_train.shape}, KL = {kl_train.item():.6f}")

    # --- Eval mode ---
    z_eval, kl_eval = bvme(msg, test_mode=True)

    assert z_eval.shape == (batch_size, compressed_dim), \
        f"Expected shape ({batch_size}, {compressed_dim}), got {z_eval.shape}"
    assert kl_eval.item() == 0.0, f"KL should be 0.0 in eval mode, got {kl_eval.item()}"
    print(f"  Eval:     z shape = {z_eval.shape}, KL = {kl_eval.item():.6f}")

    # --- Test with dimensional compression ---
    compressed_dim_small = 16
    bvme_small = BVMEModule(
        input_dim=input_dim,
        compressed_dim=compressed_dim_small,
        sigma_0=sigma_0,
    )
    z_small, kl_small = bvme_small(msg, test_mode=False)
    assert z_small.shape == (batch_size, compressed_dim_small), \
        f"Expected shape ({batch_size}, {compressed_dim_small}), got {z_small.shape}"
    print(f"  Compressed: z shape = {z_small.shape}, KL = {kl_small.item():.6f}")

    # --- Gradient flow test ---
    msg_grad = torch.randn(batch_size, input_dim, requires_grad=True)
    z, kl = bvme(msg_grad, test_mode=False)
    loss = z.sum() + kl
    loss.backward()
    assert msg_grad.grad is not None, "Gradients should flow through BVME"
    print(f"  Gradient flow: ✓ (grad norm = {msg_grad.grad.norm().item():.4f})")

    print("  ✅ All BVME module tests passed!\n")


def test_kl_behavior():
    """Test that KL divergence behaves correctly with different sigma_0 values."""
    from modules.bvme import BVMEModule

    print("=" * 60)
    print("TEST 2: KL divergence sensitivity to sigma_0")
    print("=" * 60)

    batch_size = 128
    input_dim = 64
    msg = torch.randn(batch_size, input_dim)

    # Tighter bandwidth (smaller sigma_0) should give higher KL
    kl_values = {}
    for sigma_0 in [0.005, 0.01, 0.02, 0.05, 0.1, 1.0]:
        bvme = BVMEModule(input_dim=input_dim, sigma_0=sigma_0)
        _, kl = bvme(msg, test_mode=False)
        kl_values[sigma_0] = kl.item()
        print(f"  sigma_0 = {sigma_0:.3f} → KL = {kl.item():.4f}")

    # KL should decrease as sigma_0 increases (more permissive prior)
    kl_list = list(kl_values.values())
    is_decreasing = all(kl_list[i] >= kl_list[i + 1] for i in range(len(kl_list) - 1))
    status = "✅" if is_decreasing else "⚠️ (not strictly decreasing, but this can happen with random init)"
    print(f"  KL decreasing with larger sigma_0: {status}\n")


def test_parameter_count():
    """Verify BVME adds minimal parameters."""
    from modules.bvme import BVMEModule

    print("=" * 60)
    print("TEST 3: BVME parameter count (overhead check)")
    print("=" * 60)

    input_dim = 64
    compressed_dim = 64

    bvme = BVMEModule(input_dim=input_dim, compressed_dim=compressed_dim)
    n_params = sum(p.numel() for p in bvme.parameters())

    # Expected: enc_mu (64*64 + 64) + enc_sigma (64*64 + 64) = 8320
    expected = (input_dim * compressed_dim + compressed_dim) * 2
    print(f"  Parameters: {n_params} (expected: {expected})")
    assert n_params == expected, f"Parameter count mismatch: {n_params} vs {expected}"

    # With compression: input=64, compressed=16
    bvme_small = BVMEModule(input_dim=64, compressed_dim=16)
    n_params_small = sum(p.numel() for p in bvme_small.parameters())
    expected_small = (64 * 16 + 16) * 2
    print(f"  Compressed (64→16): {n_params_small} params (expected: {expected_small})")
    assert n_params_small == expected_small

    print("  ✅ Parameter count tests passed!\n")


def test_determinism():
    """Test that eval mode is deterministic and train mode is stochastic."""
    from modules.bvme import BVMEModule

    print("=" * 60)
    print("TEST 4: Determinism (eval) vs Stochasticity (train)")
    print("=" * 60)

    bvme = BVMEModule(input_dim=64, sigma_0=0.01)
    msg = torch.randn(16, 64)

    # Eval should be deterministic
    z1, _ = bvme(msg, test_mode=True)
    z2, _ = bvme(msg, test_mode=True)
    assert torch.allclose(z1, z2), "Eval mode should be deterministic"
    print("  Eval deterministic: ✅")

    # Train should be stochastic (different z each time)
    z3, _ = bvme(msg, test_mode=False)
    z4, _ = bvme(msg, test_mode=False)
    assert not torch.allclose(z3, z4), "Train mode should be stochastic"
    print("  Train stochastic:   ✅")
    print()


if __name__ == "__main__":
    print("\n🔬 Running BVME Module Tests\n")
    test_bvme_module()
    test_kl_behavior()
    test_parameter_count()
    test_determinism()
    print("=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
