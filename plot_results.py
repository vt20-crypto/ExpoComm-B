"""
Generate all figures for ExpoComm-B final report.
All data values come directly from log files. No assumptions.
"""
import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Helper ──────────────────────────────────────────────────────────────────
def parse_section(filepath, start_marker, end_marker=None):
    with open(filepath) as f:
        content = f.read()
    s = content.find(start_marker)
    if s == -1:
        return []
    e = content.find(end_marker, s) if end_marker else len(content)
    section = content[s:e] if e != -1 else content[s:]
    pairs = []
    t_envs = re.findall(r't_env:\s+(\d+)\s*/\s*\d+\n', section)
    returns = re.findall(r'test_return_mean:\s+([-\d.]+)', section)
    step = 25000  # log_interval
    for i, r in enumerate(returns):
        pairs.append(((i + 1) * step, float(r)))
    return pairs

# ── FIGURE 1: MPE Learning Curves (4 methods) ───────────────────────────────
# Data from mpe_baselines.log and ablation_nots.log — confirmed
qmix_vals   = [-81.8745,-57.8042,-49.4835,-45.0665,-42.5472,-32.1761,-31.8141,-30.9996,-30.8510,-30.3271,-30.1961,-29.7657,-29.5751,-29.1989,-28.7812,-28.0194,-27.5157,-27.2560,-26.8052,-26.5966]
excom_vals  = [-75.5276,-54.5863,-46.8091,-42.9017,-40.4366,-31.4187,-30.8786,-31.1741,-31.2644,-31.7971,-32.1989,-31.8864,-31.2882,-31.0993,-29.9079,-28.8666,-28.6163,-28.1698,-27.3993,-27.1132]
bvme_vals   = [-69.2301,-55.8654,-47.8480,-43.5894,-41.0573,-33.6607,-31.7457,-31.5347,-31.3708,-31.2868,-30.9664,-30.6066,-30.6556,-30.4405,-30.3910,-30.2545,-29.9501,-29.1990,-29.2314,-28.8962]
expob_vals  = [-37.0687,-27.0310,-21.3326,-18.2579,-16.5327,-10.9006,-9.2667,-8.9693,-8.8283,-8.5727,-8.5985,-8.4963,-8.4902,-8.4929,-8.4957,-8.3463,-8.3139,-8.2628,-8.2411,-8.2139]
steps = [i * 25000 for i in range(1, 21)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, qmix_vals,  label='QMIX (full-comm)',         alpha=0.85, linewidth=1.5)
ax.plot(steps, excom_vals, label='ExpoComm (sparse, no compress)', alpha=0.85, linewidth=1.5)
ax.plot(steps, bvme_vals,  label='BVME-only (full-graph + compress)', alpha=0.85, linewidth=1.5)
ax.plot(steps, expob_vals, label='ExpoComm-B (ours)', color='red', linewidth=2.2)
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Test Return Mean (↑ better)')
ax.set_title('MPE Simple Spread — Learning Curves (500k steps)')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(0, 500000)
plt.tight_layout()
plt.savefig('mpe_learning_curves.png', dpi=200)
plt.close()
print("Saved mpe_learning_curves.png")

# ── FIGURE 2: MAgent Battle Bar Chart ───────────────────────────────────────
# Source: EXPERIMENT_RESULTS.md (Ansh's runs — no raw log on this machine)
methods  = ['QMIX\n(no comm)', 'ExpoComm\n(sparse)', 'ExpoComm-B\nσ₀=0.005', 'ExpoComm-B\nσ₀=0.01', 'ExpoComm-B\nσ₀=0.02', 'ExpoComm-B\nσ₀=0.05']
survivors= [72.4, 60.6, 70.4, 50.6, 64.5, 3.6]
colors   = ['#4C72B0','#4C72B0','#DD8452','#DD8452','#DD8452','#DD8452']

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(methods, survivors, color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(81, color='gray', linestyle='--', linewidth=1, label='Total enemies (81)')
ax.set_ylabel('Avg. Enemy Survivors (↓ better)')
ax.set_title('MAgent Battle — Enemy Survivors by Method')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.2, f'{b.get_height():.1f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, 90)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('magent_survivors.png', dpi=200)
plt.close()
print("Saved magent_survivors.png")

# ── FIGURE 3: KL Weight (λ) Ablation ────────────────────────────────────────
# From ablation_output.log (λ=0.01) and ablation_nots.log (rest)
lambdas = [0.01, 0.1, 1.0, 5.0, 10.0]
kl_ret  = [-8.5074, -8.1750, -8.2139, -8.4163, -8.3556]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lambdas, kl_ret, marker='o', linewidth=2, color='steelblue')
ax.set_xscale('log')
ax.set_xlabel('KL Weight λ (log scale)')
ax.set_ylabel('Final Test Return Mean (↑ better)')
ax.set_title('MPE Simple Spread — KL Weight Ablation')
for x, y in zip(lambdas, kl_ret):
    ax.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(-8.8, -7.9)
plt.tight_layout()
plt.savefig('mpe_kl_ablation.png', dpi=200)
plt.close()
print("Saved mpe_kl_ablation.png")

# ── FIGURE 4: Compression Ratio Ablation ────────────────────────────────────
# From ablation_nots.log
dims   = [64, 32, 16, 8]
ratios = [1.0, 0.5, 0.25, 0.125]
cr_ret = [-8.2139, -8.2620, -8.2652, -8.2094]
labels = ['64\n(1.0×)', '32\n(0.5×)', '16\n(0.25×)', '8\n(0.125×)']

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(4), cr_ret, marker='s', linewidth=2, color='seagreen')
ax.set_xticks(range(4))
ax.set_xticklabels(labels)
ax.set_xlabel('Compressed Dimension (Ratio)')
ax.set_ylabel('Final Test Return Mean (↑ better)')
ax.set_title('MPE Simple Spread — Compression Ratio Ablation')
for i, y in enumerate(cr_ret):
    ax.annotate(f'{y:.4f}', (i, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(-8.5, -8.0)
plt.tight_layout()
plt.savefig('mpe_cr_ablation.png', dpi=200)
plt.close()
print("Saved mpe_cr_ablation.png")
