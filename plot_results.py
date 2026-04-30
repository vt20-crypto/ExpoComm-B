import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_for_returns(logfile, start_marker, end_marker=None):
    steps = []
    returns = []
    with open(logfile, 'r') as f:
        lines = f.readlines()
        
    in_section = False
    for i, line in enumerate(lines):
        if start_marker in line:
            in_section = True
            
        if in_section and end_marker and end_marker in line:
            break
            
        if in_section:
            if "t_env:" in line and "Episode:" in line:
                match = re.search(r't_env:\s+(\d+)', line)
                if match:
                    steps.append(int(match.group(1)))
            
            if "test_return_mean:" in line:
                match = re.search(r'test_return_mean:\s+([-.\d]+)', line)
                if match:
                    returns.append(float(match.group(1)))
                    
    # Ensure they match in length, sometimes logs truncate
    min_len = min(len(steps), len(returns))
    return np.array(steps[:min_len]), np.array(returns[:min_len])

# 1. MPE 4-Method Comparison (Learning Curves)
print("Parsing logs for MPE baselines...")
steps_qmix, ret_qmix = parse_log_for_returns("mpe_baselines.log", ">>> [1/3] QMIX", ">>> [2/3] ExpoComm")
steps_expocomm, ret_expocomm = parse_log_for_returns("mpe_baselines.log", ">>> [2/3] ExpoComm", ">>> [3/3] BVME-only")
steps_bvme, ret_bvme = parse_log_for_returns("mpe_baselines.log", ">>> [3/3] BVME-only", ">>> ALL DONE")
steps_expocomm_b, ret_expocomm_b = parse_log_for_returns("ablation_nots.log", ">>> Running: λ = 1.0 (baseline)", ">>> Finished: λ = 1.0")

plt.figure(figsize=(8, 5))
plt.plot(steps_qmix, ret_qmix, label='QMIX (Full Comm)', alpha=0.8)
plt.plot(steps_expocomm, ret_expocomm, label='ExpoComm (Sparse)', alpha=0.8)
plt.plot(steps_bvme, ret_bvme, label='BVME-only (Compression)', alpha=0.8)
plt.plot(steps_expocomm_b, ret_expocomm_b, label='ExpoComm-B (Ours)', linewidth=2.5, color='red')
plt.xlabel("Environment Steps")
plt.ylabel("Test Return Mean")
plt.title("MPE Simple Spread: Training Performance")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("mpe_learning_curves.png", dpi=300)

# 2. MAgent Battle Bar Chart (Enemy Survivors)
# Values from EXPERIMENT_RESULTS.md
methods = ["QMIX\n(No Comm)", "ExpoComm\n(Sparse)", "ExpoComm-B\nσ₀=0.01", "ExpoComm-B\nσ₀=0.005"]
survivors = [72.4, 60.6, 50.6, 70.4]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, survivors, color=['#1f77b4', '#ff7f0e', '#d62728', '#9467bd'])
plt.axhline(y=81, color='r', linestyle='--', label='Initial Enemy Count (81)')
plt.ylabel("Average Enemy Survivors (lower is better)")
plt.title("MAgent Battle: Coordination Effectiveness")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom')
plt.legend()
plt.tight_layout()
plt.savefig("magent_survivors.png", dpi=300)

# 3. MPE KL Weight Sensitivity
lambdas = [0.01, 0.1, 1.0, 5.0, 10.0]
returns_kl = [-8.507, -8.163, -8.160, -8.555, -8.138]

plt.figure(figsize=(7, 4))
plt.plot(lambdas, returns_kl, marker='o', linestyle='-', color='purple', linewidth=2)
plt.xscale('log')
plt.xlabel("KL Regularization Weight (λ)")
plt.ylabel("Final Test Return Mean")
plt.title("MPE Simple Spread: KL Weight Ablation")
plt.grid(True, linestyle='--', alpha=0.6)
for i, txt in enumerate(returns_kl):
    plt.annotate(f'{txt:.3f}', (lambdas[i], returns_kl[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig("mpe_kl_ablation.png", dpi=300)

# 4. MPE Compression Ratio Ablation
ratios = ["64 (1.0)", "32 (0.5)", "16 (0.25)", "8 (0.125)"]
returns_cr = [-8.160, -8.205, -8.163, -8.219]

plt.figure(figsize=(7, 4))
plt.plot(ratios, returns_cr, marker='s', linestyle='-', color='green', linewidth=2)
plt.xlabel("Compressed Dimension (Ratio)")
plt.ylabel("Final Test Return Mean")
plt.title("MPE Simple Spread: Compression Robustness")
plt.grid(True, linestyle='--', alpha=0.6)
for i, txt in enumerate(returns_cr):
    plt.annotate(f'{txt:.3f}', (ratios[i], returns_cr[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig("mpe_cr_ablation.png", dpi=300)

print("Saved all 4 figures successfully.")
