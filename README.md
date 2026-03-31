# ExpoComm-B: Bandwidth-Constrained Exponential Topology

This repository contains the implementation of **ExpoComm-B**, a coursework project for COMP 559 (Machine Learning with Graphs) at Rice University. 

This project builds upon the original **ExpoComm** architecture by natively integrating a **Bandwidth-constrained Variational Message Encoding (BVME)** bottleneck. This allows reinforcement learning agents to learn sparse communication graphs while also heavily compressing the messages they send, reducing overall bandwidth requirements.

## 🚀 Getting Started

We recommend using Miniforge to manage your cluster environments (as Anaconda is deprecated on clusters like Rice NOTS).

**1. Create Conda Environment**
```bash
conda create -n expocomm-b python=3.8
conda activate expocomm-b
```

**2. Install Core Dependencies**
```bash
cd src
pip install -r requirements.txt
cd ..
```

**3. Install Custom MAgent/PettingZoo**
Because this repository relies on legacy MARL benchmarks, you must use these specific versions:
```bash
pip install pettingzoo[magent]==1.14.0 supersuit==3.3.0
cp env/battle_v3_view7.py PATH_TO_YOUR_PETTINGZOO_ENV/pettingzoo/magent/
cp env/adversarial_pursuit_view8_v3.py PATH_TO_YOUR_PETTINGZOO_ENV/pettingzoo/magent/
```

---

## 💻 Running Experiments

The master run script uses the PyMARL registry system. You can launch training runs directly on an HPC cluster:

```bash
python src/main.py --config=ExpoComm_B_qmix --env-config=MAgent_Battle run_name="GPU-Run"
```

All experiment results, metrics, and PyTorch `.pt` checkpoints will be stored in the `work_dirs/` directory and automatically synced to Weights & Biases if configured.

---

## 🙏 Acknowledgement & Citations

The baseline codebase, exponential topology implementation, and multi-agent controllers are built directly upon the open-source **ExpoComm** repository, which utilized [EPyMARL](https://github.com/uoe-agents/epymarl) and [MAgent 2](https://github.com/Farama-Foundation/MAgent2). Our BVME module is integrated into their framework.

**If you use the original exponential topology code in your research or find it helpful, please cite the original authors' paper:**

```bibtex
@inproceedings{liexponential,
  title={Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning},
  author={Li, Xinran and Wang, Xiaolu and Bai, Chenjia and Zhang, Jun},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
