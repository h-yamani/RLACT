# RLACT
### Robot Learning: Imitation, Visuomotor Policies, and Reinforcement Learning

**RLACT** is a modular research and engineering project for developing, training, evaluating, and extending learning-based robotic control policies.

Built around **PyTorch** and **Hugging Face LeRobot**, the project explores modern robot learning approaches, including Behaviour Cloning (BC), Action Chunking Transformer (ACT), Diffusion Policy, Vision-Language-Action (VLA) models, and reinforcement learning.

The project is being developed incrementally, from reproducible imitation learning baselines to advanced visuomotor policies and data-efficient robot learning methods.

## Objectives

- Establish reproducible robot learning experiments using LeRobot.
- Understand and implement imitation learning and visuomotor control pipelines.
- Train and evaluate ACT and Diffusion Policy models.
- Explore pretrained VLA models and language-conditioned robot control.
- Investigate reinforcement learning and imitation learning integration.
- Study data efficiency, experience replay, policy generalisation, and robustness.
- Develop reusable tools for dataset inspection, training, evaluation, and visualisation.
- Explore simulation-based evaluation and potential real-world robotic deployment.

## Research Scope

| Area | Methods and technologies |
|---|---|
| Imitation Learning | Behaviour Cloning, supervised policy learning |
| Visuomotor Learning | Image encoders, proprioception, multimodal observations |
| Transformer Policies | ACT, action chunking |
| Generative Policies | Diffusion Policy |
| Vision-Language-Action | Pretrained VLA models, language-conditioned policies |
| Reinforcement Learning | TD3, SAC, off-policy learning |
| Data Efficiency | Experience replay, demonstration reuse |
| Robotics | Robot state, actions, calibration, control, simulation |
| Infrastructure | PyTorch, LeRobot, Hugging Face Hub, Git |

## Planned Architecture

```text
RLACT/
├── README.md
├── requirements.txt
├── configs/
│   ├── datasets/
│   ├── policies/
│   └── experiments/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_behaviour_cloning.ipynb
│   └── 03_act_exploration.ipynb
├── src/
│   └── rlact/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── control/
│       └── utils/
├── scripts/
├── tests/
└── results/
```

This structure is a proposed architecture and will evolve as the project progresses.

## Development Roadmap

### Phase 1: Environment and Data Foundations
- [ ] Configure the Python and PyTorch environment.
- [ ] Install and explore Hugging Face LeRobot.
- [ ] Download and inspect an existing robotics dataset.
- [ ] Understand observations, actions, episodes, and timestamps.
- [ ] Visualise robot demonstrations.

### Phase 2: Behaviour Cloning
- [ ] Implement a basic Behaviour Cloning policy.
- [ ] Build a supervised training pipeline.
- [ ] Evaluate prediction accuracy and closed-loop behaviour.
- [ ] Investigate compounding errors and distribution shift.

### Phase 3: Action Chunking Transformer
- [ ] Study the ACT architecture.
- [ ] Understand action chunking and temporal ensembling.
- [ ] Train an ACT policy using LeRobot.
- [ ] Evaluate predicted action sequences.
- [ ] Compare ACT with a basic Behaviour Cloning baseline.

### Phase 4: Diffusion Policy
- [ ] Study diffusion-based action generation.
- [ ] Understand noise prediction and iterative denoising.
- [ ] Train a Diffusion Policy using LeRobot.
- [ ] Compare ACT and Diffusion Policy performance.

### Phase 5: Vision-Language-Action Models
- [ ] Explore pretrained VLA models.
- [ ] Study vision, language, and action representations.
- [ ] Investigate language-conditioned robotic tasks.
- [ ] Evaluate pretrained policies where feasible.

### Phase 6: Reinforcement Learning
- [ ] Establish an off-policy reinforcement learning baseline.
- [ ] Investigate TD3 and SAC for continuous control.
- [ ] Explore imitation learning and reinforcement learning integration.
- [ ] Investigate replay-based and data-efficient policy improvement.

### Phase 7: Simulation and Robotic Integration
- [ ] Configure a suitable simulation environment.
- [ ] Study robot observations and action spaces.
- [ ] Implement calibration and action-space mappings.
- [ ] Evaluate learned policies in closed-loop simulation.
- [ ] Explore deployment to compatible robotic hardware.

## Evaluation

Depending on the task and available environment, experiments may consider:

- Task success rate
- Training and sample efficiency
- Policy generalisation
- Action prediction error
- Closed-loop control performance
- Robustness to observation noise and distribution shift
- Inference latency and control frequency

Evaluation protocols will be documented alongside individual experiments.

## Technology Stack

**Core:** Python, PyTorch, Hugging Face LeRobot

**Robot Learning:** Behaviour Cloning, ACT, Diffusion Policy, VLA, TD3, SAC

**Data and Experimentation:** Hugging Face Datasets, NumPy, Pandas, Matplotlib

**Robotics and Simulation:** MuJoCo and compatible robotic environments

**Development:** Git, GitHub, virtual environments, Jupyter notebooks

## Reproducibility

As the project develops, experiments will document:

- Dataset and model versions
- Training configurations
- Random seeds
- Evaluation procedures
- Hardware and software dependencies
- Model checkpoints and experimental results

## References

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [MuJoCo](https://mujoco.org/)

## Project Status

**Status:** Early development

RLACT is an evolving research and learning project. The roadmap describes planned capabilities, not completed implementations. Components, experiments, and documentation will be added and validated progressively.

## License

No open-source license has been selected. The repository is currently intended for private development.

---

**RLACT | Robot Learning Research and Engineering**
