# Third-party projects

TILT relies on and/or adapts components from the following open-source
projects. Follow each upstream project's license and citation instructions.

| Project | Use in TILT | Repository |
|---|---|---|
| Genesis | Physics simulation, Crazyflie model, and the basis of the hovering environment | [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) |
| Genesis hovering example | Upstream reference for the environment structure | [examples/drone/hover_train.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_train.py) |
| rsl_rl | PPO runner, `MLPModel`, and activation utilities | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) |
| PyTorch | Tensor operations, neural networks, CUDA execution, and Sobol sequences | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| NumPy | Numerical processing and evaluation statistics | [numpy/numpy](https://github.com/numpy/numpy) |
| TensorBoard | Training-log visualization | [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) |

The links above point to the original repositories rather than forks. They are
provided for attribution and to help users locate upstream documentation and
license terms.
