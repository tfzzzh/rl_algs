# Reinforcement Learning Algorithms
This repository contains a list of reinforcement learning algorithms.

## Experimental Results
Evaluation rewards of DDPG and PPO on  halfcheetah task:
![Evaluation Reward](figures/results.png)

## Install The Package
Follow these steps to set up the environment
1. Create a new virtual environment:
```shell
conda create -n rl_env python=3.12
```
2. Intall PyTorch:
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. Install other dependence
```shell
pip3 install -r ./requirements.txt
```

4. Install the package in development mode
```shell
pip3 install -e .
```
# Run an Example
To run an example, use the following command:
```shell
python ./scripts/run_ddpg.py -cfg ./experiments/ddpg/halfcheetah_reparametrize.yaml
```
I use tensorboard to log trainning results. To view them, you can use the following command:
```shell
tensorboard --logdir ./data
```

## Acknowledgements

This project was inspired by and adapted from the following repositories:
- [Berkeley CS285](https://github.com/berkeleydeeprlcourse/homework_fall2023)