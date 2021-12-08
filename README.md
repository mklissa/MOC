# Flexible Option Learning

This repository contains code for the paper [Flexible Option Learning](https://arxiv.org/abs/2112.03097) presented as a Spotlight at NeurIPS 2021. The implementation is based on [gym-miniworld](https://github.com/maximecb/gym-miniworld), OpenAI's  [baselines](https://github.com/openai/baselines) and the Option-Critic's [tabular implementation](https://github.com/jeanharb/option_critic/tree/master/fourrooms).


Contents:
- [FourRooms Experiments](#tabular-experiments-four-rooms)
- [Continuous Control Experiments](#continuous-control-mujoco)
- [Visual Navigation Experiments](#maze-navigation-miniworld)
- [Citation](#cite)





## Tabular Experiments (Four-Rooms)

#### Installation and Launch code

```
pip install gym==0.12.1
cd diagnostic_experiments/
python main_fixpol.py --multi_option # for experiments with fixed options
python main.py --multi_option # for experiments with learned options
```


## Continuous Control (MuJoCo)

#### Installation

```
virtualenv moc_cc --python=python3
source moc_cc/bin/activate
pip install tensorflow==1.12.0 
cd continuous_control
pip install -e . 
pip install gym==0.9.3
pip install mujoco-py==0.5.1
```
#### Launch

```
cd baselines/ppoc_int
python run_mujoco.py --switch --nointfc --env AntWalls --eta 0.9 --mainlr 8e-5 --intlr 8e-5 --piolr 8e-5
```


## Maze Navigation (MiniWorld)

#### Installation

```
virtualenv moc_vision --python=python3
source moc_vision/bin/activate
pip install tensorflow==1.13.1
cd vision_miniworld
pip install -e .
pip install gym==0.15.4
```

#### Launch

```
cd baselines/
# Run agent in first task
python run.py --alg=ppo2_options --env=MiniWorld-WallGap-v0 --num_timesteps 2500000 --save_interval 1000  --num_env 8 --noptions 4 --eta 0.7

# Load and run agent in transfer task
python run.py --alg=ppo2_options --env=MiniWorld-WallGapTransfer-v0 --load_path path/to/model --num_timesteps 2500000 --save_interval 1000  --num_env 8 --noptions 4 --eta 0.7
```


## Cite

If you find this work useful to you, please consider adding you to your references. 


```
@inproceedings{
klissarov2021flexible,
title={Flexible Option Learning},
author={Martin Klissarov and Doina Precup},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=L5vbEVIePyb}
}
```
