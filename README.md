# Flexible Option Learning

This repository contains code for the paper [Flexible Option Learning](https://arxiv.org/abs/2112.03097) presented as a Spotlight at NeurIPS 2021. The implementation is based on [gym-miniworld](https://github.com/maximecb/gym-miniworld), OpenAI's baselines [baselines](https://github.com/openai/baselines) and the Option-Critic's [tabular implementation](https://github.com/jeanharb/option_critic/tree/master/fourrooms).


Contents:
- [FourRooms Experiments](#four-rooms)
- [Continuous Control Experiments](#control-experiments-tmaze--halfcheetah)
- [Visual Navigation Experiments](#visual-navigation-experiments-miniworld)





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
pip install -e . (in the main directory)
pip install gym==0.9.3
pip install mujoco-py==0.5.1
```
#### Launch

```
cd baselines/ppoc_int
python run_mujoco.py --switch --nointfc --env AntWalls --eta 0.9 --mainlr 8e-5 --intlr 8e-5 --piolr 8e-5
```


## Continuous Control (MuJoCo)

#### Installation

```
virtualenv moc_vision --python=python3
source moc_vision/bin/activate
pip install tensorflow==1.12.0 
cd continuous_control
pip install -e . (in the main directory)
pip install gym==0.9.3
pip install mujoco-py==0.5.1
```
#### Launch

```
cd baselines/ppoc_int
python run_mujoco.py --switch --nointfc --env AntWalls --eta 0.9 --mainlr 8e-5 --intlr 8e-5 --piolr 8e-5
```


