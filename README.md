# Flexible Option Learning

This repository contains code for the paper [Flexible Option Learning](https://arxiv.org/abs/2112.03097) presented as a Spotlight at NeurIPS 2021. The implementation is based on [gym-miniworld](https://github.com/maximecb/gym-miniworld), OpenAI's baselines [baselines](https://github.com/openai/baselines) and the Option-Critic's [tabular implementation](https://github.com/jeanharb/option_critic/tree/master/fourrooms).


Contents:
- [FourRooms Experiments](#four-rooms)
- [Continuous Control Experiments](#control-experiments-tmaze--halfcheetah)
- [Visual Navigation Experiments](#visual-navigation-experiments-miniworld)





## Tabular Experiments (Four-Rooms)

#### Dependencies and Launch code

```
pip install gym==0.12.1
cd diagnostic_experiments/
python main_fixpol.py --multi_option # for experiments with fixed options
python main.py --multi_option # for experiments with learned options
```
