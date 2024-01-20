# ODICE: Revealing the Mystery of Distribution Correction Estimation via Orthogonal-gradient Update

This is the official implementation for the paper **ODICE: Revealing the Mystery of Distribution Correction Estimation via Orthogonal-gradient Update** accepted as **Notable-top-5%** at ICLR'2024.

### Usage
To reproduce the experiments in Offline RL part, please run:
> python main_odice_rl.py --env_name your_env_name --Lambda your_lambda --eta your_eta --type orthogonal_true_g

To reproduce the experiments in Offline IL part, please run:
> python main_odice_il.py --env_name your_env_name --Lambda your_lambda --eta your_eta --type orthogonal_true_g

Note that although we set "--type" as "orthogonal_true_g" for ODICE, you can check the results of other gradient types("true_g" and "semi_g") if you like. The choice of other hyper-parameters are listed in appendix D.

### Bibtex
```
@inproceedings{mao2024ODICE,
  title  = {ODICE: Revealing the Mystery of Distribution Correction Estimation via Orthogonal-gradient Update},
  author = {Liyuan Mao, Haoran Xu, Weinan Zhang, Xianyuan Zhan},
  year   = {2024},
  booktitle = {International Conference on Learning Representations},
}
```