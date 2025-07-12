# ODIL ( One-Shot Dual Arm imitation learning ) 
**Note**: This repo contains files from the repo "A Comparison of Imitation Learning Algorithms for Bimanual Manipulation".

This codebase contains the implementation of the algorithms and environments evaluated in [*A Comparison of Imitation Learning Algorithms for Bimanual Manipulation*](https://bimanual-imitation.github.io/).


### Installation

#### Step 1:
*Install the conda environment*. Depending on your platform, install the correct `<arch> = x86` or `<arch> = arm` environments, located in the [requirements](./requirements) folder.  The installation command for the corresponding algorithms are given below:
- `conda env create -f torch_<arch>.yml` for: [**ACT**, **Diffusion**] (and experimental: **BC**)
- `conda env create -f tensorflow_<arch>.yml` for: [**IBC**]
- `conda env create -f theano_x86.yml` for: [**GAIL**, **DAgger**, **BC**] (x86 only)


#### Step 2:
*Activate the conda environment*. The conda environments for the corresponding algorithms are given below:
- `conda activate irl_torch` for: [**ACT**, **Diffusion**] (and experimental: **BC**)
- `conda activate irl_tensorflow` for: [**IBC**]
- `conda activate irl_theano` for: [**GAIL**, **DAgger**, **BC**]

#### Step 3:
*Install bimanual_imitation*. Change to main repo directory and run: `pip install -e .` to install the bimanual_imitation library.


#### Step 4:
To run the ODIL algorithm

```
cd irl_control
python3 final/main.py
```
For reference of all the work that has been done: PHASE 1 and 2

https://drive.google.com/file/d/1jE4V6wLOJiNybncz7Nnf4A5Z54wtQJki/view?usp=sharing
---
