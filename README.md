# GSFT: Geometric Slosh-Free Tracking for Robotic Manipulators

This repository contains the code for the submission "Geometric Slosh-Free Tracking for Robotic Manipulators" for ICRA24.

## Installation

### Dependencies

Initialize git submodules with

```
    git submodule init
    git submodule update
```

### Python environment

Install the specific versions of every package from `requirements.txt` in a new conda environment:

```
conda create --name gsfc python=3.9
conda activate gsfc
pip install -r requirements.txt
```

To ensure that Python paths are properly defined, update the `~/.bashrc` by adding the following lines

```
export GSFT_PATH=/path_to_gsfc
export PYTHONPATH=$PYTHONPATH:/$GSFT_PATH
```

## Usage

### Overall

To run an example, execute [this file](gsfc/gsfcrm0.py). For example,

```
python -i gsfcrm0.py --t_nav 6.5 --case h
```

runs the `helix` case study for a navigation time of `6.5 s`. The other two case studies are `b` (backflip, named "loop" in the paper) or `l` (lissajous). You can also add the `--no_visualization` and `--nsf` (non slosh-free) flags when running the command. Other seettings can be modified from the [configuration file](gsfc/config.py)

### Simulation results

To automate the result generation you can run [this file](gsfc/results/generate_results.py) (generates data) and [this other file](gsfc/results/visualize_results.py) (plots the generated data).

### Experiments

To conduct the experiments we feedforward the joint space values obtained in the simulations. For this purpose, we resample the joint space values at 1KHz and save them into pickle files stored in [this folder](gsfc/xperiments/data). To do this, you need to activate [this if condition](gsfc/gsfcrm0.py#L1060).

## Paper and video content

If you want to visualize the figures and scenes for the paper/video or save data, you might have to activate the functions after [this line](gsfc/gsfcrm0.py#L1046).
