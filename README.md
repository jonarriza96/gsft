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
conda create --name gsft python=3.9
conda activate gsft
pip install -r requirements.txt
```

To ensure that Python paths are properly defined, update the `~/.bashrc` by adding the following lines

```
export GSFT_PATH=/path_to_gsfc
export PYTHONPATH=$PYTHONPATH:/$GSFT_PATH
```

## Usage

To run an example, execute [this file](gsfc/gsfcrm0.py). For example,

```
python gsfcrm0.py --t_nav 6.5 --case h
```

runs the `helix` case study for a navigation time of `6.5 s`. The other two case studies are `b` (backflip, named "loop" in the paper) or `l` (lissajous). You can also add the `--no_visualization` and `--nsf` (non slosh-free) flags when running the command. Other seettings can be modified from the [configuration file](gsfc/config.py)
