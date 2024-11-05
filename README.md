# The Differentiable Feasibility Pump
![Tests badge](https://github.com/MCacciola94/DiffPump/actions/workflows/main.yml/badge.svg?branch=main)

This repository provides method to run the differentiable feasibility pump algorithm. It is based on the paper *The Differentiable Feasibility Pump* by Matteo Cacciola, Alexandre Forel, Andrea Lodi, and Antonio Frangioni. The paper is [available here](https://arxiv.org/).

## Getting Started

### Installation
First, we recommend creating a virtual Python environment:
```shell
    python -m venv env_fp/
    source env_fp/bin/activate

```

Then, install the dependencies using:
```shell
    python -m pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install .
```

The installation can be checked by running:
```shell
    python -m pytest
```

### Dependencies

This project requires the gurobi solver. Free academic licenses can be obtained at:
 - https://www.gurobi.com/academia/academic-program-and-licenses/
 - https://www.gurobi.com/downloads/end-user-license-agreement-academic/


### Running experiments
The experiments in the paper are based on instances form the [MIPLIB library](https://miplib.zib.de/). The instances should be in `.mps` format. By default, the instances are read in the folder `data\MIPLIB`. A few small instances are included in this repository for test purposes.

Experiments can be run using the script `script\main.py` and specifying the instance name:
```shell
    python script\main.py instance_name
```
The differentiable feasibility pump can be configured through optional command line arguments. For instance, the variant called DP4 in the paper can be run using:
```shell
    python script\main.py instance_name --lr 0.6 --integ_loss 10 --feas_loss 0.001 --reg_loss 0.1 --p 2
```
By default, the feasibility pump of Fischetti et al. (2005) is run.