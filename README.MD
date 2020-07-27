# Hiring Environment for OpenAI Gym

## Releases

Version | Date | Description
--- | --- | ---

## File Descriptions

```
experiments/                            Implements RL experiments on variations of Hiring Env.
    optimal-fairness-policy.ipynb       Executes optimal policy for gender equal outcome constraint.
    utils.py                            Defines RL utilty methods.

gym_hiring/                             Defines gym_hiring package.
    envs/
        __init__.py                     Imports HiringEnv.
        hiring_env.py                   Defines HiringEnv.
    __init__.py                         Registers hiring-env with gym.

conda.yaml                              Defines development environment requirements.
setup.py                                Describes gym_hiring package meta information.
```

## Development Setup

The following steps show how to setup the conda environment so that all dependencies are correctly installed, as well as how to run the Jupyter Notebooks in the context of the conda environment.

```sh
# Create the conda environment from the config file
conda env create -f=conda.yaml

# Activate the conda environment
conda activate gym-hiring

# Create an IPython kernel which will allow you to run the Jupyter Notebook in the conda environment
python3.6 -m ipykernel install --user --name gym-hiring

# Start the jupyter notebook
jupyter notebook
```

Then when you're in the Jupyter Notebook, select `Kernel > Change Kernel > gym-hiring`.
