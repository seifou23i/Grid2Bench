# RLBenchPlot
This is a repository to evaluate the agent under the grid2op framework

*   [1 Installation](#installation)
    *   [1.1 Setup a Virtualenv (optional)](#setup-a-virtualenv-optional)
    *   [1.2 Install using Poetry](#install-using-poetry)
    *   [1.3 Install from PyPI](#install-from-pypi)
*   [2 Main features of RLBenchPlot](#main-features-of-rlbenchplot)
*   [3 Getting Started](#getting-started)
*   [4 Documentation](#documentation)
*   [5 License information](#license-information)

# Installation
## Requirements:
*   Python >= 3.9

## Setup a Virtualenv (optional)
### Create a virtual environment 
```commandline
mkdir my-project-folder
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv venv_rlbench
```
### Enter virtual environment
```commandline
source venv_grid2op/bin/activate
```

## Install using poetry
```commandline
git clone https://github.com/seifou23i/RLBenchPlot.git
cd RLBenchPlot
poetry install
```

## Install from PyPI
In future 
```commandline
pip3 install rlbenchplot
```

# Main features of RLBenchPlot
```diff  
- Descirbe breifly the main features allowing to analyze and compare the agents...
```

# Usage
An example of how to use the library is provided below:
```python
from rlbenchplot.AgentsAnalytics import AgentsAnalytics as agts
from rlbenchplot.EpisodesPlot import EpisodesPlot

# parent directory for agents log files
input_data_path = os.path.abspath("../data/input")

# Loading agents and required episodes
agents = agts(data_path=input_data_path, agents_names= ["PPO_Agent", "MazeRL_Agent", "Expert_Agent" ]) 

# Visualize the cumulative reward for the first agent in the list
agents.agents_data[0].plot_cumulative_reward()

```

# Getting Started
Some Jupyter notebook are provided as tutorials for the RLBenchPlot package. They are located in the 
[getting_started](getting_started) directories. 

Getting_Started notebook contains simple examples to show how we use the functions defined in the framework:

   * Loading agent results- fot this part there are 2 options:
      * First option: you can load agent's resulsts separately [using EpisodeData class]
      * Second Option: you can load all the agents' results at onece [using AgentsAnalytics class]
   * Action Frequency
   * Impact of actions on objects 
   * Action Execution Time
   * Action Sequence length
   * Agents bahaviour analysis 


   

# Documentation
Under progress...

# License information
Copyright 2022-2023 IRT SystemX & RTE

    IRT SystemX: https://www.irt-systemx.fr/
    RTE: https://www.rte-france.com/

This Source Code is subject to the terms of the Mozilla Public License (MPL) v2 also available 
[here](https://www.mozilla.org/en-US/MPL/2.0/)
