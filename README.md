# Physical Learning in Resistor Networks

Code accompanying the manuscript **“Sequential Learning and Catastrophic Forgetting in Differentiable Resistor Networks.”**

This repository studies sequential learning and catastrophic forgetting in differentiable resistor networks. The model treats an electrical network as a weighted graph with trainable edge conductances. For fixed boundary voltages, the network state is obtained by solving a linear system based on the weighted graph Laplacian. Learning adjusts conductances so that the output node matches a desired target.

## Repository structure

- `src/` – core model and training code  
- `experiments/` – scripts for running experiments  
- `results/` – saved numerical outputs  
- `plots/` – generated figures

## Setup

Open the project in VS Code, then create and activate a Python environment in the terminal.

### Using pip

python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

### Using pip

conda env create -f environment.yml
conda activate resistor-networks

## Reproducing results

Run the experiment scripts in experiments/ to generate numerical results, then run the plotting scripts to recreate the figures in plots/. Outputs should be saved to results/ and plots/.

### Main experiments

The repository includes experiments on:

baseline sequential training

anchor regularization

regularization-strength sweep

gradient conflict and forgetting

localization of conductance updates

current-update correlations

task recovery under repeated training

EWC-style anchoring baseline

network-size effects

## Reproducibility

For reproducibility, fixed random seeds should be used where applicable, and the archived release corresponding to the manuscript submission should be cited.

## License

This project is released under the MIT License. See LICENSE for details.

### MIT License

Copyright (c) 2026 Maniru Ibrahim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.