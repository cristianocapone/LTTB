# Learning Through Target Bursts (LTTB)

This is the accompanying source code of the <a href="https://arxiv.org/abs/2201.11717">arXiv 2201.11717</a> preprint paper: Cristiano Capone<sup>\*</sup>, Cosimo Lupo<sup>\*</sup>, Paolo Muratore, Pier Stanislao Paolucci (2022) "*Burst-dependent plasticity and dendritic amplification support target-based learning and hierarchical imitation learning*".

Please give credit to this paper if you use or modify the code in a derivative work. This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

The code is written in Python and organized as follows. The network model itself is defined into the `lttb.py` module (which defines the `LTTB` model object). The model parameters used for the different experiments are organized in different sections of a unique `json` configuration file.

To easily replicate the experiments presented in the paper we provide Jupyter Notebooks that can be run on their own.

## External Dependences
To run the different experiments the following external libraries should be installed on the machine:

```python
jupyter notebook    # for a easier user interface
numpy               # Basic numerical array library
matplotlib          # Used for producing the various visualizations
json                # Use to parse the model configuration file
tqdm
```

## Bursts & Context Switch

To reproduce the results presented in `Figure 1` and `Figure 2` of the paper, simply run the associated notebooks: `Figure_1.ipynb` and `Figure_2.ipynb`.

## Button & Food Task

To reproduce the results presented in `Figure 3` of the paper, run the associated notebook `Figure_3.ipynb`.
