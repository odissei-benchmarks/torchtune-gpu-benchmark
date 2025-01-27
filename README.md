# Benchmarking torchtune on different systems

Code for performance benchmarking of Llama model finetuning

Systems tested:
- Della (Princeton HPC)
- Snellius (Dutch HPC)
- OSSC (Dutch secure HPC)


### Contents of directories
- `cbs/` contains a README I sent along the model upload
- `della-scripts`, `snellius-scripts` and `ossc-scripts` contain scripts for running code on the respective systems. They also have READMEs describing details for running the scripts
- `plots/` contains some code from Matt for comparing the runs across systems
- `requirements/` contains files related to managing dependencies
-  `configs/` contains configuration files for torchtune.
- `datasets/` contains training datasets.
    - `alpaca_data_cleaned.json` contains text that is fed to the model for updating the parameters.
    - The dataset is licensed under `datasets/LICENSE`, while the remaining code in this repository falls under `./LICENSE`.
