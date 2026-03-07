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


### Performance results

| Batch Size 6 Comparison On: | Della | Snellius | OSSC  |
|:---------------------------:|:-----:|:--------:|:-----:|
|            1 A100           | 19422 |   17500  | 17700 | 
|           2 A100s           | 18247 |   16500  | 16500 |
|           4 A100s           | 18019 |   16400  | 16400 |
|            1 H100           | 36668 |   31100  | 31000 |
|           2 H100s           | 34228 |   28800  | 28600 |
|           4 H100s           | 33650 |   28600  | 28500 |

The difference between Snellius and Della is down to memory clock speeds:
|      | Della    | Snellius |
|------|----------|----------|
| A100 | 1600 MHz | 1215 MHz |
| H100 | 2600 MHz | 1590 MHz |

### Running on your system

1. Create an account on [Weights & Biases](https://wandb.ai/).
1. Download the foundation model and place it in the `models` directory.
This project uses the [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) model.
1. Install the required Python dependencies listed in `requirements/baseline.txt`.
1. Adapt the SLURM job scripts to match your system configuration.

1. First run the models in an interactive mode so that it prompts for Weights & Biases login details.
1. Run SLURM jobs.