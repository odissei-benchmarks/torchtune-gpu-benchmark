# torchtune-gpu-benchmark
Code for running gpu benchmarks with torchtune on della

### Setup

1. Create a conda environment (you can change the name from ttenv, an abbreviation for torchtune test conda):

```
module load anaconda3/2024.6
conda create -n ttenv pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate ttenv
pip install torchao
pip install torchtune
pip install wandb
```

2. Download Llama to the desired location:

```
tune download meta-llama/Llama-3.2-1B-Instruct --output-dir <output_dir> --hf-token <hf-token>
```

By default, setup-della.sh is suggesting "/scratch/gpfs/${USER}/torchtune_models/Llama-3.2-1B-Instruct" so I'd stick with that.

3. Cache alpaca-cleaned so that we have it

Since compute nodes have no internet connection and you may not have alpaca-cleaned downloaded, run the following in a one-off python shell with your conda environment activated:

```
from datasets import load_dataset
ds = load_dataset("yahma/alpaca-cleaned")
```

4. Edit setup-della.sh if necessary

Params to tweak such as BATCH_SIZE are near the top but other changes can be done as needed if MODEL_DIR and/or OUTPUT_DIR need to differ.

5. Edit slurm scripts if necessary

I don't have an elegant way to change the job name dynamically so when I change batch size I also manually change the job name number after "b" (for batch_size); otherwise, the only change you'd have to make would be if your conda environment has a different name (TODO: abstract the env name into setup-della.sh)

6. Submit jobs

Since batch_size is read when the job initialized, it's best to set a batch size and then do a bunch of jobs for that one value (e.g. different # of GPUs, different models of GPUs) and don't submit more until the last job is running.

### Running jobs

(Copied from snellius-scripts...)

Torchtune configuration can be managed through yaml files or through command line arguments. Command line arguments override the configuration file.

The configuration files in `config/` contain those parameters that are not changing (much) across machines and runs. Command line arguments are used for paths, which are machine-specific, and `max_steps_per_epoch`, which is mostly for fast devruns. 

(For della, I've been running these one at a time just to be safe unlike the script for snellius to submit a bunch at once.)