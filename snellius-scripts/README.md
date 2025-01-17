# torchtune-gpu-benchmark
Code for running gpu benchmarks with torchtune

Based on this tutorial: https://pytorch.org/torchtune/main/tutorials/first_finetune_tutorial.html. 

### Setup

```
# create model directory
model_dir="/projects/0/prjs1019/torchtune/models/Llama-2-7b-hf/"

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate

pip install -r requirements/snellius.txt

tune download meta-llama/Llama-2-7b-hf \
    --output-dir "$model_dir" \
    --hf-token <ACCESS TOKEN>

``` 


### Running jobs

Torchtune configuration can be managed through yaml files or through command line arguments. Command line arguments override the configuration file.

The configuration files in `config/` contain those parameters that are not changing (much) across machines and runs. Command line arguments are used for paths, which are machine-specific, and `max_steps_per_epoch`, which is mostly for fast devruns. 

Jobs for benchmarking can be run with 

```
cd torchtune-gpu-benchmarks
source snellius-scripts/run_jobs.sh
```
The script submits all jobs related to the benchmarking.

There is one slurm script for each permutation of (gpu type, N gpus). The have common parameters for the model and output location, which are stored in `snellius-scripts/setup.sh`. 

Dependencies for snellius are in `requirements/load_venv.sh`.

I tried to keep names for `wandb` specific to the experiment runs, but the `name` and `job_type` parameter did not work. For now, I'm using the `id` parameter which creates directories such as `wandb/offline-run-20250115_122206-{id}`.
 
### Performance tracking

- dataset: torchtune.datasets.alpaca_cleaned_dataset
- batch size > 6 leads to OOM. how do we know bs=6 will not OOM after the max steps we're doing?

| cluster  | gpus 	| cpus | date     | tokens/s  | batch size |
|----------|-----------	|------|----------|-----------|------------
| Snellius | 1xA100     | 18   | 17/01/25 | 17500     | 6 
| Snellius | 2xA100     | 18   | 17/01/25 | 16500     | 6 
| Snellius | 4xA100     | 18   | 17/01/25 | 16400     | 6
| Snellius | 1xH100     | 18   | 17/01/25 | 31100     | 6  
| Snellius | 2xH100     | 18   | 17/01/25 | 28800     | 6
| Snellius | 4xH100     | 18   | 17/01/25 | 28600     | 6 
|          |     	|      |          |           |
|          |      	|      |          |           |



