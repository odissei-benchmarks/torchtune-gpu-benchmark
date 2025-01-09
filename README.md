# torchtune-gpu-benchmark
Code for running gpu benchmarks with torchtune

Based on this tutorial: https://pytorch.org/torchtune/main/tutorials/first_finetune_tutorial.html. 

### Setup: Snellius

```
# create model directory
model_dir="/projects/0/prjs1019/torchtune/models/Llama-2-7b-hf/"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source .venv/bin/activate

pip install -r requirements/snellius.txt

tune download meta-llama/Llama-2-7b-hf \
    --output-dir "$model_dir" \
    --hf-token <ACCESS TOKEN>

``` 


### Creating config files

1. Run `tune cp llama2/7B_lora_single_device custom_config.yaml`
2. Update `custom_config.yaml`, in particular the paths to the checkpoints and the outputs. They may also be provided from the command line but I have not tried this.
3. Before running a job, run `tune validate config_file.yaml` to check if the config is well-formed


### Performance tracking

`s/it` is taken from the log file.

| cluster  | gpus 	| cpus | dataset                                   | date     | s/it	  | 
|----------|-----------	|------|-------------------------------------------|----------|-----------|
| Snellius | 1xA100     | 18   | torchtune.datasets.alpaca_cleaned_dataset | 09/01/24 | 2.4s/it   |
|          |     	|      |                                           |          |     	  |
|          |      	|      |                                           |          |      	  |



