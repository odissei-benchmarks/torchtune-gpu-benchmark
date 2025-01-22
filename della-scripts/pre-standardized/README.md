# Instructions

1. Create the conda environment using (or referencing) della.yml
2. Activate the conda environment
3. (Optional) If the listed MODEL_DIR is not usable, change it as appropriate and make sure you download the model with:
  - $ tune download meta-llama/<model_name> --output-dir <output_dir> --hf-token <hf-token>
4. (Optional) If you're getting an error about alpaca-cleaned not existing, open a python shell on a login node and run:
  - > from datasets import load_dataset
  - > load_dataset("yahma/alpaca-cleaned")
5. Choose one of the slurm scripts and change the following params before running:
    - job-name (noting changes to batch_size, etc.)
    - gres=gpu:# (if changing the number of GPUs)
    - BATCH_SIZE
    - OUTPUT_DIR
    - nproc_per_node (should match GPU # if more than 1)

(I do plan on abstracting the above a bit more to make it easier to change params in the future)