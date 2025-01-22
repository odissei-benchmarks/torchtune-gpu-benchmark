# Model upload: Llama 3.2 1B

These files include weights of a large language model (Llama 3.2 1B Instruct, [openly](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) available). 
This is not "data" in the conventional sense, but rather parameter settings from a statistical model that predicts the order of words in a text. The model has been pretrained on public text. The key component of the model are its weights, which represent everything the model has learned during its training. The weights are grouped into 'tensors'. A tensor is essentially a group of numbers (weights); each tensor has a name so that the numbers stored in the tensor can be accessed via this name. The "safetensors" format is a way to package these numbers (weights) that's faster and more secure than some older formats. It allows the model to be loaded and used more efficiently.

Because the model is trained on public data, there is no information in the data that is not already public.

The purpose of this current upload is to run the model on the Odissei Secure Supercomputer and compare its performance to other HPC systems. At a later stage, we may train the model on CBS microdata.

The zip file contains the following
- `model.safetensors`: This file stores the model weights. Below is a description on how the file can be opened.
- `original/tokenizer.model`: This file stores the model of the tokenizer. A tokenizer is a program that splits text into components -- words or subwords.
- Various `.json`, `.md`, `.yaml` files that can be opened in a text editor. The files contain configurations and READMEs from the Llama model.


### Security
The model was downloaded from this repository: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main.
The repository host, Huggingface, checks all files for their safety, as evidenced with a clickable `Safe` symbol for each file. 


### Inspecting the `.safetensors` file

The python code below allows inspecting the contents of the files. The code does the following. For each specified tensor, it prints:
- The tensor name
- The shape of the tensor
- The data type of the tensor
- The first 10 values of the tensor
- Some basic statistics based on all the numbers (weights) stored in the tensor - mean, standard deviation, min, and max

```bash
pip install safetensors
pip install torch
```


```python
from safetensors import safe_open
import torch


# put the path to one of the .safetensors file here, e.t. 'project9469_model-00001-of-00004.safetensors'
file_path = "/path/to/file/model.safetensors" 


with safe_open(file_path, framework="pt", device="cpu") as f:
    # Get the list of all tensor names
    tensor_names = f.keys()
    
    print("Available tensors:")
    for name in tensor_names:
        print(name)
    
    # Select specific tensors to inspect (select from printed tensor names) e.g. these three
    tensors_to_inspect = ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.mlp.down_proj.weight", 
                          "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.post_attention_layernorm.weight",
                         "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.self_attn.q_proj.weight",
                         ] 
    
    for tensor_name in tensors_to_inspect:
        if tensor_name in tensor_names:
            # Load the tensor
            tensor = f.get_tensor(tensor_name)
            
            # Print tensor information
            print(f"\nTensor: {tensor_name}")
            print(f"Shape: {tensor.shape}")
            print(f"Data type: {tensor.dtype}")
            
            # Print the first few values
            print("First few values:")
            print(tensor.flatten()[:10])  # Print first 10 values
            
            # Print some statistics
            print(f"Mean: {torch.mean(tensor)}")
            print(f"Standard deviation: {torch.std(tensor)}")
            print(f"Min: {torch.min(tensor)}")
            print(f"Max: {torch.max(tensor)}")
        else:
            print(f"\nTensor {tensor_name} not found in the file.")

```

### Inspecting the `tokenizer.model` file

The tokenizer is stored in a format unknown to me, and it is not human-readable.
The tokenizer can be loaded as below. It results in a `Llama3Tokenizer` as implemented [here](https://github.com/pytorch/torchtune/blob/75965d4281b9b76c454630d015221b9933c77bf3/torchtune/models/llama3/_tokenizer.py#L43).


```bash
pip install torch
pip install torchtune
pip install torchao
```

```python
from torchtune.models.llama3 import Llama3Tokenizer

model_name = "/path/to/model/original/tokenizer.model

tokenizer = Llama3Tokenizer(model_name)

# the tokenizer object can be inspected with `vars(tokenizer)` and `dir(tokenizer)`.
```












