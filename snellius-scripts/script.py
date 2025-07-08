import os
import torch
import torch.distributed as dist

def main():
    # Set NCCL debug info
    os.environ['NCCL_DEBUG'] = 'INFO'

    #Initialize distributed process group
    dist.init_process_group(backend='nccl')

    # Get rank and world size
    # rank = dist.get_rank()
    rank = int(dist.get_rank()) % 4

    world_size = dist.get_world_size()

    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(int(rank) % 4)

    print(f"Rank {rank}/{world_size} initialized on device {device}")

    # Simple all-reduce test
    tensor = torch.ones(1, device=device) # * rank
    print(f"Rank {rank}: Before all-reduce: {tensor.item()}")

    dist.all_reduce(tensor)
    print(f"Rank {rank}: After all-reduce: {tensor.item()}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()