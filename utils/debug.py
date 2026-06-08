import torch

def mem_info(tag=""):
    torch.cuda.synchronize()
    print(f"{tag} | alloc={torch.cuda.memory_allocated()/1e6:.1f}MB "
          f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB")