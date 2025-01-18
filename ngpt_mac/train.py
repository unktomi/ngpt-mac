import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import NGPT

def train(
    model,
    train_dataset,
    *,
    batch_size=16,
    grad_accum_steps=8,
    learning_rate=3e-4,
    max_iters=100000,
    lr_decay=True,
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    dataloader = iter(dataloader)
    
    # Training loop
    pbar = tqdm(range(max_iters), mininterval=10.0, desc='training')
    
    for i in pbar:
        for _ in range(grad_accum_steps):
            # Get batch
            try:
                batch = next(dataloader)
            except StopIteration:
                dataloader = iter(DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True
                ))
                batch = next(dataloader)
                
            # Get loss
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            
            # Backward
            loss.backward()
            
        # Update
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update learning rate
        if lr_decay:
            lr = learning_rate * (0.1 ** (i / max_iters))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Log
        pbar.set_description(f'loss: {loss.item():.4f}')

def create_model(
    vocab_size,
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    ff_mult=4
):
    return NGPT(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        ff_mult=ff_mult
    )