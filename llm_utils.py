import torch
import numpy as np

def text_generation(model, query, tokenizer, max_generated_tokens, context_size, temperature=1.4, top_k=30, device='cuda', alpaca=False):
    query = torch.tensor(tokenizer.encode(query)).unsqueeze(0).to(device)
    results = []
    
    for _ in range(max_generated_tokens):
        input_query = query[:, -context_size:] # Limits the amount of tokens of each batch to up to context_size
        
        with torch.no_grad():
            out = model(input_query.to(device)) # Returns [batch, tokens, vocab_size]
        
        # Takes the last row of tokens in the output tensor, becomes [batch, vocab_size]
        out = out[:, -1, :]
        
        # Top-k sampling, take only the top k logits with the highest probability
        _, top_indices = torch.topk(out, top_k)
        mask = torch.full_like(out, False, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)
        out = torch.where(mask, out, torch.tensor(float('-inf'), device=out.device))
        
        probs = torch.softmax(out/temperature, dim=-1)
        next_tok_idx = torch.multinomial(probs, num_samples=1)
        query = torch.cat((query.to(device), next_tok_idx), dim=1)
        
    for i in range(len(query)):
        results.append(tokenizer.decode(query[i].tolist()))
        
    return results

def calc_batch_loss(input_batch, target_batch, model, criterion, device):
    input_batch = input_batch.long().to(device)
    target_batch = target_batch.long().to(device)
    logits = model(input_batch)
    
    loss = criterion(logits.flatten(0, 1), target_batch.flatten())
    
    return loss

def calc_total_loss(model, loader, criterion, device):
    total_loss = 0

    for input_batch, target_batch in loader:
        loss = calc_batch_loss(input_batch, target_batch, model, criterion, device)
        total_loss += loss.item()

    return total_loss / len(loader)

def train(model, train_loader, val_loader, optimizer, criterion, device, n_epochs, tokenizer, eval_freq, early_stop=0):
    train_losses, val_losses = [], []
    tokens_seen_at_step = []
    tokens_seen = 0
    step = 0
    
    for epoch in range(n_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_batch_loss(input_batch, target_batch, model, criterion, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            step += 1
            
            if step % eval_freq == 0:
                model.eval()
                train_loss = calc_total_loss(model, train_loader, criterion, device)
                val_loss = calc_total_loss(model, val_loader, criterion, device)
                
                if early_stop and len(val_losses) > early_stop:
                    val_loss_avg = np.mean(val_losses[-early_stop:])
                    if val_loss < val_loss_avg * 0.95:
                        print(f'Early stop {epoch} - {step}')
                        return train_losses, val_losses, tokens_seen_at_step
                
                tokens_seen_at_step.append(tokens_seen)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                model.train()
                
                print(f'Epoch: {epoch} - {step}, train_loss: {train_loss}, val_loss: {val_loss}')
                
    return train_losses, val_losses, tokens_seen_at_step