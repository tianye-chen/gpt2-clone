import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        
        self.data = data
        self.encoded_data = []
        
        for entry in data:
            alpaca_entry = to_alpaca_format(entry)
            response = f"### Response:\n{entry['output']}\n\n"
            full_text = alpaca_entry + response
            self.encoded_data.append(tokenizer.encode(full_text))
            
    def __getitem__(self, idx):
        return self.encoded_data[idx]
        
    def __len__(self):
        return len(self.encoded_data)

def to_alpaca_format(entry):
    instruction = (
        f"Below is an instruction that describes a task\n"
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{entry['instruction']}\n\n"
    )
    
    input_text = f"### Input:\n{entry['input']}\n\n" if entry['input'] else ""
    
    return instruction + input_text

def collate_fn(batch, pad_token_id=50256, ignore_index=-100, max_context_length=1024):
    batch_len = max(len(seq) + 1 for seq in batch)
    input_list, target_list = [], []
    
    for seq in batch:
        # Pad sequence to ensure all sequences are the same length
        new_seq = seq + [pad_token_id] # Additional padding for use when shifting the sequence for target
        new_seq += [pad_token_id] * (batch_len - len(new_seq))
        
        # Take the entire original sequence as input and shift it for the target
        input_tensor = torch.tensor(new_seq[:-1]) # Account extra padding
        target_tensor = torch.tensor(new_seq[1:])
        
        mask = target_tensor == pad_token_id
        pad_indices = torch.nonzero(mask, as_tuple=True)[0] # Get the first index where the mask is True

        # Ignore the first padding token in the target sequence
        if pad_indices.numel() > 1:
            mask[pad_indices[0].item()] = False
        
        target_tensor.masked_fill_(mask, ignore_index)  # Set padding tokens to ignore_index
        
        target_tensor = target_tensor[:max_context_length]
        input_tensor = input_tensor[:max_context_length]
        
        input_list.append(input_tensor)
        target_list.append(target_tensor)
        
    input_tensor = torch.stack(input_list)
    target_tensor = torch.stack(target_list)
    
    return input_tensor, target_tensor
        