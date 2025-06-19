import json
import os
import time
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['DEEPSPEED_DISABLE_MPI'] = '1'
os.environ['LOCAL_RANK'] = '0'

import numpy as np
import tensorflow as tf
import tiktoken
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader, random_split

from InstructionDataset import InstructionDataset, collate_fn
from llm_utils import train
from SimpleGPT import SimpleGPT


if __name__ == "__main__":
    MODEL_DIR = "./355M"
    USE_MLA = False # Set to True to use Multihead Latent Attention
    CONFIG = json.load(open("model_config.json", "r"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleGPT(CONFIG, use_mla=USE_MLA)
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f'Using {DEVICE}')
    print(f'Training with {"Multihead Latent Attention" if USE_MLA else "Multihead Attention"}')

    # Since the original model is trained with TensorFlow, we read it using TensorFlow

    tf_checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
    tf_model_settings = json.load(open(os.path.join(MODEL_DIR, "hparams.json")))

    params = {"blocks": [{} for _ in range(tf_model_settings['n_layer'])]}

    for name, _ in tf.train.list_variables(tf_checkpoint_path):
        var = np.squeeze(tf.train.load_variable(tf_checkpoint_path, name))
        var_name_no_prefix = name.split("/")[1:]
        
        target_dict = params
        
        if var_name_no_prefix[0].startswith("h"):
            layer_id = int(var_name_no_prefix[0][1:])
            target_dict = params["blocks"][layer_id]
        
        for key in var_name_no_prefix[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        
        last_key = var_name_no_prefix[-1]
        target_dict[last_key] = var
        
    model.pos_emb.weight = nn.Parameter(torch.tensor(params['wpe']))
    model.token_emb.weight = nn.Parameter(torch.tensor(params['wte']))

    # Load pretrained model parameters
    for i in range(len(params['blocks'])):
        q_w, k_w, v_w = np.split((params['blocks'][i]['attn']['c_attn'])['w'], 3, axis=-1)
        q_b, k_b, v_b = np.split((params['blocks'][i]['attn']['c_attn'])['b'], 3, axis=-1)
        
        if not USE_MLA:
            model.transformer_blocks[i].attn.W_q.weight = nn.Parameter(torch.tensor(q_w.T))
            model.transformer_blocks[i].attn.W_k.weight = nn.Parameter(torch.tensor(k_w.T))
            model.transformer_blocks[i].attn.W_v.weight = nn.Parameter(torch.tensor(v_w.T))
            
            model.transformer_blocks[i].attn.W_q.bias = nn.Parameter(torch.tensor(q_b))
            model.transformer_blocks[i].attn.W_k.bias = nn.Parameter(torch.tensor(k_b))
            model.transformer_blocks[i].attn.W_v.bias = nn.Parameter(torch.tensor(v_b))
        
        model.transformer_blocks[i].attn.output_projection.weight = nn.Parameter(torch.tensor(params['blocks'][i]['attn']['c_proj']['w'].T))
        model.transformer_blocks[i].attn.output_projection.bias = nn.Parameter(torch.tensor(params['blocks'][i]['attn']['c_proj']['b']))
        
        model.transformer_blocks[i].ff_block[1].weight = nn.Parameter(torch.tensor(params['blocks'][i]['mlp']['c_fc']['w'].T))
        model.transformer_blocks[i].ff_block[1].bias = nn.Parameter(torch.tensor(params['blocks'][i]['mlp']['c_fc']['b']))
        model.transformer_blocks[i].ff_block[3].weight = nn.Parameter(torch.tensor(params['blocks'][i]['mlp']['c_proj']['w'].T))
        model.transformer_blocks[i].ff_block[3].bias = nn.Parameter(torch.tensor(params['blocks'][i]['mlp']['c_proj']['b']))
        
        model.transformer_blocks[i].attn_block[0].scale = nn.Parameter(torch.tensor(params['blocks'][i]['ln_1']['g']))
        model.transformer_blocks[i].attn_block[0].shift = nn.Parameter(torch.tensor(params['blocks'][i]['ln_1']['b']))
        model.transformer_blocks[i].ff_block[0].scale = nn.Parameter(torch.tensor(params['blocks'][i]['ln_2']['g']))
        model.transformer_blocks[i].ff_block[0].shift = nn.Parameter(torch.tensor(params['blocks'][i]['ln_2']['b']))

    model.final_norm.scale = nn.Parameter(torch.tensor(params['g']))
    model.final_norm.shift = nn.Parameter(torch.tensor(params['b']))
    model.out.weight = nn.Parameter(torch.tensor(params['wte']))

    model.to(DEVICE)

    alpaca_data = json.load(open("alpaca_data.json", "r"))

    train_size = int(len(alpaca_data) * 0.8)
    test_size = int(len(alpaca_data) * 0.1)
    val_size = len(alpaca_data) - train_size - test_size

    dataset = InstructionDataset(alpaca_data, tokenizer)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    # Pre-fill the collate function with the context length
    collate_fn = partial(collate_fn, max_context_length=CONFIG['context_length'])

    train_loader = DataLoader(train_dataset, 
                            batch_size=CONFIG['batch_size'],
                            collate_fn=collate_fn,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=False)

    test_loader = DataLoader(test_dataset,
                            batch_size=CONFIG['batch_size'],
                            collate_fn=collate_fn,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False)

    val_loader = DataLoader(val_dataset,
                            batch_size=CONFIG['batch_size'],
                            collate_fn=collate_fn,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    time_start = time.localtime()
    print(f'Training started at {time.strftime("%H:%M:%S", time_start)}')
    
    ds_model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=CONFIG['deepspeed_config'],
    )

    train_losses, val_losses, tokens_seen = train(
        model=ds_model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        tokenizer=tokenizer,
        device=DEVICE,
        n_epochs=1,
        eval_freq=10,
        early_stop=1,
        use_deepspeed=True
    )

    time_end = time.localtime()
    print(f'Training finished at {time.strftime("%H:%M:%S", time_end)}')

    print(f'Train losses: {train_losses[-1]} \t Val losses: {val_losses[-1]} \t Tokens seen: {tokens_seen[-1]}')

    torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, 
            "simple_gpt_355M_MA.pth"
        )