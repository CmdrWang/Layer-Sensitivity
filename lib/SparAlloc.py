import time 
import heapq 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
from pdb import set_trace as st 
from collections import defaultdict
from torch.cuda.amp import autocast
import pandas as pd
import os
from tqdm import tqdm
from rich.tree import Tree
from rich import print
from rich.console import Console

def get_sparalloc_strategy_tree():
    return {
        "Default": {
            "Uniform": {
                "Strategy": "Uniform",
                "Level": "-",
                "Time": "~0s",
                "Explanation": "Uniform sparsity allocation across all Transformer blocks."
            }
        },
        "Custom-designed": {
            "Mag": {
                "L1-norm": {
                    "Time": "~1s",
                    "Explanation": "Use the L1 norm of weights to compute layer importance."
                },
                "L2-norm": {
                    "Time": "~1s",
                    "Explanation": "Use the L2 norm of weights to compute layer importance."
                }
            },
            "Blockwise Perplexity": {
                "Time": "~1h",
                "Explanation": "Remove each transformer block and measure perplexity change."
            },
            "Cosine-Similarity": {
                "Time": "~1min",
                "Explanation": "Measure cosine similarity between adjacent hidden representations."
            }
        },
        "Extracted from pruning algorithm": {
            "Simple-Function": {
                "Linear-Increase": {
                    "Time": "~0s",
                    "Explanation": "Sparsity increases linearly from bottom to top."
                },
                "Linear-Decrease": {
                    "Time": "~0s",
                    "Explanation": "Sparsity decreases linearly from bottom to top."
                },
                "Log-Increase": {
                    "Time": "~0s",
                    "Explanation": "Sparsity increases logarithmically across layers."
                },
                "Log-Decrease": {
                    "Time": "~0s",
                    "Explanation": "Sparsity decreases logarithmically across layers."
                }
            },
            "FrontBackByPass": {
                "Time": "~0s",
                "Explanation": "Skip pruning for the first n and last m transformer blocks."
            }
        },
        "Open-source": {
            "OWL": {
                "Time": "~5min",
                "Explanation": "Based on outlier weight distribution across transformer layers."
            }
        }
    }
def show_sparalloc_method_tree():
    from rich.tree import Tree
    from rich import print
    from rich.console import Console

    data = get_sparalloc_strategy_tree()
    console = Console()
    tree = Tree("[bold white on black]SparAlloc Strategies[/]")  # 更清晰对比感

    for category, content in data.items():
        cat_node = tree.add(f"[bold yellow]{category}[/]")  # 分类：亮黄
        for method, value in content.items():
            if isinstance(value, dict) and "Explanation" not in value:
                # 有子策略的策略块，如 Mag → L1/L2
                meth_node = cat_node.add(f"[bold cyan]{method}[/]")  # 一级策略：亮青
                for sub_name, sub_info in value.items():
                    strat_text = (
                        f"[magenta]{sub_name}[/] "
                        f"- [white]{sub_info['Explanation']}[/] "
                        f"[dim]({sub_info['Time']})[/]"
                    )
                    meth_node.add(strat_text)
            else:
                # 单一策略项，如 Blockwise Perplexity
                strat_text = (
                    f"[bold cyan]{method}[/] "
                    f"- [white]{value['Explanation']}[/] "
                    f"[dim]({value['Time']})[/]"
                )
                cat_node.add(strat_text)

    console.print(tree)


def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None,}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache
    
    position_ids=None

    return inps, outs, attention_mask, position_ids 




def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def check_sparsity_mask(mask):


    W = mask
    count = 0 
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()



    print(f" density {float(count)/total_params:.6f}")



def check_outlier(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.max(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def check_outlier_mean(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity



def OWL_SparAlloc(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            
            
            
 
            layer_wmetric.append(W_metric)    
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps



        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        
        


    print ("before adjustment",all_layer_ratio)

    

    
    
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

   
    
                
        
    
    print ("after adjustment",all_layer_ratio)
    all_layer_ratio = [1-x for x in all_layer_ratio]
    return all_layer_ratio

def Func_SparAlloc(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):

# 参数设置
    n_layers = len(model.model.layers)
    global_sparsity = args.sparsity_ratio
    strategy = args.func_strategy
    min_sparsity = args.min_sparsity  # 原始稀疏率起点
    max_sparsity = args.max_sparsity  # 原始稀疏率终点

    def generate_raw_sparsity(strategy):
        if strategy in ('log_increase', 'log_decrease'):
            x = np.arange(n_layers)
            raw = min_sparsity + (max_sparsity - min_sparsity) / np.log(n_layers) * np.log(1 + x)
            if strategy == 'log_decrease':
                raw = raw[::-1]
        elif strategy in ('linear_increase', 'linear_decrease'):
            raw = np.linspace(min_sparsity, max_sparsity, n_layers)
            if strategy == 'linear_decrease':
                raw = raw[::-1]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return raw

    def normalize_mean(sparsity, global_target):
        # 保留 shape，线性缩放使 mean 对齐
        scale = (global_target * len(sparsity)) / np.sum(sparsity)
        return sparsity * scale

# 可视化所有策略

    raw = generate_raw_sparsity(strategy)
    scaled = normalize_mean(raw, global_sparsity)
    return scaled

def blockwise_sparsity(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    csv_path = os.path.join("/home/zhujunhan/Projects/OWL", "freeze_ppl_with_sparsity.csv")
    df = pd.read_csv(csv_path)

    # 读取allocated_sparsity列
    sparsity_list = df['allocated_sparsity'].tolist()

    # 确保层数匹配
    total_blocks = len(sparsity_list)
    return sparsity_list

def mag_sparsity(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    model.eval()


    norm_type = args.norm_type
# 修改后的代码
    layer_scores = []

    for i, layer in tqdm(enumerate(model.model.layers), total=32):
        total_mag = 0.0

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            module = getattr(layer.self_attn, proj_name)
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data.to(torch.float32)
                mag = torch.norm(weight, p=norm_type).item()
                total_mag += mag

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            module = getattr(layer.mlp, proj_name)
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data.to(torch.float32)
                mag = torch.norm(weight, p=norm_type).item()
                total_mag += mag

        print(f"Layer {i}: Magnitude = {total_mag:.4e}")
        layer_scores.append(total_mag)


# 转为 DataFrame
    df = pd.DataFrame({
        "layer": list(range(len(layer_scores))),
        "score": layer_scores
    })
    lamda = 0.05
    df['score'] = np.log(df['score'])
# 归一化为重要性分数
    df['norm_score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min()) ** lamda
# Sparsity 分配：保留度越高的层，稀疏越少
    total_sparsity = 0.7
    n_layers = len(df)
    allocatable_sparsity = total_sparsity * n_layers

    inverse_score = 1 - df['norm_score']

    weight = inverse_score / inverse_score.sum()
    df['allocated_sparsity'] = weight * allocatable_sparsity
    last_sparsity = [x for x in df['allocated_sparsity']]
    return last_sparsity