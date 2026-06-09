import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from datetime import datetime, date
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, Tuple
from lib.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm
except ImportError:
    Qwen3Attention = None
    Qwen3RMSNorm = None

import fnmatch
import gc

import lib.torch_pruning as tp 
from lib.pruner import hf_llama_pruner as llama_pruner
from lib.utils.logger import LoggerWithDepth
from lib.evaluator.ppl import PPLMetric
from lib.datasets.loader import get_examples, get_loaders
from lib.templates.prompts import prompts

def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


# Model configuration mappings
MODEL_CONFIGS = {
    # LLaMA family
    "llama": {
        "layers_key": "model.layers",
        "attention_key": "self_attn",
        "mlp_key": "mlp",
        "norm_key": "input_layernorm",
        "post_norm_key": "post_attention_layernorm",
        "embed_key": "model.embed_tokens",
        "lm_head_key": "lm_head",
        "seqlen": 2048,
        "supported_models": ["llama-7b", "llama-13b", "llama2-7b", "llama2-13b", "llama2-70b"],
        "model_type": "llama"
    },
    # Aya-Expanse family
    "aya-expanse": {
        "layers_key": "model.layers",
        "attention_key": "self_attn",
        "mlp_key": "mlp",
        "norm_key": "input_layernorm",
        "post_norm_key": "post_attention_layernorm",
        "embed_key": "model.embed_tokens",
        "lm_head_key": "lm_head",
        "seqlen": 2048,
        "supported_models": ["aya-expanse-8b", "aya-expanse-32b"],
        "model_type": "aya-expanse"
    },
    # Cohere family
    "cohere": {
        "layers_key": "model.layers",
        "attention_key": "self_attn",
        "mlp_key": "mlp",
        "norm_key": "input_layernorm",
        "post_norm_key": None,  # Cohere doesn't have post attention norm
        "embed_key": "model.embed_tokens",
        "lm_head_key": "lm_head",
        "seqlen": 2048,
        "supported_models": ["cohere", "c4ai", "command-r", "command-r-plus", "aya-expanse"],
        "model_type": "cohere",
        "has_position_embeddings": True,
        "position_embeddings_key": "rotary_emb"
    },
    # Qwen family
    "qwen": {
        "layers_key": "model.layers",
        "attention_key": "self_attn",
        "mlp_key": "mlp",
        "norm_key": "input_layernorm",
        "post_norm_key": "post_attention_layernorm",
        "embed_key": "model.embed_tokens",
        "lm_head_key": "lm_head",
        "seqlen": 2048,
        "supported_models": ["Qwen3-8B", "Qwen3-32B", "qwen3", "qwen-3"],
        "model_type": "qwen",
        "has_position_embeddings": True,
        "position_embeddings_key": "rotary_emb"
    },
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model-specific configuration.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        Model configuration dictionary
    """
    family = detect_model_family(model_name)
    return MODEL_CONFIGS[family]

def detect_model_family(model_name: str) -> str:
    """
    Detect the model family based on model name.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        Model family string
    """
    model_name_lower = model_name.lower()
    
    # Check for Cohere models first (more specific)
    if any(keyword in model_name_lower for keyword in ["cohere", "c4ai", "command-r", "aya-expanse"]):
        return "cohere"
    
    # Check for Qwen3 models
    if any(keyword in model_name_lower for keyword in ["qwen3", "qwen-3"]):
        return "qwen"
    
    for family, config in MODEL_CONFIGS.items():
        for supported_model in config["supported_models"]:
            if supported_model in model_name_lower:
                return family
    
    # Default to llama if no specific family detected
    return "llama"

def is_model_supported(model_name: str) -> bool:
    """
    Check if a model is supported by DLP.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        True if supported, False otherwise
    """
    try:
        family = detect_model_family(model_name)
        return family in MODEL_CONFIGS
    except:
        return False

def load_model_with_support(model_name: str, **kwargs) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with proper configuration for different model families.
    
    Args:
        model_name: Name or path of the model
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    default_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "trust_remote_code": True
    }

    default_kwargs.update(kwargs)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **default_kwargs)
    
    # Load tokenizer
    if "qwen" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Set pad token for Qwen models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set model sequence length
    config = get_model_config(model_name)
    model.seqlen = config["seqlen"]
    
    return model, tokenizer

def get_llm(model_name, device="cuda"):
    """
    Load model with proper support for different model families.
    """
    # Check if model is supported
    if not is_model_supported(model_name):
        print(f"Warning: Model {model_name} may not be fully supported. Proceeding with default configuration.")

    model, tokenizer = load_model_with_support(model_name)

    # print("="*30)
    # print(f"Attention Class: {type(model.model.layers[0].self_attn)}")
    # print(f"RMSNorm Class:   {type(model.model.layers[0].input_layernorm)}")
    # print("="*30)
    return model, tokenizer

class LoadDatetime(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)

def eval_ppl(model, tokenizer, device=torch.device("cuda:0"), dataset="c4", languages=['en']):
    results = dict()
    for lang in languages:
        # Print status
        print(f"evaluating on {dataset} with lang {lang}")

        # Get the test loader
        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, language=lang, tokenizer=tokenizer
        )

        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl = eval_ppl_dataset(model, testloader, args.eval_batch_size, device)
        results[lang] = ppl
    return results 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_dataset(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # st()
        
        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to("cuda:1")
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        # print ("nlls",nlls)
        sys.stdout.flush()

    
    print ('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Clean up to prevent memory leaks
    del nlls
    # torch.cuda.empty_cache()

    return ppl.item()

def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    model, tokenizer = get_llm(args.base_model)

    if args.device != "cpu":
        model.half()
        
    model.to(args.device)

    if args.test_before_train:
        previous_ppl_results_path = os.path.join(logger.log_dir, "previous_ppl_results.json")

        if os.path.exists(previous_ppl_results_path) and args.force_recompute == False:
            print(f"Found existing previous PPL results at {previous_ppl_results_path}, loading...")
            with open(previous_ppl_results_path, "r") as f:
                previous_ppl_results = json.load(f)
            for lang, ppl in previous_ppl_results.items():
                print(f"ppl on {args.dataset} ({lang}): {ppl}")
            print("Skipping previous PPL computation.")
            logger.log(f"Loaded previous PPL results from {previous_ppl_results_path}")
        else:
            print("No existing previous PPL results found or force_recompute is True, computing PPL...")
            logger.log("\n==================Generation Results before Pruning================\n")
            model.eval()
            with torch.no_grad():
                for prompt in prompts:
                    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)

                    generation_output = model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        top_k=50,
                        max_length=args.max_seqlen,
                        top_p=args.top_p,
                        temperature=args.temperature,
                    )
                    
                    result = tokenizer.decode(generation_output[0])
                    logger.log(result)

            print("Computing PPL evaluation before pruning...")
            previous_ppl_results = eval_ppl(model, tokenizer, device = model.device, dataset=args.dataset, languages=args.eval_languages)
            for lang, ppl in previous_ppl_results.items():
                logger.log(f"ppl on {args.dataset} ({lang}): {ppl}")
            with open(previous_ppl_results_path, "w") as f:
                json.dump(previous_ppl_results, f, indent=2, cls=LoadDatetime)
            
            if args.test_before_train_then_end_exp:
                print("Ending experiment as per --test_before_train_then_end_exp flag.")
                return 0    

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            # "customized_pruners": {
            #     LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            # },
            "customized_pruners": {
                **{LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner},
                **({Qwen3RMSNorm: llama_pruner.hf_rmsnorm_pruner} if Qwen3RMSNorm is not None else {}),
                **{LlamaAttention: llama_pruner.hf_attention_pruner},
                **({Qwen3Attention: llama_pruner.hf_attention_pruner} if Qwen3Attention is not None else {}),
            },
            "root_module_types": None, 
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)],
            "multi_lang_important": args.multi_lang_important,
            "save_topk_results": args.save_topk_results,
            "logger_path": logger.log_dir,
            "logger": logger,
            "save_group_json": args.save_group_json,
        }
        # logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        # logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        if args.multi_lang_important:
            logger.log("Prepare multi-lingual calibration dataset: {}".format(args.calibration_languages))

            multi_lang_imp = {}

            for lang in args.calibration_languages:

                pruner = tp.pruner.MetaPruner(
                    model,
                    forward_prompts,
                    **kwargs
                )
                model.zero_grad()

                if pruner_type in ['taylor']:
                    if args.dataset == 'c4':
                        print("Processing language: {}".format(lang))
                        example_prompts, _ = get_examples('c4', tokenizer, nsamples = args.num_examples, seqlen = args.seqlen, seed=args.seed, language = lang, multilingual_num_examples = args.multilingual_num_examples)
                    elif args.dataset == 'mmlu':
                        print("Processing task: {}".format(lang))
                        example_prompts, _ = get_examples('mmlu', tokenizer, nsamples = args.num_examples, seqlen = args.seqlen, seed=args.seed, language = lang, multilingual_num_examples = args.multilingual_num_examples)

                    batch_size = args.batch_size
                    n_samples = min(args.num_examples, len(example_prompts))

                    for i in range(args.iterative_steps):

                        if args.taylor in ['param_mix', 'param_second']:
                            logger.log(f"language = {lang}, Calculating acc_grad (sample-by-sample) in iterative step = {i}")
                            
                            model.zero_grad()

                            for j in tqdm(range(n_samples)):
                                batch_input = example_prompts[j].unsqueeze(0).to(args.device)
                                loss = model(batch_input, labels=batch_input).loss
                                loss.backward()

                                for module_param in model.parameters():
                                    module_param.grad = module_param.grad * module_param.grad / args.num_examples
                                    if hasattr(module_param, 'acc_grad'):
                                        module_param.acc_grad += module_param.grad
                                    else:
                                        module_param.acc_grad = copy.deepcopy(module_param.grad)
                                
                                model.zero_grad()
                                del batch_input, loss.grad
                            
                            # torch.cuda.empty_cache()
                            logger.log(f"[{lang}] Iter step {i} acc_grad calculation done.")
                        
                        # model.zero_grad()

                        logger.log(f"language = {lang}, Start Backwarding in iterative step = {i}")

                        for start in tqdm(range(0, n_samples, batch_size)):
                            end = min(start + batch_size, n_samples)
                            batch_input = example_prompts[start:end].to(args.device)

                            # forward & compute loss
                            loss = model(batch_input, labels=batch_input).loss

                            # Scale loss to align with total num_examples
                            loss = loss / (n_samples / batch_size)

                            loss.backward()

                            del batch_input, loss
                            torch.cuda.empty_cache()

                        logger.log(f"[{lang}] Iter step {i} done, memory: {torch.cuda.memory_allocated(args.device)/1024**2:.2f} MB")

                global_imp, global_importance = pruner.culculate_global_importance()

                multi_lang_imp[lang] = global_imp

                # Clean the gradient in the model
                model.zero_grad()
                for name, module in model.named_parameters():
                    if 'weight' in name:
                        module.grad = None

                del pruner
            
            len_imp = len(multi_lang_imp[args.calibration_languages[0]])
            print('#####################################################')
            print(len_imp)
            print('#####################################################')

            if args.merge_methods is not None:
                print("merge_methods", args.merge_methods)

                if 'mean' in args.merge_methods:
                    print("Using mean merge method for multi-lingual importance...")
                    multi_lang_imp = sum(multi_lang_imp.values()) / len(multi_lang_imp)
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))
                elif 'max' in args.merge_methods:
                    print("Using max merge method for multi-lingual importance...")
                    multi_lang_imp = torch.max(torch.stack(list(multi_lang_imp.values())), dim=0)[0]
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))
                elif 'min' in args.merge_methods:
                    print("Using min merge method for multi-lingual importance...")
                    multi_lang_imp = torch.min(torch.stack(list(multi_lang_imp.values())), dim=0)[0]
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))
                elif 'sum' in args.merge_methods:
                    print("Using sum merge method for multi-lingual importance...")
                    multi_lang_imp = sum(multi_lang_imp.values())
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))
                elif 'weighted' in args.merge_methods:
                    print("Using weighted merge method for multi-lingual importance...")
                    if args.merge_weights is None:
                        args.merge_weights = [1.0 / len(args.calibration_languages)] * len(args.calibration_languages)
                    assert len(args.merge_weights) == len(args.calibration_languages)
                    total_weight = sum(args.merge_weights)
                    normalized_weights = [w / total_weight for w in args.merge_weights]
                    print("normalized_weights", normalized_weights)
                    multi_lang_imp = sum(multi_lang_imp[lang] * weight for lang, weight in zip(args.calibration_languages, normalized_weights))
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))
                elif 'weighted_lang' in args.merge_methods:
                    print("Using weighted_lang merge method for multi-lingual importance...")
                    # Normalize importance for each language
                    for lang in multi_lang_imp:
                        lang_imp = multi_lang_imp[lang]
                        lang_imp = (lang_imp - lang_imp.min()) / (lang_imp.max() - lang_imp.min() + 1e-12)
                        multi_lang_imp[lang] = lang_imp
                    # Merge normalized importances by mean
                    stacked_imp = torch.stack(list(multi_lang_imp.values()))
                    multi_lang_imp = stacked_imp.mean(dim=0)
                    # Apply topk sorting to get indices
                    selected_mask = torch.ones(len_imp, dtype=torch.bool, device=stacked_imp.device)
                    _, topk_indices = torch.topk(multi_lang_imp, k=len_imp, dim=-1, largest=True, sorted=True)
                    # Mark important indices as False (keep, don't prune)
                    selected_mask[topk_indices] = False
                    multi_lang_imp = selected_mask
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))
                    
                elif 'weighted_lang_min' in args.merge_methods:
                    print("Using weighted_lang_min merge method for multi-lingual importance...")
                    # Normalize importance for each language
                    for lang in multi_lang_imp:
                        lang_imp = multi_lang_imp[lang]
                        lang_imp = (lang_imp - lang_imp.min()) / (lang_imp.max() - lang_imp.min() + 1e-12)
                        multi_lang_imp[lang] = lang_imp
                    # Merge normalized importances by min
                    stacked_imp = torch.stack(list(multi_lang_imp.values()))
                    multi_lang_imp = stacked_imp.min(dim=0)[0]
                    # Apply topk sorting to get indices
                    selected_mask = torch.ones(len_imp, dtype=torch.bool, device=stacked_imp.device)
                    _, topk_indices = torch.topk(multi_lang_imp, k=len_imp, dim=-1, largest=True, sorted=True)
                    # Mark important indices as False (keep, don't prune)
                    selected_mask[topk_indices] = False
                    multi_lang_imp = selected_mask
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))

                elif 'weighted_lang_max' in args.merge_methods:
                    print("Using weighted_lang_max merge method for multi-lingual importance...")
                    # Normalize importance for each language
                    for lang in multi_lang_imp:
                        lang_imp = multi_lang_imp[lang]
                        lang_imp = (lang_imp - lang_imp.min()) / (lang_imp.max() - lang_imp.min() + 1e-12)
                        multi_lang_imp[lang] = lang_imp
                    # Merge normalized importances by max
                    stacked_imp = torch.stack(list(multi_lang_imp.values()))
                    multi_lang_imp = stacked_imp.max(dim=0)[0]
                    # Apply topk sorting to get indices
                    selected_mask = torch.ones(len_imp, dtype=torch.bool, device=stacked_imp.device)
                    _, topk_indices = torch.topk(multi_lang_imp, k=len_imp, dim=-1, largest=True, sorted=True)
                    # Mark important indices as False (keep, don't prune)
                    selected_mask[topk_indices] = False
                    multi_lang_imp = selected_mask
                    print("multi_lang_imp", multi_lang_imp.shape, len(multi_lang_imp))

                elif 'topk_imp' in args.merge_methods:
                    print("Using topk_imp merge method for multi-lingual importance...")
                    k = args.topk_imp if args.topk_imp is not None else len_imp
                    sparsity = args.pruning_ratio if args.pruning_ratio is not None else 0.5

                    # Stack all language importances, shape: (num_langs, num_elements)
                    lang_list = list(multi_lang_imp.keys())
                    stacked_imp = torch.stack(list(multi_lang_imp.values()))
                    num_langs, num_elems = stacked_imp.shape

                    # Store top-k indices for each language
                    topk_imp_index = []
                    for lang_imp in multi_lang_imp.values():
                        _, topk_indices = torch.topk(lang_imp, k=k, dim=-1, largest=True, sorted=True)
                        topk_imp_index.append(topk_indices)

                    # Initialize all positions as True (to be pruned)
                    selected_mask = torch.ones(num_elems, dtype=torch.bool, device=stacked_imp.device)

                    filled = 0
                    j = 0
                    while filled < num_elems:
                        lang_order = range(num_langs) if j % 2 == 0 else reversed(range(num_langs))
                        exhausted = True
                        for i in lang_order:
                            if j >= k:
                                continue
                            idx = int(topk_imp_index[i][j].item())
                            if not selected_mask[idx]:
                                continue
                            # Mark important index as False (keep, don't prune)
                            selected_mask[idx] = False
                            filled += 1
                            exhausted = False
                            if filled >= num_elems:
                                break
                        if exhausted:
                            j += 1
                            if j >= k:
                                break
                        else:
                            j += 1

                    # Fill remaining with highest importance values
                    if filled < num_elems:
                        remaining_indices = selected_mask.nonzero(as_tuple=True)[0]
                        if len(remaining_indices) > 0:
                            remaining_vals = stacked_imp[0, remaining_indices]
                            remaining_vals, sorted_idx = torch.sort(remaining_vals, descending=True)
                            extra_indices = remaining_indices[sorted_idx[:num_elems - filled]]
                            selected_mask[extra_indices] = False

                    # Adjust based on sparsity (True = prune)
                    num_prune = int(num_elems * sparsity)
                    current_prune = selected_mask.sum().item()

                    if current_prune > num_prune:
                        # Too many to prune -> restore some (keep more important ones)
                        avg_imp = stacked_imp.mean(dim=0)
                        prune_indices = selected_mask.nonzero(as_tuple=True)[0]
                        prune_vals = avg_imp[prune_indices]
                        _, sorted_idx = torch.sort(prune_vals, descending=True)
                        keep_indices = prune_indices[sorted_idx[:current_prune - num_prune]]
                        selected_mask[keep_indices] = False
                    elif current_prune < num_prune:
                        # Too few to prune -> prune more (remove less important ones)
                        keep_indices = (~selected_mask).nonzero(as_tuple=True)[0]
                        avg_imp = stacked_imp.mean(dim=0)
                        keep_vals = avg_imp[keep_indices]
                        _, sorted_idx = torch.sort(keep_vals)
                        extra_indices = keep_indices[sorted_idx[:num_prune - current_prune]]
                        selected_mask[extra_indices] = True

                    multi_lang_imp = selected_mask

                    print("#####################################################")
                    print("n_pruned:", multi_lang_imp.sum().item(),
                        "target_sparsity:", sparsity,
                        "initial_total_channels:", multi_lang_imp.shape)
                    logger.log("n_pruned: {}, target_sparsity: {}, initial_total_channels: {}".format(
                        multi_lang_imp.sum().item(), sparsity, multi_lang_imp.shape))
                    print("#####################################################")
                else:
                    raise NotImplementedError


            pruner = tp.pruner.MetaPruner(
                model,
                forward_prompts,
                **kwargs
            )
            model.zero_grad()

            all_groups_details = pruner.step(multi_lang_imp = multi_lang_imp, global_importance = global_importance, save_topk_results = args.save_topk_results)

            json_path = os.path.join(logger.log_dir, f"group_details.json")
            with open(json_path, "w") as f:
                json.dump(all_groups_details, f, indent=2)
            print(f"Group details saved to {json_path}")

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            i = 0
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

            # Clean the gradient in the model
            model.zero_grad()
            for name, module in model.named_parameters():
                if 'weight' in name:
                    module.grad = None

            del pruner

        else:
            pruner = tp.pruner.MetaPruner(
                model,
                forward_prompts,
                **kwargs
            )
            model.zero_grad()

            logger.log("Start Pruning, model: Remix Calibration data")
            for i in range(args.iterative_steps):

                if pruner_type in ['taylor']:

                    if args.dataset == 'c4':
                        example_prompts, _ = get_examples('c4',
                                                        tokenizer,
                                                        nsamples = args.num_examples,
                                                        seqlen = args.seqlen,
                                                        seed=args.seed,
                                                        languages = args.calibration_languages,
                                                        multilingual_num_examples = args.multilingual_num_examples)
                    elif args.dataset == 'mmlu':
                        example_prompts, _ = get_examples('mmlu',
                                                        tokenizer,
                                                        nsamples = args.num_examples,
                                                        seqlen = args.seqlen,
                                                        seed=args.seed,
                                                        languages = args.calibration_languages)

                    logger.log("Start Backwarding in iterative steps = {}...".format(i))
                    if args.taylor in ['param_mix', 'param_second']:
                        for j in tqdm(range(args.num_examples)):
                            batch_input = example_prompts[j].unsqueeze(0).to(args.device)
                            loss = model(batch_input, labels=batch_input).loss
                            # logger.log("Loss = {}".format(loss))
                            loss.backward()

                            for module_param in model.parameters():
                                module_param.grad = module_param.grad * module_param.grad / args.num_examples
                                if hasattr(module_param, 'acc_grad'):
                                    module_param.acc_grad += module_param.grad
                                else:
                                    module_param.acc_grad = copy.deepcopy(module_param.grad)
                            model.zero_grad()
                            del loss.grad

     
                    batch_size = args.batch_size
                    n_samples = min(args.num_examples, len(example_prompts))

                    for i in range(args.iterative_steps):
                        logger.log(f"Mixed language, Start Backwarding in iterative step = {i}")

                        for start in tqdm(range(0, n_samples, batch_size)):
                            end = min(start + batch_size, n_samples)
                            batch_input = example_prompts[start:end].to(args.device)

                            # forward & compute loss
                            loss = model(batch_input, labels=batch_input).loss

                            # Scale loss to align with total num_examples
                            loss = loss / (n_samples / batch_size)

                            loss.backward()

                            del batch_input, loss
                            torch.cuda.empty_cache()

                        logger.log(f"[Mixed Langs] Iter step {i} done, memory: {torch.cuda.memory_allocated(args.device)/1024**2:.2f} MB")


                # pruner.step()

            # pruner = tp.pruner.MetaPruner(
            #     model,
            #     forward_prompts,
            #     **kwargs
            # )
            # model.zero_grad()

            global_imp, global_importance = pruner.culculate_global_importance()
            
            all_groups_details = pruner.step(multi_lang_imp = global_imp, global_importance = global_importance, save_topk_results = args.save_topk_results)
            
            json_path = os.path.join(logger.log_dir, f"group_details.json")
            with open(json_path, "w") as f:
                json.dump(all_groups_details, f, indent=2)
            print(f"Group details saved to {json_path}")

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

            # Clean the gradient in the model
            model.zero_grad()
            for name, module in model.named_parameters():
                if 'weight' in name:
                    module.grad = None

            del pruner

    elif args.channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                #LlamaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
            "multi_lang_important": args.multi_lang_important,
        }

        if args.multi_lang_important:
            logger.log("Prepare multi-lingual calibration dataset: {}".format(args.calibration_languages))

            multi_lang_imp = {}

            for lang in args.calibration_languages:

                pruner = tp.pruner.MetaPruner(
                    model,
                    forward_prompts,
                    **kwargs
                )
                model.zero_grad()


                print("Processing language: {}".format(lang))
                example_prompts, _ = get_examples('c4', tokenizer, nsamples = args.num_examples, seqlen = args.seqlen, language = lang, seed=args.seed)
                example_prompts = example_prompts.to(args.device)

                
                for i in range(args.iterative_steps):
                    if pruner_type in ['taylor']:
                        logger.log("language = {}, Start Backwarding in iterative step = {}".format(lang, i))
                        loss = model(example_prompts, labels=example_prompts).loss
                        logger.log("Language = {}, Iterative step Loss = {}".format(lang, loss))
                        loss.backward()

                # lang_imp = pruner.culculate_global_importance(lang = lang)
                global_importance = pruner.step(interactive=False, lang=lang)

                multi_lang_imp[lang] = lang_imp

                print(lang_imp)

                # The logic of this part need to be checked again
                # Clean the gradient in the model
                model.zero_grad()
                for name, module in model.named_parameters():
                    if 'weight' in name:
                        module.grad = None

                # modify inferece-related attributes
                model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
                model.zero_grad()
                
                del pruner

            print(multi_lang_imp)
                
                
        else: 
            pruner = tp.pruner.MetaPruner(
                model,
                forward_prompts,
                **kwargs
            )
            model.zero_grad()

            logger.log("Start Pruning")
            for i in range(args.iterative_steps):

                if pruner_type in ['taylor']:
                    # example_prompts = get_examples('bookcorpus', tokenizer, 10, seqlen = 64)
                    # get_examples(dataset, tokenizer, nsamples, seqlen = 128, seed=42, languages=None)
                    example_prompts, _ = get_examples('c4', tokenizer, nsamples = args.num_examples, seqlen = args.seqlen, seed=args.seed, languages = args.calibration_languages)
                    example_prompts = example_prompts.to(args.device)
                    logger.log("Start Backwarding in iterative steps = {}...".format(i))
                    loss = model(example_prompts, labels=example_prompts).loss
                    logger.log("Loss = {}".format(loss))
                    loss.backward()

                pruner.step()

                after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

            # Clean the gradient in the model
            model.zero_grad()
            for name, module in model.named_parameters():
                if 'weight' in name:
                    module.grad = None

            # modify inferece-related attributes
            model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
            model.zero_grad()
            
            del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:

        model.half()
        # Save in HuggingFace format for compatibility with lm-evaluation-harness
        hf_save_path = os.path.join(logger.log_dir, "hf_model")
        os.makedirs(hf_save_path, exist_ok=True)
        model.save_pretrained(hf_save_path)
        tokenizer.save_pretrained(hf_save_path)
        print(f"Saved HF-format model to {hf_save_path}")
        # Also keep the original torch.save for backward compatibility
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)


    if args.eval_device != "cpu":
        model.half()
    model.to(args.device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        ppl_save_path = os.path.join(logger.log_dir, "ppl_results.json")
        
        if os.path.exists(ppl_save_path) and args.force_recompute == False:
            print(f"Found existing PPL results at {ppl_save_path}, loading...")
            with open(ppl_save_path, "r") as f:
                ppl_results = json.load(f)
            for lang, ppl in ppl_results.items():
                print(f"ppl on {args.dataset} ({lang}): {ppl}")
            print("Skipping PPL computation after pruning.")
            logger.log(f"Loaded PPL results after pruning from {ppl_save_path}")
            return
        else:
            print("No existing PPL results found or force_recompute is True, computing PPL after pruning...")
            logger.log("\n==================Generation Results After Pruning================\n")
            
            model.eval()
            with torch.no_grad():
                for prompt in prompts:
                    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                    generation_output = model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        top_k=50,
                        max_length=args.max_seqlen,
                        top_p=args.top_p,
                        temperature=args.temperature,
                    )
                    
                    result = tokenizer.decode(generation_output[0])
                    logger.log(result)
            
            logger.log("\n==================Finish================\n")

            logger.log("Computing PPL evaluation after pruning...")

            if args.multilingual_eval:
                print(f"Evaluating on multiple languages: {args.eval_languages}")
                ppl_results = eval_ppl(model, tokenizer, device = args.device, dataset=args.dataset, languages=args.eval_languages)
                for lang, ppl in ppl_results.items():
                    print(f"ppl on {args.dataset} ({lang}): {ppl}")
                    logger.log(f"ppl on {args.dataset} ({lang}): {ppl}")
            else:
                ppl_results = eval_ppl(model, tokenizer, device = args.device, dataset=args.dataset, languages=[args.language])
                print(f"ppl on {args.dataset} ({args.language}): {ppl_results}")
                logger.log(f"ppl on {args.dataset} ({args.language}): {ppl_results}")

            with open(ppl_save_path, "w") as f:
                json.dump(ppl_results, f, indent=2, cls=LoadDatetime)
            print(f"Saved PPL results to {ppl_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="meta-llama/Meta-Llama-3-8B", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="Llama-3.2-1B_llmpruner_Test", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.7, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='taylor', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seqlen', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=0)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=32)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=0)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=32)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=100)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for computing importance')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Eval batch size for computing importance')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--test_before_train_then_end_exp', action='store_true', help='continue')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    parser.add_argument('--calibration_languages', type=str, nargs='+', default=["ar", "iw", "cs", "ru", "de", "en", "es", "id", "zh"], help='Languages to use for calibration data (multi-language calibration). If not specified, uses --language for single language calibration.')
    parser.add_argument('--multilingual_num_examples', type=int, nargs='+', default=None, help='Language num for multi-language calibration. Should match the number of calibration languages.')
    parser.add_argument('--multilingual_eval', action="store_true", help="Evaluate on multiple languages")
    parser.add_argument('--eval_languages', type=str, nargs='+', default=["ar", "iw", "cs", "ru", "de", "en", "es", "id", "zh"], help='Languages for evaluation')
    parser.add_argument('--language', type=str, default='en', help='Single language for evaluation (when --multilingual_eval is not set)')

    parser.add_argument('--seqlen', type=int, default=128, help='Sequence length')
    parser.add_argument('--dataset', type=str, default="c4", help='Dataset for evaluation and calibration')

    parser.add_argument('--save_group_json', type=bool, default=False, help='whether save the group results during the pruning')
    parser.add_argument('--save_topk_results', type=bool, default=False, help='whether save the topk results during the pruning')
    parser.add_argument('--multi_lang_important', type = bool, default = False, help='for multi-language important merge')
    parser.add_argument('--force_recompute', type=bool, default=False, help='force to recompute the PPL')

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='the cuda devices to use')

    # importance merging
    parser.add_argument('--merge_methods', type=str, default='max', help='mean/max/min/sum/weighted/topk_imp')
    parser.add_argument('--merge_weights', type=list, default=None, help='importance merge weight for weighted sum')
    parser.add_argument('--topk_imp', type=int, default=None, help='k for topk importance merge method')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    print(args.eval_languages)
    main(args)
    
