import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from datasets import load_dataset,load_from_disk

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        
def get_mmlu(nsamples, seed, seqlen, task, tokenizer):
    from .mmlu_data import mmlu_multi_task_data
    train_path = f"./data/mmlu/{mmlu_multi_task_data[task]['train']}"
    val_path = f"./data/mmlu/{mmlu_multi_task_data[task]['validation']}"
    cache_dir = '.cache'

    # Parquet files need to specify format="parquet"
    traindata = load_dataset("parquet", data_files={"train": train_path}, split="train", cache_dir=cache_dir)
    valdata = load_dataset("parquet", data_files={"validation": val_path}, split="validation", cache_dir=cache_dir)

    def make_prompt(example):
        q = example['question']
        choices = example['choices']
        choices_str = '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"Q:{q}\nChoices:{choices_str}\nA:"
        return prompt

    # Limit nsamples to not exceed dataset size
    nsamples = min(len(traindata), nsamples)
    print("training data length:", len(traindata))

    # Build fixed prompts without random sampling
    trainloader = []
    for idx in range(nsamples):
        example = traindata[idx]
        prompt = make_prompt(example)
        trainenc = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=seqlen
        )
        trainloader.append(trainenc.input_ids)

    trainloader = torch.cat(trainloader, dim=0)

    # Process validation set
    val_examples = valdata.select(range(min(1100, len(valdata))))
    prompts = [make_prompt(ex) + " " + str(ex["answer"]) for ex in val_examples]

    valenc = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seqlen
    )
    valenc = valenc.input_ids
    valenc = TokenizerWrapper(valenc)


    return trainloader, valenc


def get_c4(nsamples, seed, seqlen, language, tokenizer):
    from .languages import c4_multilingual_data
    # from multilingual_data import c4_multilingual_data
    train_path = f"./data/c4/{c4_multilingual_data[language]['train']}"
    val_path = f"./data/c4/{c4_multilingual_data[language]['validation']}"
    cache_dir = '.cache'
    # train_path = c4_multilingual_data[language]['train']
    # val_path = c4_multilingual_data[language]['validation']
    # cache_dir = '.cache'
    traindata = load_dataset("json", data_files={"train": train_path}, split="train", cache_dir=cache_dir)
    valdata = load_dataset("json", data_files={"validation": val_path}, split="validation", cache_dir=cache_dir)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        # tar = inp.clone()
        # tar[:, :-1] = -100
        trainloader.append(inp)

    trainloader = torch.cat(trainloader, dim=0)

    # Prepare validation dataset — shuffle with seed so different seeds yield different samples
    random.seed(seed + 1000)
    val_indices = list(range(len(valdata)))
    random.shuffle(val_indices)
    val_indices = val_indices[:1100]
    valenc = tokenizer(' '.join([valdata[i]['text'] for i in val_indices]), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_bookcorpus(tokenizer, nsamples, seqlen, seed=0):
    random.seed(seed)
    traindata = load_dataset('bookcorpus', split='train')

    tokenized_samples, history = [], []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seqlen and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seqlen)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seqlen])

    # Build a val set from a separate random draw (different from train indices)
    val_texts = []
    for _ in range(200):
        i = random.randint(0, len(traindata) - 1)
        val_texts.append(traindata[i]['text'])
    valenc = tokenizer(' '.join(val_texts), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return torch.cat(tokenized_samples, dim=0), valenc


def get_wikipedia(nsamples, seed, seqlen, language, tokenizer):
    data = load_dataset("wikimedia/wikipedia", f"20231101.{language}",
                        split="train")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]['text'], return_tensors='pt')
            if enc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        trainloader.append(enc.input_ids[:, i:i+seqlen])

    # Val set: separate random slice to avoid overlap with train
    random.seed(seed + 1000)
    val_texts = [data[random.randint(0, len(data) - 1)]['text'] for _ in range(200)]
    valenc = tokenizer(' '.join(val_texts), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return torch.cat(trainloader, dim=0), valenc


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        trainloader.append(trainenc.input_ids[:, i:i+seqlen])

    testenc = tokenizer(' '.join(testdata['text']), return_tensors='pt')
    testenc = testenc.input_ids[:, :(256 * seqlen)]
    testenc = TokenizerWrapper(testenc)

    return torch.cat(trainloader, dim=0), testenc

def get_multi_task_mmlu(tokenizer, nsamples, seqlen, seed, tasks, dataset_type="mmlu"):
    from .mmlu_data import mmlu_multi_task_data

    samples_per_task = nsamples // len(tasks)
    remaining_samples = nsamples % len(tasks)

    trainloader = []
    random.seed(seed)

    def make_prompt(example):
        q = example["question"]
        choices = example["choices"]
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"Q: {q}\nChoices:\n{choices_str}\nA:"
        return prompt

    for task_idx, task in enumerate(tasks):
        task_samples = samples_per_task + (1 if task_idx < remaining_samples else 0)

        assert task in mmlu_multi_task_data, f"Task [{task}] not supported in MMLU"
        train_path = f"./data/mmlu/{mmlu_multi_task_data[task]['train']}"
        traindata = load_dataset("parquet", data_files={"train": train_path}, split="train", cache_dir=".cache")

        # Sampling
        for i in range(min(task_samples, len(traindata))):
            ex = traindata[i]
            prompt = make_prompt(ex)
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seqlen
            )
            trainloader.append(enc.input_ids)

    # Concatenate all training samples
    trainloader = torch.cat(trainloader, dim=0)

    # Validation set (using first task only)
    first_task = tasks[0]
    val_path = f"./data/mmlu/{mmlu_multi_task_data[first_task]['validation']}"
    valdata = load_dataset("parquet", data_files={"validation": val_path}, split="validation", cache_dir=".cache")
    val_examples = valdata.select(range(min(1100, len(valdata))))
    prompts = [make_prompt(ex) + " " + str(ex["answer"]) for ex in val_examples]

    valenc = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seqlen
    )
    valenc = valenc.input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
    
def get_multilingual_calibration(tokenizer, nsamples, seqlen, seed, languages, dataset_type, multilingual_num_examples):
    
    # from .languages import c4_multilingual_data
    from .languages import c4_multilingual_data

    if multilingual_num_examples is not None:
        assert len(multilingual_num_examples) == len(languages), "Length of multilingual_num_examples must match length of languages"
        nsamples = sum(multilingual_num_examples)

        print("#"*10 , "total data:" , nsamples , "#"*10)

        trainloader = []
        random.seed(seed)

        for lang_idx, language in enumerate(languages):

            lang_samples = multilingual_num_examples[lang_idx]
            
            if dataset_type == "c4":
                assert language in c4_multilingual_data, f"Language [{language}] is not supported in C4"
                train_path = c4_multilingual_data[language]['train']
                val_path = c4_multilingual_data[language]['validation']
                cache_dir = '.cache'
                traindata = load_dataset('./data/c4', data_files={'train': train_path}, split='train', cache_dir=cache_dir)

            else:
                raise ValueError(f"Dataset type {dataset_type} not supported")
            
            for _ in range(lang_samples):
                while True:
                    i = random.randint(0, len(traindata) - 1)
                    if dataset_type == "c4":
                        text = traindata[i]['text']
                    else:
                        text = traindata[i]['text']
                    trainenc = tokenizer(text, return_tensors='pt')
                    if trainenc.input_ids.shape[1] > seqlen:
                        break
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                # tar = inp.clone()
                # tar[:, :-1] = -100
                # why we need to do this? what is tar?
                trainloader.append(inp)
                # testloader.append(tar)

    else:    
        samples_per_language = nsamples // len(languages)
        remaining_samples = nsamples % len(languages)
        
        trainloader = []
        # testloader = []
        random.seed(seed)

        for lang_idx, language in enumerate(languages):

            lang_samples = samples_per_language + (1 if lang_idx < remaining_samples else 0)
            
            if dataset_type == "c4":
                assert language in c4_multilingual_data, f"Language [{language}] is not supported in C4"
                train_path = c4_multilingual_data[language]['train']
                val_path = c4_multilingual_data[language]['validation']
                cache_dir = '.cache'
                traindata = load_dataset('./data/c4', data_files={'train': train_path}, split='train', cache_dir=cache_dir)

            else:
                raise ValueError(f"Dataset type {dataset_type} not supported")
            
            for _ in range(lang_samples):
                while True:
                    i = random.randint(0, len(traindata) - 1)
                    if dataset_type == "c4":
                        text = traindata[i]['text']
                    else:
                        text = traindata[i]['text']
                    trainenc = tokenizer(text, return_tensors='pt')
                    if trainenc.input_ids.shape[1] > seqlen:
                        break
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                # tar = inp.clone()
                # tar[:, :-1] = -100
                # why we need to do this? what is tar?
                trainloader.append(inp)
                # testloader.append(tar)

    trainloader = torch.cat(trainloader, dim=0)
    # testloader = torch.cat(testloader, dim=0)


    # Prepare validation dataset using the first language
    first_language = languages[0]
    if dataset_type == "c4":
        val_path = c4_multilingual_data[first_language]['validation']
        valdata = load_dataset('./data/c4', data_files={'validation': val_path}, split='validation', cache_dir=cache_dir)
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    else:
        # For other dataset types, use a subset of training data for validation
        val_texts = [traindata[i]['text'] for i in range(min(1000, len(traindata)))]
        valenc = tokenizer(' '.join(val_texts), return_tensors='pt')
    
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    
    return trainloader, valenc

def get_examples(dataset, tokenizer, nsamples, seqlen = 128, seed=42, languages=None, language = None, multilingual_num_examples=None):
    # if dataset == 'c4' and languages is None:
    #     return get_c4(tokenizer, nsamples, seqlen)
    
    if dataset == 'c4' and languages is None and language is not None:
        return get_c4(nsamples, seed, seqlen, language, tokenizer)

    ### Logic has some problems, need to fix ###
    elif dataset == 'c4' and languages is not None:
        return get_multilingual_calibration(tokenizer, nsamples, seqlen, seed, languages, dataset_type=dataset, multilingual_num_examples=multilingual_num_examples)

    elif dataset == 'mmlu' and languages is None and language is not None:
        return get_mmlu(nsamples, seed, seqlen, language, tokenizer)
    
    elif dataset == 'mmlu' and languages is not None:
        return get_multi_task_mmlu(tokenizer, nsamples, seqlen, seed, languages, dataset_type=dataset)
        
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, nsamples, seqlen)
    else:
        raise NotImplementedError

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, language='en', tokenizer=None, model_family=None, languages=None):
    # If multiple languages are provided, use multilingual calibration
    if languages and len(languages) > 1:
        return get_multilingual_calibration(tokenizer, nsamples, seed, seqlen, languages, dataset_type=name)
    
    # Single language loading (original behavior)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, language, tokenizer)
    if "oscar" in name:
        raise NotImplementedError("OSCAR dataset loading is not yet implemented.")
    if "pile" in name:
        raise NotImplementedError("Pile dataset loading is not yet implemented.")
    if "wikipedia" in name:
        return get_wikipedia(nsamples, seed, seqlen, language, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if "bookcorpus" in name:
        return get_bookcorpus(tokenizer, nsamples, seqlen, seed)

    # Default to C4 if no specific dataset is specified
    return get_c4(nsamples, seed, seqlen, language, tokenizer)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/aya-expanse-8b", trust_remote_code=True)
    samples = get_examples('c4', tokenizer, nsamples = 10, seqlen=128, languages = ["ar", "iw", "cs", "ru", "de", "en", "es", "id" ,"zh"])
    # print(samples.shape)