import os
from typing import Union, Dict, Any
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.set_seed import set_seed
from utils.dataset import Text2CADLoader
from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.data_utils import create_cmdonly_data_loader
from utils.wrapper import DualSeqCMDOnlyWrapper
from utils.criterion import DualSeqCMDOnlyCriterion
from utils.trainer import DualSeqCMDOnlyTrainer

from models.t5_enc_t5_dec_cad import T5EncT5DecCAD


Config = dict[str, Union[str, "Config"]]

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config: Config | str,
                verbose_all: bool = False) :
    """
    End-to-end function to train the model from model and data loading, training loop, evaluation, and saving.
    config can be either a dict of json or a path to the config json file.
    """
    if isinstance(config, str):
        config = load_config(config)
    
    USE_VAL = config["data"]["eval_split_ratio"] > 0
    SAVE_ROOT = f"out/{config['run_name']}"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    CHECKPOINT_SAVE_PATH = f"{SAVE_ROOT}/checkpoint.pt"
    RENDER_RESULTS_PATH = f"{SAVE_ROOT}/render_results"
    TEST_RESULT_PATH = f"{SAVE_ROOT}/test_result.csv"
    
    
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Tokenizer loading
    if config["tokenizer"]["source"] == "huggingface" :
        text_tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["model_name"])
    else :
        raise ValueError(f"Unsupported tokenizer source: {config['tokenizer']['source']}")
    
    # Raw data loading
    if config["data"]["source_data_type"] == "text2cad" :
        loader = Text2CADLoader(config["data"]["data_root"],
                                max_samples=config["data"]["max_samples"] if "max_samples" in config["data"] else None)
        df = loader.load()
        
        dual_seqs = DualSeq.from_text2cad_df(df)
        
        if USE_VAL :
            train_loader, val_loader = create_cmdonly_data_loader(dual_seqs,
                                                                  text_tokenizer,
                                                                  description_level=config["data"]["description_level"],
                                                                    batch_size=config["data"]["batch_size"],
                                                                    num_workers=config["data"]["num_workers"],
                                                                    val_ratio=config["data"]["eval_split_ratio"],
                                                                    shuffle=True)
            
        else :
            train_loader = create_cmdonly_data_loader(dual_seqs,
                                                     text_tokenizer,
                                                     description_level=config["data"]["description_level"],
                                                     batch_size=config["data"]["batch_size"],
                                                     num_workers=config["data"]["num_workers"],
                                                     val_ratio=0.0,
                                                     shuffle=True)
            val_loader = None
                           
    else :
        raise ValueError(f"Unsupported data source type: {config['data']['source_data_type']}")
    
    # Wrapper and Model loading
    if config["model"]["source"] == "local" :
        if config["model"]["cls"] == "T5EncT5DecCAD" :
            model_cls = T5EncT5DecCAD
        else :
            raise ValueError(f"Unsupported model class: {config['model']['cls']}")
    elif config["model"]["source"] == "huggingface" :
        raise NotImplementedError("Directly loading pretrained model from HuggingFace is not implemented yet. Please set model source to 'local' and load the pretrained model using the wrapper's from_pretrained method.")
        # model_cls = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name"])
    else :
        raise ValueError(f"Unsupported model source: {config['model']['source']}")
    
    
    model_kwargs = {"vocab_size": get_dualseq_schema()["n_tokens"], 
                    **config["model"]["kwargs"]}
    
    if not config["model"]["is_pretrained"] :
        # If the model is not pretrained, initialize the model, then the wrapper
        model = model_cls(**model_kwargs)
        # Init wrapper
        if config["data"]["is_cmdonly"] :
            wrapper = DualSeqCMDOnlyWrapper(model, text_tokenizer)
        elif not config["data"]["is_cmdonly"] :
            raise NotImplementedError("Wrapper for non-cmdonly data is not implemented yet.")
        
    else :
        # If the model is pretrained, load the model from the specified path, directly using wrapper
        if config["data"]["is_cmdonly"] :
            wrapper = DualSeqCMDOnlyWrapper.from_pretrained(
                model_cls,
                config["pretrained_path"],
                seq2seq_model_name=config["tokenizer"]["model_name"],
                model_kwargs=model_kwargs
            )
        elif not config["data"]["is_cmdonly"] :
            raise NotImplementedError("Wrapper for non-cmdonly data is not implemented yet.")
        
    # Criterion loading
    if config["trainer"]["criterion"]["source"] == "local" :
        if config["trainer"]["criterion"]["cls"] == "DualSeqCMDOOnlyCriterion" :
            criterion_cls = DualSeqCMDOnlyCriterion
        else :
            raise ValueError(f"Unsupported criterion class: {config['trainer']['criterion']['cls']}")
    else :
        raise ValueError(f"Unsupported criterion source: {config['trainer']['criterion']['source']}")
    
    criterion = criterion_cls(**config["trainer"]["criterion"]["kwargs"])
    
    # Optimizer loading
    if config["trainer"]["optimizer"] == "AdamW" :
        optimizer_cls = torch.optim.AdamW
    else :
        raise ValueError(f"Unsupported optimizer: {config['trainer']['optimizer']}")
    optimizer = optimizer_cls(wrapper.parameters(), **config["trainer"]["optimizer_kwargs"])
    
    
    # Trainer initialization
    if config["data"]["is_cmdonly"] :
        trainer = DualSeqCMDOnlyTrainer(
            wrapper,
            criterion,
            optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            **config["trainer"]["kwargs"]
        )
    else :
        raise NotImplementedError("Trainer for non-cmdonly data is not implemented yet.")
    
    progression = trainer.train(config["trainer"]["epochs"], verbose=verbose_all)

    # save the model
    torch.save(wrapper.model.state_dict(), CHECKPOINT_SAVE_PATH)
    
    # inference test for ten random samples
    rand_idxs = torch.randperm(len(dual_seqs))[:10]
    results = [
        {
            "input": dual_seqs[i].descriptions[config["data"]["description_level"]],
            "target_cmds": dual_seqs[i].cmds,
            "generated_cmds": wrapper.generate(dual_seqs[i].descriptions[config["data"]["description_level"]], 
                                               max_new_tokens=config["trainer"]["max_new_cmds"])
        } for i in rand_idxs.tolist()
    ]
    # save to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(TEST_RESULT_PATH, index=False)
    
    return progression

def plot_progression(progression, out_path=None) :
    viz_keys = {
        "loss" : {
            "train": [h["train_loss"] for h in progression],
            "val": [h["val_loss"] for h in progression],
        },
        "val_perplexity" : {
            "val": [h["val_perplexity"] for h in progression],
        },
        "val_accuracy" : {
            "val": [h["val_accuracy"] for h in progression],
        },
    }

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    ax[0].plot(viz_keys["loss"]["train"], label="Train Loss")
    ax[0].plot(viz_keys["loss"]["val"], label="Val Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(viz_keys["val_perplexity"]["val"], label="Val Perplexity")
    ax[1].set_title("Validation Perplexity")
    ax[1].legend()

    ax[2].plot(viz_keys["val_accuracy"]["val"], label="Val Accuracy")
    ax[2].set_title("Validation Accuracy")
    ax[2].legend()

    plt.tight_layout()
    plt.show()
    
    if out_path is not None:
        fig.savefig(out_path)
        
