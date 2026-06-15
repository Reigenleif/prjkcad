import os
from typing import Union, Dict, Any
import yaml
import random

import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.set_seed import set_seed
from utils.dataset import Text2CADLoader
from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.data_utils import create_cmdonly_data_loader, create_dualseq_data_loader
from utils.wrapper import DualSeqCMDOnlyWrapper
from utils.wrapper import DualSeqWrapper
from utils.criterion import DualSeqCMDOnlyCriterion
from utils.criterion import DualSeqCriterion
from utils.trainer import DualSeqCMDOnlyTrainer
from utils.trainer import DualSeqTrainer

from models.t5_t5_t5 import T5T5T5
from models.t5_t5_cmdonly import T5T5Cmdonly
from models.t5_torch_cmdonly import T5TorchCmdonly
from models.t5_torch_torch import T5TorchTorch


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
        
        if config["data"]["sample_ratio"] :
            sample_size = int(len(dual_seqs) * config["data"]["sample_ratio"])
            dual_seqs = random.sample(dual_seqs, sample_size)
            print(f"Sampled {sample_size} instances from the dataset based on the specified sample ratio of {config['data']['sample_ratio']}.")
        
        if USE_VAL :
            if config["data"]["is_cmdonly"] :
                train_loader, val_loader = create_cmdonly_data_loader(dual_seqs,
                                                                      text_tokenizer,
                                                                      description_level=config["data"]["description_level"],
                                                                      batch_size=config["data"]["batch_size"],
                                                                      num_workers=config["data"]["num_workers"],
                                                                      val_ratio=config["data"]["eval_split_ratio"],
                                                                        shuffle=True)
            else :
                train_loader, val_loader = create_dualseq_data_loader(dual_seqs,
                                                               text_tokenizer,
                                                               description_level=config["data"]["description_level"],
                                                               batch_size=config["data"]["batch_size"],
                                                               num_workers=config["data"]["num_workers"],
                                                               val_ratio=config["data"]["eval_split_ratio"],
                                                               shuffle=True)
        else :
            if config["data"]["is_cmdonly"] :
                train_loader = create_cmdonly_data_loader(dual_seqs,
                                                         text_tokenizer,
                                                         description_level=config["data"]["description_level"],
                                                         batch_size=config["data"]["batch_size"],
                                                         num_workers=config["data"]["num_workers"],
                                                         val_ratio=0.0,
                                                         shuffle=True)
            else :
                train_loader = create_dualseq_data_loader(dual_seqs,
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
        if config["model"]["cls"] == "T5T5T5" :
            model_cls = T5T5T5
        elif config["model"]["cls"] == "T5T5Cmdonly" :
            model_cls = T5T5Cmdonly
        elif config["model"]["cls"] == "T5TorchCmdonly" :
            model_cls = T5TorchCmdonly
        elif config["model"]["cls"] == "T5TorchTorch" :
            model_cls = T5TorchTorch
        else :
            raise ValueError(f"Unsupported model class: {config['model']['cls']}")
    elif config["model"]["source"] == "huggingface" :
        raise NotImplementedError("Directly loading pretrained model from HuggingFace is not implemented yet. Please set model source to 'local' and load the pretrained model using the wrapper's from_pretrained method.")
        # model_cls = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name"])
    else :
        raise ValueError(f"Unsupported model source: {config['model']['source']}")
    
    
    if config["data"]["is_cmdonly"] :
        model_kwargs = {"vocab_size": get_dualseq_schema()["n_tokens"], 
                        **config["model"]["kwargs"]}
    else :
        model_kwargs = {"vocab_size": get_dualseq_schema()["n_tokens"], 
                        "n_args": get_dualseq_schema()["n_args"], 
                        **config["model"]["kwargs"]}
        
    
    if not config["model"]["is_pretrained"] :
        # If the model is not pretrained, initialize the model, then the wrapper
        model = model_cls(**model_kwargs)
        # Init wrapper
        if config["data"]["is_cmdonly"] :
            wrapper = DualSeqCMDOnlyWrapper(model, text_tokenizer)
        elif not config["data"]["is_cmdonly"] :
            wrapper = DualSeqWrapper(model, text_tokenizer)
        
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
            wrapper = DualSeqWrapper.from_pretrained(
                model_cls,
                config["pretrained_path"],
                seq2seq_model_name=config["tokenizer"]["model_name"],
                model_kwargs=model_kwargs
            )
        
    # Criterion loading
    if config["trainer"]["criterion"]["source"] == "local" :
        if config["trainer"]["criterion"]["cls"] == "DualSeqCMDOOnlyCriterion" :
            criterion_cls = DualSeqCMDOnlyCriterion
        elif config["trainer"]["criterion"]["cls"] == "DualSeqCriterion" :
            criterion_cls = DualSeqCriterion
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
            save_folder=SAVE_ROOT,
            **config["trainer"]["kwargs"]
        )
    else :
        trainer = DualSeqTrainer(
            wrapper,
            criterion,
            optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            save_folder=SAVE_ROOT,
            **config["trainer"]["kwargs"]
        )  
    
    progression = trainer.train(config["trainer"]["epochs"], verbose=verbose_all)
    
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

def plot_progression(config, progression: list[dict[str, float]]):
    if isinstance(config, str):
        config = load_config(config)
    out_path = f"out/{config['run_name']}/progression.png"
    
    viz_keys = {
        "loss" : {
            "train": [h["train_loss"] for h in progression],
            "val": [h["val_loss"] for h in progression],
        },
        "val_perplexity" : {
            "val": [h["val_perplexity"] for h in progression],
        },
        "val_performance" :{
            "LINE_precision": [h.get("val_LINE_precision", 0) for h in progression],
            "LINE_recall": [h.get("val_LINE_recall", 0) for h in progression],
            "LINE_f1": [h.get("val_LINE_f1", 0) for h in progression],
            "CIRCLE_precision": [h.get("val_CIRCLE_precision", 0) for h in progression],
            "CIRCLE_recall": [h.get("val_CIRCLE_recall", 0) for h in progression],
            "CIRCLE_f1": [h.get("val_CIRCLE_f1", 0) for h in progression],
            "ARC_precision": [h.get("val_ARC_precision", 0) for h in progression],
            "ARC_recall": [h.get("val_ARC_recall", 0) for h in progression],
            "ARC_f1": [h.get("val_ARC_f1", 0) for h in progression],
            "EXTRUDE_accuracy": [h.get("val_EXTRUDE_accuracy", 0) for h in progression],
        }
    }

    if not config["data"]["is_cmdonly"] :
        viz_keys["val_performance"] = {
            **viz_keys["val_performance"],
            "val_arg_mape": [h.get("val_arg_mape", 0) for h in progression],
            "val_arg_r2": [h.get("val_arg_r2", 0) for h in progression],   
        }

    fig, ax = plt.subplots(3, 1, 
                            figsize=(10, 8), 
                            gridspec_kw={"height_ratios": [1, 1, 2]})

    ax[0].plot(viz_keys["loss"]["train"], label="Train Loss")
    ax[0].plot(viz_keys["loss"]["val"], label="Val Loss")
    ax[0].set_title("Loss")

    ax[1].plot(viz_keys["val_perplexity"]["val"], label="Val Perplexity")
    ax[1].set_title("Validation Perplexity")

    ax[2].plot(viz_keys["val_performance"]["LINE_precision"], label="LINE Precision")
    ax[2].plot(viz_keys["val_performance"]["LINE_recall"], label="LINE Recall")
    ax[2].plot(viz_keys["val_performance"]["LINE_f1"], label="LINE F1")
    ax[2].plot(viz_keys["val_performance"]["CIRCLE_precision"], label="CIRCLE Precision")
    ax[2].plot(viz_keys["val_performance"]["CIRCLE_recall"], label="CIRCLE Recall")
    ax[2].plot(viz_keys["val_performance"]["CIRCLE_f1"], label="CIRCLE F1")
    ax[2].plot(viz_keys["val_performance"]["ARC_precision"], label="ARC Precision")
    ax[2].plot(viz_keys["val_performance"]["ARC_recall"], label="ARC Recall")
    ax[2].plot(viz_keys["val_performance"]["ARC_f1"], label="ARC F1")
    ax[2].plot(viz_keys["val_performance"]["EXTRUDE_accuracy"], label="EXTRUDE Accuracy")
    if not config["data"]["is_cmdonly"] :
        ax[2].plot(viz_keys["val_performance"]["val_arg_r2"], label="Val Arg R2")

    ax[2].set_title("Validation Performance Summary")
    ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    

    plt.tight_layout()
    plt.show()
    
    if out_path is not None:
        fig.savefig(out_path)
        
