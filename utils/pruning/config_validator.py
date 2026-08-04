from utils.pipeline.config import ModelConfig


# Available decoders for pruning search
VALID_ENCODERS = {"t5-small", "bert"}
VALID_DECODERS = {"sdpa", "t5-small", "mamba"}
VALID_MOE_CONFS = {"Switch", "Mixtral"}
VALID_ADAPTIVE_LAYER_TYPES = {"none", "linear", "ffn_head", "sdpa"}
VALID_EMBEDDING_TYPES = {"standard", "rope", "sdpa", "rope_sdpa"}


def is_valid_config(cfg: ModelConfig) -> bool:
    """
    Returns True if the given ModelConfig represents a valid, trainable configuration.

    This includes:
    - Dimension compatibility between encoder and d_model
    - Decoder availability
    - MoE consistency
    - New kwargs (adaptive_layer, embedding types, dropout)
    """
    d_model = getattr(cfg, "d_model", 512) or 512

    # Encoder
    enc = cfg.encoder_type
    if enc not in VALID_ENCODERS:
        return False
    if enc == "t5-small" and d_model != 512:
        return False

    # cmd decoder
    cmd_dec = cfg.cmd_decoder_type
    if cmd_dec not in VALID_DECODERS:
        return False
    if cmd_dec == "t5-small" and d_model != 512:
        return False

    # args decoder
    is_cmd_only = cfg.is_cmd_only
    args_dec = cfg.args_decoder_type

    if is_cmd_only:
        if args_dec is not None:
            return False
    else:
        if args_dec is None:
            return False
        if args_dec not in VALID_DECODERS:
            return False
        if args_dec == "t5-small" and d_model != 512:
            return False

    # MoE
    moe_conf = getattr(cfg, "moe_conf", None)

    # If either is set, both must be set and valid
    if moe_conf is not None:
        if moe_conf not in VALID_MOE_CONFS:
            return False
        # T5 decoders with custom MoE: T5 internal MoE is handled separately;
        # for simplicity during pruning, disallow moe_conf on t5-* decoders
        if cmd_dec.startswith("t5-"):
            return False
        if not is_cmd_only and args_dec.startswith("t5-"):
            return False
            
    adaptive_layer = getattr(cfg, "adaptive_layer", "none")
    if adaptive_layer not in VALID_ADAPTIVE_LAYER_TYPES:
        return False

    cmd_emb_type = getattr(cfg, "cmd_embedding_type", "standard")
    if cmd_emb_type not in VALID_EMBEDDING_TYPES:
        return False

    args_emb_type = getattr(cfg, "args_embedding_type", "standard")
    if args_emb_type not in VALID_EMBEDDING_TYPES:
        return False

    use_drop_out = getattr(cfg, "use_drop_out", True)
    drop_out_p = getattr(cfg, "drop_out_p", 0.1)
    if not isinstance(use_drop_out, bool):
        return False
    if use_drop_out and not (0.0 < drop_out_p < 1.0):
        return False

    return True
