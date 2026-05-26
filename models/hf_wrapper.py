from transformers import PreTrainedModel, Trainer

class CADSeq2SeqWrapper(PreTrainedModel):
    def __init__(self, config, base_model, criterion):
        super().__init__(config)
        self.model = base_model
        self.criterion = criterion

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,          # cmd_targets
        param_targets=None,   # custom
    ):
        # ---- Forward your model ----
        cmd_logits, param_preds = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tgt_seq=labels[:, :-1],  # teacher forcing
        )

        loss = None
        if labels is not None:
            cmd_targets = labels[:, 1:]

            loss = self.criterion(
                cmd_logits,
                param_preds,
                cmd_targets,
                param_targets
            )

        return {
            "loss": loss,
            "logits": cmd_logits,
        }
        

class CADTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        param_targets = inputs.pop("param_targets")

        outputs = model(
            **inputs,
            labels=labels,
            param_targets=param_targets
        )

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss