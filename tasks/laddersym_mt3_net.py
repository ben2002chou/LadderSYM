from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.laddersym_t5 import T5ForConditionalGeneration
from utils import get_cosine_schedule_with_warmup


class laddersym_MT3Net(pl.LightningModule):

    def __init__(self, config, optim_cfg):
        super().__init__()
        self.config = config
        self.optim_cfg = optim_cfg
        self.use_prompt = config.use_prompt
        print(f"Use prompt: {self.use_prompt}")
        
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5ForConditionalGeneration(T5config, use_prompt=self.use_prompt)
        self.num_steps_per_epoch = None  # Initialize as None
        

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Unpack batch based on use_prompt flag
        if self.use_prompt:
            mistake_inputs, score_inputs, targets, prompt, prompt_attention_mask = batch
        else:
            mistake_inputs, score_inputs, targets = batch
            prompt = None
            prompt_attention_mask = None

        # Forward pass
        lm_logits = self.forward(
            mistake_inputs=mistake_inputs,
            score_inputs=score_inputs,
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_attention_mask,
            labels=targets
        )

        # Handle prompt slicing if present
        if self.use_prompt:
            prompt_length = prompt.size(1)
            lm_logits = lm_logits[:, prompt_length:, :]

        # Flatten logits and targets
        lm_logits_flat = lm_logits.reshape(-1, lm_logits.size(-1))
        targets_flat = targets.view(-1)

        # Compute the raw loss
        loss_fct_raw = nn.CrossEntropyLoss(reduction="none")
        loss_unmasked = loss_fct_raw(lm_logits_flat, targets_flat)

        # Masks for instrument tokens and padding
        instrument_tokens_mask = (targets_flat >= 1135) & (targets_flat <= 1136)
        pad_mask = targets_flat != -100  # Excludes padding tokens

        # Debugging: Check targets and masks
        print(f"Batch {batch_idx} - Targets Flat: {targets_flat}")
        print(f"Batch {batch_idx} - Instrument Tokens Mask Sum: {instrument_tokens_mask.sum().item()}")
        print(f"Batch {batch_idx} - Pad Mask Sum: {pad_mask.sum().item()}")

        # Compute losses
        loss_instruments = torch.masked_select(loss_unmasked, instrument_tokens_mask)
        loss_masked = torch.masked_select(loss_unmasked, pad_mask & ~instrument_tokens_mask)

        # Handle cases where certain token types are absent
        if loss_instruments.numel() == 0:
            print(f"Batch {batch_idx} - No instrument tokens found. Setting loss_instruments to 0.")
            loss_instruments_sum = 0.0
        else:
            loss_instruments_sum = loss_instruments.sum()

        if loss_masked.numel() == 0:
            print(f"Batch {batch_idx} - No other tokens found. Setting loss_masked to 0.")
            loss_masked_sum = 0.0
        else:
            loss_masked_sum = loss_masked.sum()

        # Reward model for correctly predicting nothing if both types of tokens are absent
        if (loss_masked.numel() + loss_instruments.numel()) == 0:
            print(f"Batch {batch_idx} - No valid tokens found in targets. Rewarding model.")
            total_loss = torch.tensor(0.0, requires_grad=True)
        else:
            total_loss = (
                loss_masked_sum + self.optim_cfg.error_loss_weight * loss_instruments_sum
            ) / (loss_masked.numel() + loss_instruments.numel())

        # Debugging: Ensure loss is not NaN
        if torch.isnan(total_loss):
            print(f"Batch {batch_idx} - NaN Loss Detected! Debugging:")
            print(f"  Targets Flat: {targets_flat}")
            print(f"  Loss Unmasked: {loss_unmasked}")
            print(f"  Loss Instruments: {loss_instruments}")
            print(f"  Loss Masked: {loss_masked}")
            print(f"  Instrument Tokens Mask: {instrument_tokens_mask}")
            print(f"  Pad Mask: {pad_mask}")
            print(f"  Loss Masked Sum: {loss_masked_sum}")
            print(f"  Loss Instruments Sum: {loss_instruments_sum}")

        # Logging
        self.log('train_loss_other', loss_masked.mean() if loss_masked.numel() > 0 else 0.0, 
                prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_loss_inst', loss_instruments.mean() if loss_instruments.numel() > 0 else 0.0, 
                prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        return total_loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Unpack batch based on use_prompt flag
        if self.use_prompt:
            mistake_inputs, score_inputs, targets, prompt, prompt_attention_mask = batch
        else:
            mistake_inputs, score_inputs, targets = batch
            prompt = None
            prompt_attention_mask = None

        # Debugging: Print shapes of inputs
        # print(f"Validation Batch index: {batch_idx}")
        # print(f"mistake_inputs shape: {mistake_inputs.shape}")
        # print(f"score_inputs shape: {score_inputs.shape}")
        # print(f"prompt shape: {prompt.shape}")
        # print(f"prompt_attention_mask shape: {prompt_attention_mask.shape}")
        # print(f"targets shape: {targets.shape}")


        lm_logits = self.forward(
            mistake_inputs=mistake_inputs,
            score_inputs=score_inputs,
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_attention_mask,
            labels=targets
            )

        # Debugging: Print shape of lm_logits
        # print(f"lm_logits shape (before slicing): {lm_logits.shape}")

        # Get lengths of prompt and targets
        batch_size = targets.size(0)
        if self.use_prompt:
            prompt_length = prompt.size(1)
            target_length = targets.size(1)

            # Exclude the prompt outputs from lm_logits
            lm_logits = lm_logits[:, prompt_length:, :]
        else:
            target_length = targets.size(1)

        # # Debugging: Print shape after slicing lm_logits
        # print(f"lm_logits shape (after slicing): {lm_logits.shape}")
        # print(f"Expected target_length: {target_length}")

        # Ensure lm_logits and targets are aligned
        assert lm_logits.size(1) == target_length, "Mismatch in lengths of logits and targets after excluding prompt"

        # Flatten tensors for loss computation
        lm_logits_flat = lm_logits.reshape(-1, lm_logits.size(-1))
        targets_flat = targets.reshape(-1)

        # # Debugging: Print shapes after flattening
        # print(f"lm_logits_flat shape: {lm_logits_flat.shape}")
        # print(f"targets_flat shape: {targets_flat.shape}")

        # # Debugging: Check for invalid target indices
        # print(f"targets_flat min value: {targets_flat.min().item()}, max value: {targets_flat.max().item()}")
        # vocab_size = lm_logits_flat.size(-1)
        # print(f"Vocabulary size: {vocab_size}")

        # if targets_flat.min() < 0 or targets_flat.max() >= vocab_size:
        #     print("Error: targets_flat contains invalid indices.")
        #     print(f"Invalid indices found in targets_flat at positions: {(targets_flat >= vocab_size).nonzero(as_tuple=True)[0]}")

        if targets is not None:
            loss_fct_raw = nn.CrossEntropyLoss(reduction="none")

            # Adjusted indices for instrument tokens (ensure consistency)
            # Assuming instrument tokens are in the range [1135, 1136]
            instrument_tokens_mask = (targets_flat >= 1135) & (targets_flat <= 1136)
            pad_mask = targets_flat != -100

            # Debugging: Print number of instrument tokens and padding tokens
            # print(f"Number of instrument tokens: {instrument_tokens_mask.sum().item()}")
            # print(f"Number of non-padding tokens: {pad_mask.sum().item()}")

            # Compute raw loss
            loss_unmasked = loss_fct_raw(lm_logits_flat, targets_flat)

            # Apply masks
            loss_instruments = torch.masked_select(loss_unmasked, instrument_tokens_mask)
            loss_masked = torch.masked_select(loss_unmasked, pad_mask & ~instrument_tokens_mask)

            # Compute total loss
            total_loss = (
                loss_masked.sum() + self.optim_cfg.error_loss_weight * loss_instruments.sum()
            ) / (loss_masked.numel() + loss_instruments.numel())

        # Logging
        self.log('val_loss_other', loss_masked.mean(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        val_loss_inst = loss_instruments.mean() if loss_instruments.numel() > 0 else 0.0
        self.log('val_loss_inst', val_loss_inst, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


    def configure_optimizers(self):
        if self.num_steps_per_epoch is None:
            raise RuntimeError("num_steps_per_epoch has not been set. Ensure it is passed before optimizer configuration.")

        optimizer = AdamW(self.model.parameters(), self.optim_cfg.lr)
        warmup_step = int(self.optim_cfg.warmup_steps)
        print("Warmup step: ", warmup_step)

        schedule = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=self.num_steps_per_epoch * self.optim_cfg.num_epochs,
                min_lr=self.optim_cfg.min_lr,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [schedule] 
