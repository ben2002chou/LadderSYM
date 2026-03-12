from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers import T5Config, T5PreTrainedModel
from torch.utils.checkpoint import checkpoint
from transformers.models.t5.modeling_t5 import (
    Seq2SeqLMOutput,
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    T5LayerNorm,
    T5Block,
)
from transformers.utils import logging
import torch.nn as nn
import copy
import torch
from einops import rearrange
from tqdm import tqdm
from .vit import ViTEncoder

logger = logging.get_logger(__name__)


@dataclass
class Seq2SeqLMOutputNumInsts(Seq2SeqLMOutput):
    loss_inst: Optional[torch.FloatTensor] = None


class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config, use_prompt):
        super().__init__(config)
        self.model_dim = config.d_model
        # NOTE: temporary change, for MT3 please uncomment this line
        self.proj = nn.Linear(self.model_dim, self.model_dim, bias=False)
        print('using prompt')
        # NOTE: for encodec model please uncomment this line
        # self.proj = nn.Embedding(
        #     config.encoder_vocab_size, config.d_model)
        # print('config.vocab_size', config.vocab_size)
        self.decoder_embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.proj, "encoder")
        self.encoder = ViTEncoder()
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.decoder_embed_tokens, "decoder", use_prompt= use_prompt)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.mean_pool = nn.AdaptiveAvgPool1d(1)
        # self.num_inst_cls = nn.Linear(config.d_model, 16)

        # Initialize weights and apply final processing
        self.post_init()
        self.use_prompt = use_prompt

    def get_input_embeddings(self):
        return self.decoder_embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.decoder_embed_tokens = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_model_outputs(
        self,
        mistake_inputs: Optional[torch.FloatTensor] = None,
        score_inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        # Input projection is not necessary for the LadderSym encoder.
        # if inputs is not None:
        #     inputs_embeds = self.proj(inputs)
        # print('inputs_embeds', inputs_embeds[0][0][:20])
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            # encoder_outputs = self.encoder(
            #     input_ids=None,
            #     attention_mask=attention_mask,
            #     inputs_embeds=inputs_embeds,
            #     head_mask=head_mask,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            encoder_outputs = self.encoder(
                x=mistake_inputs,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs  # [0]
        # check if decoder_input_ids is provided
        # print('decoder_input_ids', decoder_input_ids, flush=True)
        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        print("initial_prompt_length before decoding", decoder_attention_mask.sum(dim=1).tolist())
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,  
            initial_prompt_lengths = decoder_attention_mask.sum(dim=1).tolist()  # List of prompt lengths per batch element,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        # mean_hidden_states = self.mean_pool(sequence_output.transpose(1, 2)).squeeze(-1)
        # inst_cls_logits = self.num_inst_cls(mean_hidden_states)

        return lm_logits, encoder_outputs, decoder_outputs

    def forward(
        self,
        mistake_inputs: Optional[torch.FloatTensor] = None,
        score_inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_insts: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )


        lm_logits, encoder_outputs, decoder_outputs = self.get_model_outputs(
            mistake_inputs=mistake_inputs,
            score_inputs=score_inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # for ddp bug
        all_params = torch.sum(torch.stack([torch.sum(p) for p in self.parameters()]))
        lm_logits = lm_logits + all_params * 0

        return lm_logits

    # TODO: Need to be updated to handle prompt (look at default T5 generate)
    def generate(
        self,
        mistake_inputs,
        score_inputs,
        decoder_input_ids=None,  # Prompt tokens
        decoder_attention_mask=None,  # Attention mask for prompt
        max_length=1024,
        output_hidden_states=False,
        **kwargs,
    ):
        batch_size = mistake_inputs.shape[0]

        # 1. Encode the inputs (encoder output acts as context)
        encoder_outputs = self.encoder(
            x=mistake_inputs,
        )
        hidden_states = encoder_outputs

        # 2. Determine prompt length and total length
        if self.use_prompt and decoder_input_ids is not None:
            prompt_length = decoder_input_ids.size(1)


            # Batch-wise initial prompt lengths
            initial_prompt_lengths = decoder_attention_mask.sum(dim=1).tolist()
        else:
            prompt_length = 1  # We'll insert the start token if no prompt is given.

        total_length = prompt_length + max_length

        # 3. Pre-allocate tensors for generated IDs and attention mask
        generated_ids = torch.full(
            (batch_size, total_length),
            self.config.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        attention_mask_full = torch.zeros(
            (batch_size, total_length),
            dtype=torch.long,
            device=self.device
        )

        # 4. Initialize the first tokens
        if self.use_prompt and decoder_input_ids is not None:
            # Place the prompt tokens at the start
            generated_ids[:, :prompt_length] = decoder_input_ids
            attention_mask_full[:, :prompt_length] = decoder_attention_mask
            # print(f"Generated IDs with prompt: {generated_ids}", flush=True)
            # print(f"Attention mask with prompt: {attention_mask_full}", flush=True)
            # print(f"shape of generated_ids: {generated_ids.shape}", flush=True)
            # print(f"shape of attention_mask_full: {attention_mask_full.shape}", flush=True)
        else:
            # If no prompt is given, start with decoder_start_token_id
            generated_ids[:, 0] = self.config.decoder_start_token_id
            attention_mask_full[:, 0] = 1

        # 5. Prepare for decoding
        # Initially, we decode from the already set tokens (prompt + start)
        decoder_input_ids_start = generated_ids[:, :prompt_length]
        if self.use_prompt:
            decoder_attention_mask_start = attention_mask_full[:, :prompt_length]

        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=self.device
        )

        # 6. Generate tokens step by step
        for l in range(max_length):
            # Decode step
            if self.use_prompt:
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids_start,
                    attention_mask=decoder_attention_mask_start,
                    encoder_hidden_states=hidden_states,
                    return_dict=True,
                    initial_prompt_lengths=initial_prompt_lengths,
                )
            else:
                # print(f"decoder_input_ids_start: {decoder_input_ids_start}", flush=True)
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids_start,
                    encoder_hidden_states=hidden_states,
                    return_dict=True,
                )
                

            # Compute logits for the next token
            sequence_output = decoder_outputs[0]
            lm_logits = self.lm_head(sequence_output)

            # Get the next tokens
            next_tokens = torch.argmax(lm_logits[:, -1, :].unsqueeze(1), dim=-1)

            # Update unfinished sequences
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(
                -1
            ) + self.config.pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
            
            # Mark the newly generated token in pre-allocated arrays
            generated_ids[:, prompt_length + l] = next_tokens
            attention_mask_full[:, prompt_length + l] = (next_tokens != self.config.pad_token_id).long()


            # Check for EOS
            eos_indices = torch.where(next_tokens == self.config.eos_token_id)[0]
            unfinished_sequences[eos_indices] = 0

            # Update decoder inputs to include the newly generated token
            # Instead of concatenating, we just redefine the "view" of decoder_input_ids_start
            decoder_input_ids_start = generated_ids[:, :prompt_length + l + 1]
            if self.use_prompt:
                decoder_attention_mask_start = attention_mask_full[:, :prompt_length + l + 1]

            # Stop if all sequences finished
            if unfinished_sequences.max() == 0:
                break
        # print(f"Generated output shape: {decoder_input_ids_start.shape}", flush=True)
        # print(f"prompt_length: {prompt_length}", flush=True)

        # 7. Remove the prompt part from the result
        if self.use_prompt:
            # sos token is removed later
            generated_output = decoder_input_ids_start[:, prompt_length - 1:]
        else:
            generated_output = decoder_input_ids_start
        # # print(f"Generated output shape (after slicing): {generated_output.shape}", flush=True)
        # print(f"Generated output: {generated_output}", flush=True)
        if output_hidden_states:
            return generated_output, decoder_outputs.hidden_states
        else:
            return generated_output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, name="", use_prompt=True):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.use_prompt = use_prompt

        self.pos_emb = FixedPositionalEmbedding(config.d_model)

        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=False)
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        self.gradient_checkpointing = False

        self.name = name

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,  # Added labels for training
        initial_prompt_lengths=None,  # Added for prompt handling
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Debugging: Input shapes and configurations
        # print(f"Forward call with use_cache={use_cache}, return_dict={return_dict}", flush=True)
        # if input_ids is not None:
        #     # print(f"Original input_ids shape: {input_ids.shape}", flush=True)
        # if attention_mask is not None:
        #     # print(f"Original attention_mask shape: {attention_mask.shape}", flush=True)
        # if labels is not None:
        #     # print(f"Original labels shape: {labels.shape}", flush=True)
            
        # Ensure input_ids and attention_mask are duplicated if necessary
        # batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        # # Adjust dynamically if required
        # if input_ids is not None:
        #     if input_ids.dim() == 1:  # If input_ids is 1D, reshape and duplicate
        #         input_ids = input_ids.unsqueeze(0).expand(batch_size, -1)
        #     elif input_ids.size(0) == 1:  # Duplicate single-batch input_ids
        #         input_ids = input_ids.expand(batch_size, -1)
        #     # print(f"Duplicated input_ids shape: {input_ids.shape}", flush=True)

        # if attention_mask is not None:
        #     if attention_mask.dim() == 1:  # If attention_mask is 1D, reshape and duplicate
        #         attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)
        #     elif attention_mask.size(0) == 1:  # Duplicate single-batch attention_mask
        #         attention_mask = attention_mask.expand(batch_size, -1)
        #     # print(f"Duplicated attention_mask shape: {attention_mask.shape}", flush=True)

        
        # Handle prompt concatenation with labels
        if self.use_prompt and labels is not None:
            # sos token is added here
            shifted_labels = self._shift_right(labels)
            # print(f"Shifted labels shape: {shifted_labels.shape}", flush=True)
            if input_ids is not None:

                pad_token_id = self.config.pad_token_id

                if pad_token_id is None:
                    raise ValueError("self.model.config.pad_token_id has to be defined.")
                # replace possible -100 values in labels by `pad_token_id`
                input_ids.masked_fill_(input_ids == -100, pad_token_id)


                input_ids = torch.cat([input_ids, shifted_labels], dim=1)
                # print(f"Concatenated input_ids shape: {input_ids.shape}", flush=True)
            else:
                input_ids = shifted_labels
                # print(f"Input_ids from shifted labels shape: {input_ids.shape}", flush=True)

            # Adjust the attention mask if provided
            # TODO: why is this attention mask beign altered?
            # print(f"Original attention_mask shape: {attention_mask.shape}", flush=True)
            if attention_mask is not None:
                prompt_length = attention_mask.size(1)
                labels_length = labels.size(1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(labels.size(0), labels_length).to(attention_mask.device),
                    ],
                    dim=1,
                )
                # print(f"Adjusted attention_mask shape: {attention_mask.shape}", flush=True)
        else:
            if input_ids is not None:
                pass
                # print(f"Input_ids shape: {input_ids.shape}", flush=True)
            if attention_mask is not None:
                prompt_length = attention_mask.size(1)
                # print(f"Prompt length: {prompt_length}", flush=True)
        
        
        # Input validation and shape extraction
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_ids is not None:
            input_shape = input_ids.size()[:2]
        else:
            raise ValueError("input_ids or inputs_embeds must be provided.")

        # print(f"Input shape: {input_shape}", flush=True)
        # print(f"Max input_id: {input_ids.max()}", flush=True)
        # print(f"Min input_id: {input_ids.min()}", flush=True)
        # Embedding input_ids if inputs_embeds is None
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # print(f"Generated inputs_embeds shape: {inputs_embeds.shape}", flush=True)

        batch_size, seq_length = input_shape[:2]
        # print(f"Batch size: {batch_size}, Sequence length: {seq_length}", flush=True)

        # Required mask sequence length
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        # print(f"Mask sequence length: {mask_seq_length}, Past key-values length: {past_key_values_length}", flush=True)

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        # Set attention mask if None
        if attention_mask is None:
            print('attention_mask is None')
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device
            )
            # print(f"Generated default attention_mask with shape: {attention_mask.shape}", flush=True)

        # Encoder attention mask setup
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            # print(f"Generated encoder_attention_mask with shape: {encoder_attention_mask.shape}", flush=True)

        # Initialize past_key_values with `None` if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
            # print(f"Initialized past_key_values with length: {len(past_key_values)}", flush=True)

        # Debugging to verify the device and shapes

        # Call the function after ensuring both tensors are on the same device
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, input_ids.device
        )


        # Step 2: Modify the extended attention mask for prompt-to-prompt self-attention
        if self.use_prompt:


            # Ensure there are prompt tokens in at least one batch element
            if any(length > 0 for length in initial_prompt_lengths):
                # print(f"Batch-wise initial prompt lengths: {initial_prompt_lengths}")

                # Iterate over the batch to handle each prompt length
                for batch_idx, initial_prompt_length in enumerate(initial_prompt_lengths):
                    if initial_prompt_length > 0:  # Only modify if the prompt length is non-zero
                        # Create a prompt-to-prompt self-attention submatrix
                        prompt_self_attention = torch.zeros(
                            (1, 1, initial_prompt_length, initial_prompt_length),  # Shape: [1, 1, P, P]
                            device=extended_attention_mask.device,
                            dtype=extended_attention_mask.dtype,
                        )

                        # Overwrite the top-left corner of the extended attention mask for the current batch element
                        extended_attention_mask[batch_idx, :, :initial_prompt_length, :initial_prompt_length] = prompt_self_attention

        # Encoder extended attention mask setup
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
            # print(f"Encoder extended attention mask shape: {encoder_extended_attention_mask.shape}", flush=True)
        else:
            encoder_extended_attention_mask = None

        # Prepare head masks
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # Positional embeddings
        tmp = self.pos_emb(seq=inputs_embeds.shape[1], offset=past_key_values_length)
        # 5120 (4096+1024), 5000
        inputs_embeds = inputs_embeds + tmp
        # print(f"Positional embeddings added, inputs_embeds shape: {inputs_embeds.shape}", flush=True)

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            # print(f"Processing layer {i + 1}/{len(self.block)}")

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # print(f"Attention Mask: {attention_mask}")
        # print(f"Extended Attention Mask: {extended_attention_mask}")

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_length=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_length = max_length

    def forward(self, seq, offset=0):
        t = torch.arange(self.max_length, device=self.inv_freq.device).type_as(
            self.inv_freq
        )
        sinusoid_inp = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        y = rearrange(emb, "n d -> () n d")
        y = y[:, offset : offset + seq, :]
        return y
