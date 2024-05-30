import os
import torch
from transformers import (
    AutoConfig,
    AutoModel
)
from dschat.model.reward_model import RewardModel
from dschat.utils import load_state_dict_into_model, print_rank_0


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def causal_lm_model_to_fp32_loss(model):
    """ Convert CausalLM model to calculate loss in fp32 """

    def causal_lm_forward(
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **deprecated_arguments,
    ):
        output = model.__original_forward__(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        return_dict = isinstance(output, dict)
        lm_logits = output.logits if return_dict else output[0]
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length))

        if not return_dict:
            # re-pack output with fp32 loss
            return ((loss,) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = causal_lm_forward


def create_critic_model(model_name_or_path,
                        tokenizer,
                        rlhf_training=False,
                        zero_stage=0,
                        compute_fp32_loss=False):
    import time

    start = time.time()
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    critic_model = AutoModel.from_config(model_config)
    end = time.time()
    print_rank_0(f">Creating model from_config took {end - start} seconds",
                 None)

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        compute_fp32_loss=compute_fp32_loss)

    if rlhf_training:
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        end = time.time()
        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)
        start = time.time()
        load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)
        end = time.time()

        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)

    return critic_model
