import argparse
import os
import torch
import json
import numpy as np
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer
)
from deepspeed.accelerator import get_accelerator


class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.v_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def forward_pointwise_rewards(self, input_ids):
        transformer_outputs = self.rwtranrsformer(input_ids)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_end_scores = []
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]
            c_inds = (input_id == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            chosen_end_scores.append(value[c_ind - 1])
        return torch.stack(chosen_end_scores)


def create_critic_model(model_name_or_path,
                        tokenizer,
                        disable_dropout=False):
    model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if disable_dropout:
        model_config.dropout = 0.0
    rm_model = AutoModel.from_pretrained(model_name_or_path,
                                         trust_remote_code=True)
    critic_model = RewardModel(rm_model, tokenizer).half()
    critic_model.load_state_dict(torch.load(model_ckpt_path, map_location='cpu'), strict=False)

    return critic_model


def load_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    model = create_critic_model(model_name_or_path, tokenizer, True)
    return model, tokenizer


def prepare_batch_input(prompt, cur_query_list, tokenizer, max_seq_len=4096):
    all_input_ids = []
    for idx, item in enumerate(cur_query_list):
        input = prompt + item
        input_ids = tokenizer(input)["input_ids"] + [tokenizer.eos_token_id]
        if len(input_ids) <= max_seq_len:
            number_token_to_padding = max_seq_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * number_token_to_padding
            all_input_ids.append(input_ids)
    return torch.tensor(np.array(all_input_ids)).to(get_accelerator().current_device_name())


def point_wise_batch_inference(filtered_dataset):
    args = parse_args()
    rm_model, tokenizer = load_model(args.path)
    rm_model.to(get_accelerator().current_device_name())
    rm_model.eval()
    with torch.no_grad():
        with open(args.data_output, "a") as f_output:
            total_items = filtered_dataset[args.start_idx:args.end_idx]
            for cur_item in total_items:
                prompt = cur_item["input"]
                cur_data = {"prompt": prompt, "candidates": []}
                output_candidates = cur_item["output"]
                for each_candidate in output_candidates:
                    cur_query_list = each_candidate.strip().split("\n")
                    batch_input = prepare_batch_input(prompt, cur_query_list, tokenizer)
                    outputs = rm_model.forward_pointwise_rewards(batch_input).cpu().tolist()
                    cur_candidate_score_list = []
                    for cur_query, cur_score in zip(cur_query_list, outputs):
                        cur_candidate_score_list.append((cur_query.strip(), round(cur_score, 4)))
                    cur_candidate_score_list_sorted = sorted(cur_candidate_score_list, key=lambda x: x[1], reverse=True)
                    cur_data["candidates"].append(cur_candidate_score_list_sorted)
                f_output.write(json.dumps(cur_data, ensure_ascii=False) + "\n")


def filter_data_by_length(input_data):
    all_data = []
    error_num = 0
    with open(input_data) as f_read:
        for item in f_read:
            infos = json.loads(item.strip())
            input = infos["input"]
            output_candidates_filter = []
            for item in infos["output_candidates"]:
                if len(item.strip().split("\n")) == 10:
                    output_candidates_filter.append(item)
                else:
                    error_num += 1
            all_data.append({"input": input, "output": output_candidates_filter})
    print(error_num)
    print(len(all_data))
    return all_data


def parse_args():
    parser = argparse.ArgumentParser(description="run reward model inference")
    parser.add_argument("--path", type=str)
    parser.add_argument("--data_input", type=str)
    parser.add_argument("--data_output", type=str)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10000000000)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    filtered_dataset = filter_data_by_length(args.data_input)
    point_wise_batch_inference(filtered_dataset)
