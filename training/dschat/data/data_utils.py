import torch
import os
import random
from torch.utils.data import Dataset, Subset, ConcatDataset
from dschat.data import raw_datasets
from deepspeed.accelerator import get_accelerator


class PromptDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len, train_phase) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train_phase = train_phase
        self.print = 0

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == "sft":
            prompt_input, prompt_output = self.dataset[idx]
            if self.print == 0:
                print(prompt_input)
                print(prompt_output)
                self.print = 1
            prompt_input_ids = self.tokenizer(prompt_input)["input_ids"]
            prompt_output_ids = self.tokenizer(prompt_output)["input_ids"]
            origin_input_ids = (prompt_input_ids + prompt_output_ids) + [self.tokenizer.eos_token_id]
            while len(origin_input_ids) > self.max_seq_len:
                idx = random.randint(0, len(self.dataset) - 1)
                prompt_input, prompt_output = self.dataset[idx]
                prompt_input_ids = self.tokenizer(prompt_input)["input_ids"]
                prompt_output_ids = self.tokenizer(prompt_output)["input_ids"]
                if self.tokenizer.add_bos_token:
                    prompt_output_ids = prompt_output_ids[1:]
                origin_input_ids = (prompt_input_ids + prompt_output_ids) + [self.tokenizer.eos_token_id]
            prompt_input_length = len(prompt_input_ids)
            total_length = len(origin_input_ids)
            labels = [-100 if i <= prompt_input_length - 1 else origin_input_ids[i] for i in
                      range(total_length)]
            number_token_to_padding = self.max_seq_len - len(origin_input_ids)
            input_ids = origin_input_ids + [self.tokenizer.pad_token_id] * number_token_to_padding
            labels = labels + [-100] * number_token_to_padding

            return {"input_ids": torch.tensor(input_ids).squeeze(),
                    "labels": torch.tensor(labels).squeeze()}

        elif self.train_phase == "reward":
            chosen_input, rejected_input = self.dataset[idx]
            prompt_chosen_ids = self.tokenizer(chosen_input)["input_ids"] + [self.tokenizer.eos_token_id]
            prompt_rejected_ids = self.tokenizer(rejected_input)["input_ids"] + [self.tokenizer.eos_token_id]

            while len(prompt_chosen_ids) > self.max_seq_len and len(prompt_rejected_ids) > self.max_seq_len:
                idx = random.randint(0, len(self.dataset) - 1)
                chosen_input, rejected_input = self.dataset[idx]
                prompt_chosen_ids = self.tokenizer(chosen_input)["input_ids"] + [self.tokenizer.eos_token_id]
                prompt_rejected_ids = self.tokenizer(rejected_input)["input_ids"] + [self.tokenizer.eos_token_id]

            number_token_to_padding = self.max_seq_len - len(prompt_chosen_ids)
            input_ids = prompt_chosen_ids + [self.tokenizer.pad_token_id] * number_token_to_padding
            chosen_token = {"input_ids": torch.tensor(input_ids).squeeze()}

            number_token_to_padding = self.max_seq_len - len(prompt_rejected_ids)
            input_ids = prompt_rejected_ids + [self.tokenizer.pad_token_id] * number_token_to_padding
            reject_token = {"input_ids": torch.tensor(input_ids).squeeze()}

            return chosen_token["input_ids"], reject_token["input_ids"]

        elif self.train_phase == "dpo" or self.train_phase == "spo":
            prompt, chosen_input, rejected_input = self.dataset[idx]
            prompt_ids = self.tokenizer(prompt)["input_ids"]
            chosen_ids = self.tokenizer(chosen_input)["input_ids"]
            rejected_ids = self.tokenizer(rejected_input)["input_ids"]

            prompt_chosen_ids = prompt_ids + chosen_ids + [self.tokenizer.eos_token_id]
            prompt_rejected_ids = prompt_ids + rejected_ids + [self.tokenizer.eos_token_id]
            prompt_length = len(prompt_ids)

            while len(prompt_chosen_ids) > self.max_seq_len or len(prompt_rejected_ids) > self.max_seq_len:
                idx = random.randint(0, len(self.dataset) - 1)
                prompt, chosen_input, rejected_input = self.dataset[idx]
                prompt_ids = self.tokenizer(prompt)["input_ids"]
                chosen_ids = self.tokenizer(chosen_input)["input_ids"]
                rejected_ids = self.tokenizer(rejected_input)["input_ids"]

                prompt_chosen_ids = prompt_ids + chosen_ids + [self.tokenizer.eos_token_id]
                prompt_rejected_ids = prompt_ids + rejected_ids + [self.tokenizer.eos_token_id]
                prompt_length = len(prompt_ids)

            total_prompt_chosen_length = len(prompt_chosen_ids)
            mask_loss_chosen = [0 if i <= prompt_length - 1 else 1 for i in range(total_prompt_chosen_length)]
            number_token_to_padding = self.max_seq_len - len(prompt_chosen_ids)
            choose_input_ids = torch.tensor(
                prompt_chosen_ids + [self.tokenizer.pad_token_id] * number_token_to_padding).squeeze()
            choose_mask_loss_chosen = torch.tensor(mask_loss_chosen + [0] * number_token_to_padding).squeeze()

            total_prompt_rejected_length = len(prompt_rejected_ids)
            mask_loss_rejected = [0 if i <= prompt_length - 1 else 1 for i in range(total_prompt_rejected_length)]
            number_token_to_padding = self.max_seq_len - len(prompt_rejected_ids)
            rejected_input_ids = torch.tensor(
                prompt_rejected_ids + [self.tokenizer.pad_token_id] * number_token_to_padding).squeeze()
            rejected_mask_loss_chosen = torch.tensor(mask_loss_rejected + [0] * number_token_to_padding).squeeze()

            return choose_input_ids, choose_mask_loss_chosen, rejected_input_ids, rejected_mask_loss_chosen


def create_dataset_split(current_dataset, tokenizer, max_seq_len, train_phase):
    dataset = []
    if train_phase == "dpo" or train_phase == "spo":
        for i, cur_data in enumerate(current_dataset):
            prompt, chosen, rejected = cur_data[0], cur_data[1], cur_data[2]
            if prompt is not None and chosen is not None and rejected is not None:
                dataset.append((prompt, chosen, rejected))
            if i != 0 and i % 100000 == 0:
                print("we have processed " + str(i) + " dataset")
    else:
        for i, cur_data in enumerate(current_dataset):
            input, output = cur_data[0], cur_data[1]
            if input is not None and output is not None:
                dataset.append((input, output))
            if i != 0 and i % 100000 == 0:
                print("we have processed " + str(i) + " dataset")
    return PromptDataset(dataset, tokenizer, max_seq_len, train_phase)


def create_dataset(data_path,
                   data_output_path,
                   train_phase,
                   tokenizer,
                   max_seq_len,
                   reload=False):
    os.makedirs(data_output_path, exist_ok=True)
    train_fname = f"{data_output_path}/traindata.pt"
    eval_fname = f"{data_output_path}/evaldata.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)

    if torch.distributed.get_rank() <= 0 and (buf_create_cache.item() != 0 or reload):
        raw_dataset = raw_datasets.GPT4PersonalizationDataset(data_path, train_phase)

        train_dataset = raw_dataset.get_train_data()
        train_dataset = create_dataset_split(train_dataset, tokenizer, max_seq_len, train_phase)

        eval_dataset = raw_dataset.get_eval_data()
        eval_dataset = create_dataset_split(eval_dataset, tokenizer, max_seq_len, train_phase)

        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0].unsqueeze(0) for f in data] + [f[1].unsqueeze(0) for f in data], dim=0)
        return batch


class DataCollatorDPO:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0].unsqueeze(0) for f in data] + [f[2].unsqueeze(0) for f in data], dim=0)
        batch["mask_loss"] = torch.cat([f[1].unsqueeze(0) for f in data] + [f[3].unsqueeze(0) for f in data], dim=0)
        return batch
