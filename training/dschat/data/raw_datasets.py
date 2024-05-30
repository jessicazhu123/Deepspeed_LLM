import os
import json


class GPT4PersonalizationDataset(object):
    def __init__(self, dataset_name, train_phase):
        self.train_data_path = os.path.join(dataset_name, "train_data")
        self.test_data_path = os.path.join(dataset_name, "test_data")
        self.train_dataset = []
        self.eval_dataset = []
        self.obtain_train_test_dataset(train_phase)

    def obtain_train_test_dataset(self, train_phase):
        with open(self.train_data_path, "r", encoding="utf-8", errors='ignore') as f_read:
            for item in f_read:
                infos = json.loads(item.strip())
                if train_phase == "sft":
                    infos = json.loads(item.strip())
                    input = infos["input"]
                    output = infos["output"]
                    self.train_dataset.append((input, output))
                elif train_phase == "reward":
                    chosen = infos["chosen"]
                    rejected = infos["rejected"]
                    self.train_dataset.append((chosen, rejected))
                elif train_phase == "dpo" or train_phase == "spo":
                    prompt = infos["prompt"]
                    chosen = infos["chosen"]
                    rejected = infos["rejected"]
                    self.train_dataset.append((prompt, chosen, rejected))

        with open(self.test_data_path, "r", encoding="utf-8", errors='ignore') as f_read:
            for item in f_read:
                infos = json.loads(item.strip())
                if train_phase == "sft":
                    input = infos["input"]
                    output = infos["output"]
                    self.eval_dataset.append((input, output))
                elif train_phase == "reward":
                    chosen = infos["chosen"]
                    rejected = infos["rejected"]
                    self.eval_dataset.append((chosen, rejected))
                elif train_phase == "dpo" or train_phase == "spo":
                    prompt = infos["prompt"]
                    chosen = infos["chosen"]
                    rejected = infos["rejected"]
                    self.eval_dataset.append((prompt, chosen, rejected))

    def get_train_data(self):
        return self.train_dataset

    def get_eval_data(self):
        return self.eval_dataset
