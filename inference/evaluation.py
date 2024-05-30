import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def read_all_data(args):
    with open(args.input_path) as f_input:
        lines = f_input.readlines()
        all_data = []
        for cur_line in lines:
            cur_dict = json.loads(cur_line.strip())
            all_data.append(cur_dict)
    return all_data


class Embedding_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
        self.model = AutoModel.from_pretrained(args.embedding_model_path).cuda()
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text):
        with torch.no_grad():
            tokenization_result = self.tokenizer(text, padding="longest", truncation=True,
                                                 return_tensors='pt',
                                                 max_length=32)
            input_ids = tokenization_result["input_ids"].to('cuda')
            attention_mask = tokenization_result["attention_mask"].to('cuda')
            model_output = self.model(input_ids, attention_mask)
            sentence_embeddings = self.mean_pooling(model_output, attention_mask)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def batch_forward(self, input_text):
        total_batch_num = int(len(input_text) / self.args.batch_size) + 1
        total_emb = []
        for i in range(total_batch_num):
            cur_batch_input = input_text[i * self.args.batch_size:(i + 1) * self.args.batch_size]
            if len(cur_batch_input) == 0:
                continue
            emb = self.forward(cur_batch_input)
            total_emb.append(emb)
        total_emb = torch.cat(total_emb, dim=0)
        return total_emb

    def item_retrieval(self, input_vec, index):
        with torch.no_grad():
            similarity_score = torch.matmul(input_vec, index)
            score, indice = torch.topk(similarity_score, 1, dim=-1, largest=True)

            scores = score.squeeze(-1).cpu().tolist()
            indices = indice.squeeze(-1).cpu().tolist()

        return (scores, indices)

    def relevance_calculation(self, all_data):
        hit_query = 0
        total_query = 0
        for cur_data in tqdm(all_data):
            total_user_behavior = []
            search_query = cur_data["search_query"]
            clicked_web_title = cur_data["clicked_web_title"]
            clicked_news_title = cur_data["clicked_news_title"]
            edge_clicks = cur_data["edge_clicks"]

            for cur_behavior in [search_query, clicked_web_title, clicked_news_title, edge_clicks]:
                if len(cur_behavior) == 0:
                    continue
                for cur_item in cur_behavior:
                    total_user_behavior.append(cur_item.strip())
            total_user_behavior = list(set(total_user_behavior))
            if len(total_user_behavior) == 0:
                continue
            cur_index = torch.transpose(self.batch_forward(total_user_behavior), 0, 1)
            query_list = cur_data["result"]
            if len(query_list) <= 1:
                print("skip")
                continue
            cur_query_emb = self.batch_forward(query_list)
            scores, indices = self.item_retrieval(cur_query_emb, cur_index)
            for cur_query, cur_score, cur_indice in zip(query_list, scores, indices):
                total_query += 1
                if cur_score >= 0.3:
                    hit_query += 1
        avg_relevance_score = hit_query / total_query
        return round(avg_relevance_score, 3)

    def interest_coverage_calculation(self, all_data):
        total_behavior = 0
        hit_behavior = 0
        for cur_data in tqdm(all_data):
            query_list = cur_data["result"]
            if len(query_list) == 0:
                continue
            cur_query_index = torch.transpose(self.batch_forward(query_list), 0, 1)

            total_user_behavior = []
            search_query = cur_data["search_query"]
            clicked_web_title = cur_data["clicked_web_title"]
            clicked_news_title = cur_data["clicked_news_title"]
            edge_clicks = cur_data["edge_clicks"]

            for cur_behavior in [search_query, clicked_web_title, clicked_news_title, edge_clicks]:
                if len(cur_behavior) == 0:
                    continue
                for cur_item in cur_behavior:
                    if len(cur_item.strip().split()) <= 3:
                        continue
                    total_user_behavior.append(cur_item.strip())
            total_user_behavior = list(set(total_user_behavior))
            if len(total_user_behavior) == 0:
                continue

            cur_behavior_emb = self.batch_forward(total_user_behavior)
            scores, indices = self.item_retrieval(cur_behavior_emb, cur_query_index)

            for cur_behavior, cur_score, cur_indice in zip(total_user_behavior, scores, indices):
                total_behavior += 1
                if cur_score >= 0.3:
                    hit_behavior += 1
        avg_coverage_rate = hit_behavior / total_behavior
        return round(avg_coverage_rate, 3)


class Query_Model(nn.Module):
    def __init__(self, quality_model_path, dim, head):
        super().__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(quality_model_path)
        self.score_head = nn.Linear(dim, head, bias=True)
        self.init_weights(self.score_head)

    def mean_pooling(self, sequence_output, attention_mask):
        token_embeddings = sequence_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        sequence_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(sequence_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        with torch.autocast(device_type="cuda"):
            predicted_score = self.score_head(sentence_embeddings).squeeze()
        return predicted_score


class Query_Inference():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.quality_model_path)
        self.quality_model = self.build_quality_model(args)
        self.ctr_model = self.build_ctr_model(args)

    def build_quality_model(self, args):
        model = Query_Model(self.args.quality_model_path, dim=768, head=4)
        state_dict = torch.load(os.path.join(args.quality_model_path, "checkpoint.bin"))
        missing_keys, extra_keys = model.load_state_dict(state_dict, strict=False)
        print("missing_keys")
        print(missing_keys)
        print("extra_keys")
        print(extra_keys)

        model = model.eval().cuda()
        return model

    def build_ctr_model(self, args):
        model = Query_Model(self.args.ctr_model_path, dim=768, head=2)
        state_dict = torch.load(os.path.join(args.ctr_model_path, "checkpoint.bin"))
        missing_keys, extra_keys = model.load_state_dict(state_dict, strict=False)
        print("missing_keys")
        print(missing_keys)
        print("extra_keys")
        print(extra_keys)

        model = model.eval().cuda()
        return model

    def large_scale_inference(self, all_data, task):
        total_score = 0
        total_query = 0
        all_query = []
        with torch.no_grad():
            for cur_data in tqdm(all_data):
                cur_input = cur_data["result"]
                if len(cur_input) <= 1:
                    continue
                tokenizer_result = self.tokenizer(cur_input, padding="max_length",
                                                  max_length=32,
                                                  truncation=True, return_tensors='pt')
                input_ids = tokenizer_result["input_ids"].cuda()
                attention_mask = tokenizer_result["attention_mask"].cuda()
                if task == "quality":
                    scores = self.quality_model.forward(input_ids, attention_mask)
                else:
                    scores = self.ctr_model.forward(input_ids, attention_mask)
                result = torch.nn.functional.softmax(scores, dim=-1).cpu().numpy().tolist()

                for cur_query, score in zip(cur_input, result):
                    total_query += 1
                    cur_query_score = sum([idx * item for idx, item in enumerate(score)])
                    total_score += cur_query_score
                    all_query.append((cur_query, cur_query_score))
            all_query_sorted = sorted(all_query, key=lambda x:x[1], reverse=True)
            print(all_query_sorted[:10])
            print(all_query_sorted[-10:])
            avg_quality_score = total_score / total_query
        return round(avg_quality_score, 3)

class Query_Retrieval(object):
    def __init__(self, args):
        self.args = args
        self.model = Embedding_Model(args).cuda()
        self.build_index()
        self.index = torch.transpose(self.id2emb, 0, 1).cuda()

    def build_index(self):
        if os.path.exists(self.args.id2emb_index_path) and os.path.exists(self.args.id2text_index_path):
            self.id2text = torch.load(self.args.id2text_index_path)
            self.id2emb = torch.load(self.args.id2emb_index_path).cuda()
        else:
            with open(self.args.input_path) as f_read:
                total_data = f_read.readlines()
                total_num = len(total_data)
            self.id2text = {}
            self.id2emb = torch.zeros((total_num, self.args.hidden_size)).cuda()
            total_batch_num = int(total_num / self.args.batch_size) + 1
            idx = 0
            for batch_id in tqdm(range(total_batch_num)):
                cur_batch = total_data[batch_id * self.args.batch_size:(batch_id + 1) * self.args.batch_size]
                if len(cur_batch) == 0:
                    continue
                infos = [line.strip().split("\t") for line in cur_batch if len(line.strip().split("\t")) == 2]
                cur_doc_text = [cur_infos[1].strip() for cur_infos in infos]
                doc_embs = self.model.forward(cur_doc_text)
                for cur_infos, cur_doc_emb in zip(infos, doc_embs):
                    self.id2text[idx] = (cur_infos[0].strip(), cur_infos[1].strip())
                    self.id2emb[idx, :] = cur_doc_emb
                    idx += 1
            torch.save(self.id2text, self.args.id2text_index_path)
            torch.save(self.id2emb, self.args.id2emb_index_path)

    def item_retrieval(self, input_vec, return_nums):
        with torch.no_grad():
            similarity_score = torch.matmul(input_vec, self.index)
            score, indice = torch.topk(similarity_score, return_nums, dim=-1, largest=True)

            scores = score.squeeze().cpu().tolist()
            indices = indice.squeeze().cpu().tolist()

        return (scores, indices)

    def do_item_inference(self, all_data):
        total_score = 0
        total_query = 0
        all_query = []
        with torch.no_grad():
            for cur_data in tqdm(all_data):
                cur_inputs = cur_data["result"]
                if len(cur_inputs) <= 1:
                    continue
                doc_embs = self.model.forward(cur_inputs)
                scores, indices = self.item_retrieval(doc_embs, 5)
                for cur_input, cur_score in zip(cur_inputs, scores):
                    avg_score = np.mean(cur_score)
                    all_query.append((cur_input, avg_score))
                    total_score += avg_score
                    total_query += 1
        all_query_sorted = sorted(all_query, key=lambda x:x[1], reverse=True)
        print(all_query_sorted[:20])
        print(all_query_sorted[-20:])
        return round(total_score/total_query, 3)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model_path", type=str, default="./mpnet_emb_model/")
    parser.add_argument("--quality_model_path", type=str, default="./query_quality_model//")
    parser.add_argument("--ctr_model_path", type=str, default="./query_ctr_model/")
    parser.add_argument("--id2emb_index_path", type=str, default="./ann_index/mpnet_id2emb_index_path.bin")
    parser.add_argument("--id2text_index_path", type=str, default="./ann_index/mpnet_id2text_index_path.bin")
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    all_data = read_all_data(args)

    ## load sentence embedding model
    emb_model = Embedding_Model(args)
    print("embedding model successful loaded")

    ## load query quality model
    query_model = Query_Inference(args)
    print("query model successful loaded")

    ## load query retrieval model
    retrieval_model = Query_Retrieval(args)
    print("query retrieval model successful loaded")

    ## query retrieval calculation
    avg_retrieval_score = retrieval_model.do_item_inference(all_data)
    print("average query retrieval score: " + str(avg_retrieval_score))

    ## quality calculation
    avg_quality_score = query_model.large_scale_inference(all_data, "quality")
    print("average query quality score: " + str(avg_quality_score))

    ## ctr calculation
    avg_ctr_score = query_model.large_scale_inference(all_data, "ctr")
    print("average query ctr score: " + str(avg_ctr_score))

    ## user interest coverage calculation
    avg_interest_coverage_score = emb_model.interest_coverage_calculation(all_data)
    print("average interest coverage score: " + str(avg_interest_coverage_score))

    ## relevance calculation
    avg_relevance_score = emb_model.relevance_calculation(all_data)
    print("average relevance score: " + str(avg_relevance_score))