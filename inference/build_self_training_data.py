import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering


def read_all_data(args):
    with open(args.input_path) as f_input:
        lines = f_input.readlines()
        all_data = []
        for cur_line in lines:
            cur_dict = json.loads(cur_line.strip())
            if len(cur_dict["candidates"]) == 0:
                continue
            cur_dict["candidates"] = [[item] for item in cur_dict["candidates"]]
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

    def relevance_calculation(self, cur_data):
        keep_candidate = []
        drop_candidate = []

        total_user_behavior = []
        search_query = cur_data["search_query"]
        clicked_web_title = cur_data["clicked_web_title"]
        clicked_news_title = cur_data["clicked_news_title"]
        edge_clicks = cur_data["edge_clicks"]

        for cur_behavior in [search_query, clicked_web_title, clicked_news_title, edge_clicks]:
            for cur_item in cur_behavior:
                if cur_item.strip() != "":
                    total_user_behavior.append(cur_item.strip())
        total_user_behavior = list(set(total_user_behavior))
        if len(total_user_behavior) == 0:
            return None, None
        cur_index = torch.transpose(self.batch_forward(total_user_behavior), 0, 1)
        cur_candidates = cur_data["candidates"]
        query_list = [cur_candidate[0] for cur_candidate in cur_candidates]
        if len(query_list) <= 1:
            return None, None
        cur_query_emb = self.batch_forward(query_list)
        scores, indices = self.item_retrieval(cur_query_emb, cur_index)
        for cur_candidate, cur_score, cur_indice in zip(cur_candidates, scores, indices):
            if cur_score >= 0.4 and cur_score <= 0.95:
                cur_candidate.append(cur_score)
                keep_candidate.append(cur_candidate)
            else:
                drop_candidate.append(cur_candidate[0].strip())
        cur_data["candidates"] = keep_candidate
        return cur_data, drop_candidate

    def interest_coverage_calculation(self, cur_data):
        total_behavior = 0
        hit_behavior = 0
        query_list = cur_data["candidates"]
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

        cur_behavior_emb = self.batch_forward(total_user_behavior)
        scores, indices = self.item_retrieval(cur_behavior_emb, cur_query_index)

        for cur_behavior, cur_score, cur_indice in zip(total_user_behavior, scores, indices):
            total_behavior += 1
            if cur_score >= 0.3:
                hit_behavior += 1
        avg_coverage_rate = hit_behavior / total_behavior
        return round(avg_coverage_rate, 3)

    def hac_cluster(self, cur_data):
        ward_model = AgglomerativeClustering(n_clusters=10, metric='euclidean',
                                             linkage='ward',
                                             compute_full_tree=True)
        cur_candidates = [cur_candidate for cur_candidate in cur_data["candidates"]]
        query_list = [cur_candidate[0] for cur_candidate in cur_candidates]
        cur_query_emb = self.batch_forward(query_list).cpu().numpy()
        ward_model.fit(cur_query_emb)
        labels = list(ward_model.labels_)
        ward2gptquerys = {}
        for cur_candidate, label in zip(cur_candidates, labels):
            if label not in ward2gptquerys:
                ward2gptquerys[label] = []
            ward2gptquerys[label].append(cur_candidate)
        final_decision = []
        for item in ward2gptquerys:
            cur_candidates = ward2gptquerys[item]
            sorted_candidates = sorted(cur_candidates, key=lambda x: x[1] * x[2] * x[3] * x[4], reverse=True)
            print(sorted_candidates)
            top1_candidate = sorted_candidates[0][0]
            final_decision.append(top1_candidate)

        cur_data["result"] = final_decision
        del cur_data["candidates"]
        return cur_data


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

    def large_scale_inference(self, cur_data, task):
        keep_candidate = []
        drop_candidate = []

        with torch.no_grad():
            cur_candidates = cur_data["candidates"]
            cur_query = [item[0] for item in cur_candidates]
            tokenizer_result = self.tokenizer(cur_query, padding="max_length",
                                              max_length=32,
                                              truncation=True, return_tensors='pt')
            input_ids = tokenizer_result["input_ids"].cuda()
            attention_mask = tokenizer_result["attention_mask"].cuda()
            if task == "quality":
                scores = self.quality_model.forward(input_ids, attention_mask)
            else:
                scores = self.ctr_model.forward(input_ids, attention_mask)
            result = torch.nn.functional.softmax(scores, dim=-1).cpu().numpy().tolist()

            for cur_candidate, score in zip(cur_candidates, result):
                cur_query_score = sum([idx * item for idx, item in enumerate(score)])
                if task == "quality" and cur_query_score >= 1.8:
                    cur_candidate.append(cur_query_score)
                    keep_candidate.append(cur_candidate)
                elif task == "ctr" and cur_query_score >= 0.08:
                    cur_candidate.append(cur_query_score)
                    keep_candidate.append(cur_candidate)
                else:
                    drop_candidate.append(cur_candidate[0])
            cur_data["candidates"] = keep_candidate
        return cur_data, drop_candidate


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

    def do_item_inference(self, cur_data):
        cur_candidates = cur_data["candidates"]
        new_candidates = []
        for cur_candidate in cur_candidates:
            doc_embs = self.model.forward(cur_candidate[0].strip())
            scores, indices = self.item_retrieval(doc_embs, 5)
            total_score = 0
            for cur_score, cur_indice in zip(scores, indices):
                total_score += cur_score
            avg_score = round(total_score / 5, 3)
            if avg_score <= 0.75:
                continue
            cur_candidate.append(avg_score)
            new_candidates.append(cur_candidate)
        cur_data["candidates"] = new_candidates
        return cur_data


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model_path", type=str,
                        default="./mpnet_emb_model/")
    parser.add_argument("--quality_model_path", type=str,
                        default="./query_quality_model/")
    parser.add_argument("--ctr_model_path", type=str,
                        default="./query_ctr_model/")
    parser.add_argument("--id2emb_index_path", type=str, default="./ann_index/mpnet_id2emb_index_path.bin")
    parser.add_argument("--id2text_index_path", type=str, default="./ann_index/mpnet_id2text_index_path.bin")
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
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
    print("query judge model successful loaded")

    ## load query retrieval model
    retrieval_model = Query_Retrieval(args)
    print("query retrieval model successful loaded")

    with open(args.output_path, "w") as f_write:
        for idx, cur_data in tqdm(enumerate(all_data)):
            try:
                ## quality calculation
                cur_data, drop_candidate = query_model.large_scale_inference(cur_data, "quality")

                ## ctr calculation
                cur_data, drop_candidate = query_model.large_scale_inference(cur_data, "ctr")

                ## relevance calculation
                cur_data, drop_candidate = emb_model.relevance_calculation(cur_data)

                ## retrieval calculation
                cur_data = retrieval_model.do_item_inference(cur_data)

                cur_data = emb_model.hac_cluster(cur_data)

                f_write.write(json.dumps(cur_data, ensure_ascii=False) + "\n")
            except:
                continue
