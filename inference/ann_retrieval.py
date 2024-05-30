import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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


class DenseRetriever(object):
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

    def do_item_inference(self):
        search_query = input("Pls Input Search Query: ").strip()
        while True:
            doc_embs = self.model.forward(search_query)
            scores, indices = self.item_retrieval(doc_embs, 3)
            for cur_score, cur_indice in zip(scores, indices):
                docid, title = self.id2text[cur_indice]
                print(title.strip())
                print(cur_score)
                print("----------------")
            search_query = input("Pls Input Search Query: ").strip()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_query_len", type=int, default=32)
    parser.add_argument("--embedding_model_path", type=str)
    parser.add_argument("--id2emb_index_path", type=str, default="./ann_index/mpnet_id2emb_index_path.bin")
    parser.add_argument("--id2text_index_path", type=str, default="./ann_index/mpnet_id2text_index_path.bin")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    retrieval = DenseRetriever(args)
    retrieval.do_item_inference()
