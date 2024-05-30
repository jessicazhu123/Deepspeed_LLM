import argparse
import json
import torch
import time
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

distillation_prompt_template_V0 = """# Task Description

Given user's browsing history while using Google, let's recommend personalized news keyphrases that match user's long-term and short-term interests.

## Requirement
You should generate exactly {number} news keyphrases

## User Browsing History

### User Search Query
{search_query}

### User Clicked Web Title
{clicked_web_title}

### User Clicked News Article Title
{clicked_news_title}

## Recommendation
"""

distillation_prompt_template_V1 = """# Task Description
Given user's browsing history while using Google, let's recommend personalized news keyphrases that match user's long-term and short-term interests.

## Requirement
You should generate exactly {number} news keyphrases

### Google Search History
{search_query}

### Google Clicked Search Results
{clicked_web_title}

### Google Clicked News Articles
{clicked_news_title}

### Chrome Clicked Results
{edge_clicks}

## Recommendation
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        help="input data path",
                        default="./data_input/test_input_5K_std")
    parser.add_argument("--output_path",
                        type=str,
                        help="output data path",
                        default="./data_output/")
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model",
                        default="./cur_best_llama3_model")
    parser.add_argument("--query_number",
                        type=int,
                        default=10)
    parser.add_argument("--generation_number",
                        type=int,
                        default=50)
    parser.add_argument("--temperature",
                        type=float,
                        default=0.6,
                        help="temperature for generation")
    parser.add_argument("--top_p",
                        type=float,
                        default=0.8,
                        help="top p for generation")
    parser.add_argument("--presence_penalty",
                        type=float,
                        default=0.5,
                        help="frequency penalty")
    args = parser.parse_args()
    return args


class model_inference_vllm():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
        self.model = LLM(model=args.path, trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                              presence_penalty=args.presence_penalty,
                                              max_tokens=512, stop_token_ids=[self.tokenizer.eos_token_id])

    def vllm_inference(self):
        with open(self.args.input_path) as f_read, open(self.args.output_path, "w") as f_write:
            with torch.no_grad():
                all_inputs = []
                all_inputs_dict = []
                for idx, item in tqdm(enumerate(f_read)):
                    infos = json.loads(item.strip())
                    BingSearchEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["BingSearchEvents"].strip())])
                    BingClickEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["BingClickEvents"].strip())])
                    EdgeClickEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["EdgeClickEvents"].strip())])
                    MSNClickEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["MSNClickEvents"].strip())])
                    cur_data_dict = {"number": str(self.args.query_number), "search_query": BingSearchEvents,
                                     "clicked_web_title": BingClickEvents,
                                     "clicked_news_title": MSNClickEvents,
                                     "edge_clicks": EdgeClickEvents}
                    cur_input = distillation_prompt_template_V1.format(**cur_data_dict)
                    if len(self.tokenizer.tokenize(cur_input.strip())) >= 7500:
                        BingSearchEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["BingSearchEvents"].strip())][:100])
                        BingClickEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["BingClickEvents"].strip())][:100])
                        EdgeClickEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["EdgeClickEvents"].strip())][:100])
                        MSNClickEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["MSNClickEvents"].strip())][:100])
                        cur_data_dict = {"number": str(self.args.query_number), "search_query": BingSearchEvents,
                                         "clicked_web_title": BingClickEvents,
                                         "clicked_news_title": MSNClickEvents,
                                         "edge_clicks": EdgeClickEvents}
                        cur_input = distillation_prompt_template_V1.format(**cur_data_dict)

                    all_inputs.append(cur_input)
                    all_inputs_dict.append(cur_data_dict)

                start = time.time()
                outputs = self.model.generate(all_inputs, self.sampling_params)
                duration = time.time() - start
                total_input_length = 0
                total_output_length = 0
                total_request = 0
                for cur_input, cur_input_dict, cur_output in zip(all_inputs, all_inputs_dict, outputs):
                    generated_text = cur_output.outputs[0].text
                    total_input_length += len(self.tokenizer.tokenize(cur_input.strip()))
                    total_output_length += len(self.tokenizer.tokenize(generated_text.strip()))
                    total_request += 1
                    cur_input_dict["search_query"] = cur_input_dict["search_query"].strip().split("\n")
                    cur_input_dict["clicked_web_title"] = cur_input_dict["clicked_web_title"].strip().split("\n")
                    cur_input_dict["clicked_news_title"] = cur_input_dict["clicked_news_title"].strip().split("\n")
                    cur_input_dict["edge_clicks"] = cur_input_dict["edge_clicks"].strip().split("\n")
                    cur_input_dict["result"] = generated_text.strip().split("\n")
                    f_write.write(json.dumps(cur_input_dict, ensure_ascii=False) + "\n")
                print("avg token per prompt:" + str(total_input_length / len(all_inputs)))
                print("avg token per second:" + str(total_output_length / duration))
                print("avg request per second:" + str(total_request / duration))

    def vllm_inference_for_dpo_exploration(self):
        with open(self.args.input_path) as f_read, open(self.args.output_path, "a") as f_write:
            with torch.no_grad():
                all_input_dicts = []
                all_inputs = []
                lines = f_read.readlines()
                for idx, item in tqdm(enumerate(lines)):
                    infos = json.loads(item.strip())
                    BingSearchEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["BingSearchEvents"].strip())])
                    BingClickEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["BingClickEvents"].strip())])
                    EdgeClickEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["EdgeClickEvents"].strip())])
                    MSNClickEvents = "\n".join(
                        [item["Title"].strip() for item in json.loads(infos["MSNClickEvents"].strip())])
                    cur_data_dict = {"number": str(self.args.query_number), "search_query": BingSearchEvents,
                                     "clicked_web_title": BingClickEvents,
                                     "clicked_news_title": MSNClickEvents,
                                     "edge_clicks": EdgeClickEvents}

                    cur_input = distillation_prompt_template_V1.format(**cur_data_dict)
                    if len(self.tokenizer.tokenize(cur_input.strip())) >= 7500:
                        BingSearchEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["BingSearchEvents"].strip())][:100])
                        BingClickEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["BingClickEvents"].strip())][:100])
                        EdgeClickEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["EdgeClickEvents"].strip())][:100])
                        MSNClickEvents = "\n".join(
                            [item["Title"].strip() for item in json.loads(infos["MSNClickEvents"].strip())][:100])
                        cur_data_dict = {"number": str(self.args.query_number), "search_query": BingSearchEvents,
                                         "clicked_web_title": BingClickEvents,
                                         "clicked_news_title": MSNClickEvents,
                                         "edge_clicks": EdgeClickEvents}
                        cur_input = distillation_prompt_template_V1.format(**cur_data_dict)

                    all_input_dicts.append(cur_data_dict)
                    all_inputs.extend([cur_input] * self.args.generation_number)

                mini_batch_size = 10
                total_batch_size = mini_batch_size * self.args.generation_number
                batch_num = int(len(all_inputs) / total_batch_size) + 1
                for batch_id in tqdm(range(batch_num)):
                    cur_inputs = all_inputs[batch_id * total_batch_size:(batch_id + 1) * total_batch_size]
                    cur_input_dicts = all_input_dicts[batch_id * mini_batch_size:(batch_id + 1) * mini_batch_size]
                    if len(cur_inputs) == 0:
                        continue
                    outputs = self.model.generate(cur_inputs, self.sampling_params)
                    for idx, cur_input_dict in enumerate(cur_input_dicts):
                        cur_final_outputs = []
                        cur_outputs = outputs[idx * self.args.generation_number:(idx + 1) * self.args.generation_number]
                        for cur_output in cur_outputs:
                            generated_text = cur_output.outputs[0].text.strip()
                            generated_text = list(set(generated_text.strip().split("\n")))
                            generated_text = [item for item in generated_text if item.strip() != ""]
                            if len(generated_text) != 0:
                                cur_final_outputs.extend(generated_text)
                        cur_final_outputs = list(set(cur_final_outputs))
                        cur_input_dict["search_query"] = cur_input_dict["search_query"].strip().split("\n")
                        cur_input_dict["clicked_web_title"] = cur_input_dict["clicked_web_title"].strip().split("\n")
                        cur_input_dict["clicked_news_title"] = cur_input_dict["clicked_news_title"].strip().split("\n")
                        cur_input_dict["edge_clicks"] = cur_input_dict["edge_clicks"].strip().split("\n")
                        cur_input_dict["candidates"] = cur_final_outputs
                        f_write.write(json.dumps(cur_input_dict, ensure_ascii=False).strip() + "\n")



if __name__ == "__main__":
    args = parse_args()
    inference_engine = model_inference_vllm(args)
    inference_engine.vllm_inference()
