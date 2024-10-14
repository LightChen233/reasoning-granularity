'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2024-03-03 15:00:53
LastEditTime: 2024-05-18 18:04:24
Description: 

'''
import tiktoken
import fire
from utils.request_tool import RequestOutput
from utils.tools import get_combined_granularity
from prettytable import PrettyTable

def loop_judge(condition_list, input_str):
    for cond in condition_list:
        if cond in input_str:
            return True
    return False



PARAM_DICT = {
    "CoT": {
        "K": 0.106,
        "K2": 0.425,
        "mode": "nl",
        "result_path": "experiments/mathematical-reasoning/gpt35-CoT.jsonl"
    },
    "Tool-Usage": {
        "K": 0.12,
        "K2": 0.53,
        "mode": "tool",
        "result_path": "experiments/mathematical-reasoning/gpt35-tool.jsonl"
    },
    "PoT": {
        "K": 0.13,
        "K2": 0.81,
        "mode": "pot",
        "result_path": "experiments/mathematical-reasoning/gpt35-PoT.jsonl"
    },
    "Complex-CoT": {
        "K": 0.107,
        "K2": 0.50,
        "mode": "nl",
        "result_path": "experiments/mathematical-reasoning/gpt35-complex.jsonl"
    },
    "LtM": {
        "K": 0.106,
        "K2": 0.44,
        "mode": "nl",
        "result_path": "experiments/mathematical-reasoning/gpt35-LtM.jsonl"
    },
    "MARP": {
        "K": 0.11,
        "K2": 0.43,
        "mode": "nl",
        "result_path": "experiments/mathematical-reasoning/gpt35-marp.jsonl"
    },
    "PoT-MARP": {
        "K": 0.135,
        "K2": 0.81,
        "mode": "pot",
        "result_path": "experiments/mathematical-reasoning/gpt35-pot-marp.jsonl"
    },
    "gpt-4o": {
        "K": 0.12,
        "K2": 0.71,
        "mode": "nl",
        "result_path": "experiments/mathematical-reasoning/gpt4o-CoT.jsonl"
    },
    "gpt-4o-MARP": {
        "K": 0.15,
        "K2": 0.99,
        "mode": "nl",
        "result_path": "private/CoT/gpt4o/marp-tp-01-2.jsonl"
    },
    "o1-preview": {
        "K": 0.301,
        "K2": 0.92,
        "mode": "nl",
        "result_path": "experiments/mathematical-reasoning/o1-CoT.jsonl"
    },
}

def main(data_split="CoT",
         K=None,
         K2=None,
         mode=None,
         result_path=None):
    if data_split == "custom":
        if K is None:
            raise ValueError("Missing Args: --K")
        elif K2 is None:
            raise ValueError("Missing Args: --K2")
        elif mode is None:
            raise ValueError("Missing Args: --mode")
        elif result_path is None:
            raise ValueError("Missing Args: --result_path")
    else:
        K = PARAM_DICT[data_split]["K"]
        K2 = PARAM_DICT[data_split]["K2"]
        mode = PARAM_DICT[data_split]["mode"]
        result_path = PARAM_DICT[data_split]["result_path"]
    
    response_list = RequestOutput(result_path)
    table = PrettyTable()
    table.field_names = ["Granularity", "Acc", "Input Token", "Output Token"]

    token_num = 0
    input_token_num = 0
    enc = tiktoken.encoding_for_model("gpt-4")
    acc = {">90%": {"correct": 0, "total": 0}, "10%~90%": {"correct": 0, "total": 0}, "<10%": {"correct": 0, "total": 0}}
    
    for idx in range(len(response_list)):
        granularity = get_combined_granularity(response_list.get_origin_input(idx))
        input_token_num += len(enc.encode(response_list.data[idx]["pred"][0]["content"][0]["text"]))
        token_num += len(enc.encode(response_list.get_last_pred_text(idx)))
        
        if granularity <= K:
            granularity_key = ">90%"
        elif granularity > K  and granularity <= K2:
            granularity_key = "10%~90%"
        elif granularity > K2:
            granularity_key = "<10%"
        if response_list.judge_correct(idx, mode=mode):
            acc[granularity_key]["correct"] += 1
        acc[granularity_key]["total"] += 1
    total = 0
    correct = 0
    for key in acc:
        if acc[key]['total']>0:
            total += acc[key]['total']
            correct += acc[key]['correct']
            table.add_row([
                key,
                round(acc[key]['correct']/acc[key]['total'] * 100, 2),
                "-",
                "-"
            ], divider=key == "<10%")
        else:
            table.add_row([
                key,
                "-",
                "-",
                "-"
            ], divider=key == "<10%")
    table.add_row(["All", round(correct/total * 100, 2), round(input_token_num/total, 2), round(token_num/total, 2)])
    print(table)
    
if __name__ == "__main__":
    fire.Fire(main)