'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-18 14:54:57
LastEditTime: 2024-05-18 16:38:28
Description: 

'''

import asyncio
from functools import partial
import random
import fire

from utils.request_tool import request_LLM
from utils.tools import read_jsonl

random.seed(42)

def create_prompt(data, prompt_config):
    instruction = """You need to perform multi-step reasoning, with each step carrying out as many basic operations as possible.

Remember, you can only complete tasks that contain up to 5 basic operations per step, and multiplication operations must be less than 1.5e5. The upper limit of the multiplication operations decreases as the number of operations per step increases.

[EXAMPLE]
Question: Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?
Answer: Leo finished the first and second parts of the assignment in 25 + 25*2 = <<25+25*2=75>>75 minutes.
Therefore, it took Leo 60 x 2 - 75 = <<60*2-75=45>>45 minutes to finish the third part of the assignment.
#### 45

Question: Liza bought 10 kilograms of butter to make cookies. She used one-half of it for chocolate chip cookies, one-fifth of it for peanut butter cookies, and one-third of the remaining butter for sugar cookies. How many kilograms of butter are left after making those three kinds of cookies?
Answer: Liza used 10 / 2 + 10 / 5 = <<10/2+10/5=7>>7 kilograms of butter for the chocolate and peanut butter cookies.
Then, Liza used (10 - 7) / 3 = <<(10-7)/3=1>>1 kilograms of butter for the sugar cookies.
Therefore, only 10-7-1 = <<10-7-1=2>>2 kilograms of butter were left.
#### 2

Question: Tina makes $18 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?
Answer: She works 5 days and makes 5 * 8 * $18 = $<<8*18*5=720>>720 regular pay.
Her overtime pay is 18+18*0.5 = $<<18+18*0.5=27>>27.
She works 2 hours of overtime for 5 days and makes 27*2*5 = $<<27*(10-8)*5=270>>270 in overtime pay.
She makes $720 + $270 = $<<720+270=990>>990.
#### 990

[REQUEST]
Question: """
    instruction += data["question"]
    return instruction


class DataLoader:
    def __init__(self, load_path: str) -> None:
        input_data = []
        for i, data in enumerate(read_jsonl(load_path)):
            if "index" not in data:
                data["index"] = str(i)
            input_data.append(data)
            
        self.data = input_data


def run(total=1, 
        split=0, 
        model_type="gpt",
        model_name="gpt-3.5-turbo",
        api_key="sk-xxx",
        request_proxy=None,
        max_tokens=300,
        temperature=0.2):
    
    model_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    asyncio.run(request_LLM(
        total=total,
        split=split,
        dataset=DataLoader("data/biggsm/data.jsonl"),
        save_path="experiments/CoT/gpt4o/marp-tp-05.jsonl",
        consumer_size=25,
        create_prompt_fn=partial(create_prompt, prompt_config=None),
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        enable_multi_turn = False,
        request_proxy=request_proxy,
        return_origin=True,
        model_config=model_config
        ))

if __name__ == "__main__":
    fire.Fire(run)
