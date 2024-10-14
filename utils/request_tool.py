import asyncio
from copy import deepcopy
import json
import os
import re
from tqdm import tqdm

from utils.tools import read_jsonl, write_jsonl

def judge_error(pred):
    try:
        float(pred)
    except:
        return False
    return True

class RequestOutput:
    def __init__(self, load_path, auto_index=True) -> None:
        temp_list = []
        if auto_index:
            for i, temp in enumerate(read_jsonl(load_path)):
                temp["index"] = str(i)
                temp_list.append(temp)
            self.data = sorted(read_jsonl(load_path), key=lambda x: int(x["index"]))
        else:
            self.data = read_jsonl(load_path)
    
    def save(self, save_path):
        write_jsonl(save_path, self.data, mode="w")
    
    def get_last_pred_text(self, index):
        return self.data[index]["pred"][-1]["content"][0]["text"]
            
    def get_origin_input(self, index):
         return self.data[index]["origin"]
    
    def search_by_question(self, question):
        for i, d in enumerate(self.data):
            if d["origin"]["question"] == question:
                return i
    
    def __len__(self):
        return len(self.data)

    def get_pred_answer(self, idx):
        pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', self.get_last_pred_text(idx).replace(",", "").strip(".").split("=")[-1])]
        if len(pred_list) == 0:
            pred1 = -1
        else:
            pred1 = pred_list[-1]
        return pred1
    
    def get_parsed_pred_answer(self, idx):
        pred_str = self.get_last_pred_text(idx)
        if "var1" not in pred_str or "<<" not in pred_str:
            pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', pred_str.replace(",", "").strip(".").split("=")[-1])]
            if len(pred_list) == 0:
                pred1 = -1
            else:
                pred1 = pred_list[-1]
            return pred1
        else:
            try:
                eqs = [s for s in re.findall(r'<<(.*?)>>', pred_str) if "=" in s]
                eqs = sorted(eqs, key=lambda x: int(x.split("=")[0].strip("var")))
                var_list = {eq.split("=")[0]: None for eq in eqs}
                for eq in eqs:
                    if "=" in eq:
                        func_str = eq.split("=")[1]
                        for var in var_list:
                            if var_list[var] is not None:
                                func_str = func_str.replace(var, str(var_list[var]))
                        if var_list[eq.split("=")[0]] is None:
                            try:
                                var_list[eq.split("=")[0]] = eval(func_str)
                            except:
                                return -1
                    elif "var" in eq:
                        pred_str += "#### " + eq
                if "####" in pred_str:
                    var_key = pred_str.split("####")[-1].strip().strip(".").replace("<", "").replace(">", "")
                    if var_key in var_list:
                        return var_list[var_key]
                last_var = var_list[list(var_list.keys())[-1]]
                if last_var is None:
                    last_var = -1
            except:
                return -1
            return last_var
    
    def get_program_answer(self, idx):
        pred_str = self.get_last_pred_text(idx)
        if "def " not in pred_str or "```" not in pred_str:
            return self.get_pred_answer(idx)
        else:
            if "```" in pred_str:
                pred_str = pred_str.split("```")[1]
            if "while" in pred_str:
                return -1
            g = {}
            l = {}
            
            try:
                exec(pred_str.strip(), g, l)
                return l["solver"]()
            except Exception as e:
                # print(e)
                # print(pred_str)
                return -1
    
    def get_text_answer(self, idx):
        return self.get_last_pred_text(idx).split("####")[-1]

    def judge_correct(self, idx, mode="nl"):
        golden_answer_str = self.get_origin_input(idx)["answer"].replace(",", "").strip(".").split("\n#### ")[-1]
        golden_answer = round(float(golden_answer_str), 2)
        if mode == "nl":
            pred = self.get_pred_answer(idx)
        elif mode == "tool":
            pred = self.get_parsed_pred_answer(idx)
        elif mode == "pot":
            pred = self.get_program_answer(idx)

        return judge_error(pred) and abs(abs(round(float(pred), 2)) - abs(round(golden_answer, 2))) < 0.01

class MMRequestor:
    def __init__(self,
                 model_type="gpt",
                 model_name="MODEL_NAME",
                 api_key="YOUR_API_KEY",
                 enable_multi_turn=False,
                 request_proxy=None) -> None:
        
        self.model_type = model_type
        self.model_name = model_name
        self.enable_multi_turn = enable_multi_turn
        self.chat = []
        if model_type == "gpt":
            from openai import AsyncOpenAI
            if request_proxy is not None:
                client = AsyncOpenAI(api_key=api_key, base_url=request_proxy)
            else:
                client = AsyncOpenAI(api_key=api_key)
            self.requestor = client
        else:
            raise ValueError("Not Supported other model besides ['gpt']")
    
    async def request(self, prompts, **kargs):
        
        if self.model_type == "gpt":
            if isinstance(prompts, list):
                for prompt in prompts:
                    self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                    response = await self.requestor.chat.completions.create(
                        model=self.model_name,
                        messages=self.chat,
                        **kargs
                        )
                    self.chat.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": response.choices[0].message.content,
                        }]
                    })
            else:
                prompt = prompts
                self.chat.append({
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": prompt,
                        }],
                })
                response = await self.requestor.chat.completions.create(
                    model=self.model_name,
                    messages=self.chat,
                    **kargs
                    )
                self.chat.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": response.choices[0].message.content,
                    }]
                })
            res_str = deepcopy(self.chat)
            self.chat = []
            return res_str
        
        
def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf8") as f:
        f.write(json_string + "\n")

async def producer(queue, dataset, save_path, bar, create_prompt):
    if os.path.exists(save_path):
        last_request = [x["index"] for x in read_jsonl(save_path)]
    else:
        last_request = []
    for i, data in enumerate(dataset.data):
        if data["index"] in last_request:
            bar.update(1)
            print(f"Skip {data['index']}")
            continue
        prompt = create_prompt(data)
        print("Loaded\t\t#", i)
        data.update({"index": data["index"], "text": prompt})
        await queue.put(data)
    print("Dataset Loaded.")
    await queue.put(None)


async def consumer(queue,
                   save_path,
                   bar,
                   model_type,
                   model_name,
                   api_key,
                   enable_multi_turn,
                   request_proxy,
                   return_origin,
                   model_config
                   ):
    while True:
        item = await queue.get()
        if item is None:
            print("Consumer Break")
            queue.task_done()
            break
        text = item["text"]
        
        
        
        try:
            print("Requesting\t\t#", item["index"])
            requestor = MMRequestor(model_type=model_type,
                                    model_name=model_name,
                                    api_key=api_key,
                                    enable_multi_turn=enable_multi_turn,
                                    request_proxy=request_proxy)
            result = await requestor.request(
                prompts=text,
                **model_config
            )
            if return_origin:
                append_to_jsonl({"index": item["index"], "pred": result, "origin": item}, save_path)
            else:
                append_to_jsonl({"index": item["index"], "pred": result}, save_path)
        except Exception as e:
            print(e)
        print("Saved\t\t#", item["index"])
        bar.update(1)
        queue.task_done()
        print(f"Queue left: {queue.qsize()}, Finished: {item['index']}")

async def request_LLM(total,
                model_type,
                model_name,
                api_key,
                enable_multi_turn,
                split=0,
                dataset=None,
                save_path = "",
                consumer_size=15,
                create_prompt_fn=None,
                request_proxy=None,
                return_origin=True,
                model_config={}):
    queue = asyncio.Queue(maxsize=120)  
    
    if dataset is None:
        return
    
    step = int(len(dataset.data)/total)
    dataset.data = dataset.data[step*split:min(len(dataset.data), step*(split+1))]
    bar = tqdm(total=len(dataset.data), desc=f"Total: {total} Split: {split}")
    producer_task = asyncio.create_task(producer(queue, dataset, save_path, bar, create_prompt_fn))
    consumer_tasks = [asyncio.create_task(consumer(queue, save_path, bar, model_type, model_name, api_key, enable_multi_turn, request_proxy, return_origin, model_config)) for _ in range(consumer_size)]  
    
    
    await producer_task
    await queue.join() 
    print("Cleaned Queue!")
    
    for task in consumer_tasks:
        task.cancel()
    print("Cleaned Tasks!")
    exit()