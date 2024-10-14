'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2024-03-05 14:48:28
LastEditTime: 2024-07-05 16:27:15
Description: 

'''
from copy import deepcopy
import json
import random
import re
random.seed(42)


import re


class GSM8KData():
    def __init__(self, obj) -> None:
        self.obj = obj
    
    def get_answer(self):
        res_str = self.obj["answer"].replace(",", "").strip(".").split("\n#### ")[-1]
        try:
            return round(float(res_str), 2)
        except:
            return -1
    
    def get_text_answer(self):
        return self.obj["answer"]
    
    def extract_equation(self, obj):
        exp_list = [s for s in re.findall(r'<<(.*)?>>', obj["answer"])]
        equation_list = []
        obj["operation"] = {"+": 0, "-": 0, "*": 0, "/": 0}
        for exp in exp_list:
            exp = exp.strip(".0").strip(".00")
            ans = exp.split("=")[-1].strip()
            exp = exp.split("=")[0]
            operations = re.findall(r"\+|\-|\*|\/", exp)
            for operation in operations:
                obj["operation"][operation] += 1
            if ans == "":
                ans = "0"
            equation_list.append({"func": exp, "ans": ans})
        return obj, equation_list
    
    
    def __str__(self) -> str:
        return json.dumps(self.obj)