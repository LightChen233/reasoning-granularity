'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-10-13 14:24:40
LastEditTime: 2024-03-14 21:48:48
Description: 

'''
import json
import os
import re
import time


def read_jsonl(data_path):
    input_data = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                input_data.append(json.loads(line.strip()))
    else:
        print(f"Missing {data_path}")
    return input_data


def write_jsonl(save_path, save_object, mode="a"):
    with open(save_path, mode, encoding="utf8") as f:
        for obj in save_object:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def evaluate_expression(expression):
    max_dict = {"plus": 0, "time": 0}
    def parse_expression(i):
        value, i = parse_term(i)
        while i < len(expression) and expression[i] in '+-':
            if expression[i] == '+':
                i += 1
                right_value, i = parse_term(i)
                value += right_value
                if abs(value) > abs(max_dict["plus"]):
                    max_dict["plus"] = abs(value)
            elif expression[i] == '-':
                i += 1
                right_value, i = parse_term(i)
                value -= right_value
                
        return value, i
    
    def parse_term(i):
        value, i = parse_factor(i)
        while i < len(expression) and expression[i] in '*/':
            if expression[i] == '*':
                i += 1
                right_value, i = parse_factor(i)
                value *= right_value
                if abs(value) > abs(max_dict["time"]):
                    max_dict["time"] = abs(value)
            elif expression[i] == '/':
                i += 1
                right_value, i = parse_factor(i)
                value /= right_value
        return value, i
    
    def parse_factor(i):
        if expression[i] == '(':
            i += 1  # Skip '('
            value, i = parse_expression(i)
            i += 1  # Skip ')'
        else:
            start_i = i
            while i < len(expression) and (expression[i] == "." or expression[i].isdigit()):
                i += 1
            if "." in expression:
                value = float(expression[start_i:i])
            else:
                value = int(expression[start_i:i])
        return value, i
    
    
    expression = expression.replace(' ', '')
    if expression.startswith("-"):
        expression = "0" + expression
    value, _ = parse_expression(0)
    return value, max_dict


def get_combined_granularity(origin_data, return_dict=False):
    N = 1.6e5
    M = 7.0
    SIGMA = 20000
    origin_eqs = [s for s in re.findall(r'<<(.*)?>>', origin_data["answer"])]
    
    operation_list = [operation for eq1 in origin_eqs for operation in re.findall(r'[\+\-\*/]', eq1.split("=")[0])]
    
    max_time = 0
    
    for eq0 in origin_eqs:
        _, max_dict = evaluate_expression(eq0.split("=")[0])
        if max_time < max_dict["time"]:
            max_time = max_dict["time"]
    
    calculate_granularity = max_time
    
    if len(operation_list) == len(origin_eqs):
        plan_granularity= len([x for x in origin_eqs if not x.strip("0.").startswith("0")])
    else:
        plan_granularity = len(operation_list)
    if return_dict:
        return {
            "plan_granularity": plan_granularity,
            "calculate_granularity": calculate_granularity,
            "combined_granularity": 1/(M/plan_granularity + N/(calculate_granularity + SIGMA))
        }
    return 1/(M/plan_granularity + N/(calculate_granularity + SIGMA))


    