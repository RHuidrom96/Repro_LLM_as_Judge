#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================================================
# Created By  : Rudali Huidrom
# Created Date: Fri Jan 31 2025
# ================================================================================================
"""
    The module create prompt for the following models: mistral, llama3-8b, command-r-plus, granite,
                                                       deepseek-llama3-70b, llama3-70b, qwen2.5-7b, 
                                                       qwen2-72b.
"""
# =================================================================================================
# Imports
# =================================================================================================

import argparse
from expt_params import params
from data_call import rotowire_data, atanasova_data, feng_data

all_params = params()
rotowire_lst = rotowire_data()
atanasova_lst = atanasova_data()
feng_lst = feng_data()

# Pudupully and Lapata repro
summaries = all_params[0]['summaries']
sys_summaries = all_params[0]['sys_summaries']
A = all_params[0]['A']
B = all_params[0]['B']
rank_criteria = all_params[0]['rank_criteria']
Coherence = all_params[0]['Coherence']
Repetition = all_params[0]['Repetition']
Grammaticality = all_params[0]['Grammaticality']
answer = all_params[0]['answer']
best = all_params[0]['best']
worst = all_params[0]['worst']
analysis = all_params[0]['analysis']

# Atanasova et al repro
claim = all_params[0]['claim']
label = all_params[0]['label']
justification_1 = all_params[0]['justification_1']
justification_2 = all_params[0]['justification_2']
justification_3 = all_params[0]['justification_3']

# Feng et al repro
summary = all_params[0]['summary']

mistral_bos = all_params[1]['BOS']
mistral_eos = all_params[1]['EOS']
mistral_sot = all_params[1]['start_token']
mistral_eot = all_params[1]['end_token']

llama3_bos = all_params[2]['BOS']
llama3_eos = all_params[2]['EOS']
llama3_sot = all_params[2]['start_token']
llama3_eot = all_params[2]['end_token']

commandrplus_instruction = all_params[3]['Instruction']
commandrplus_input = all_params[3]['Input']
commandrplus_output = all_params[3]['Output']
commandrplus_criterion = all_params[3]['Criterion']

llama70_bos = all_params[4]['BOS']
llama70_eos = all_params[4]['EOS']
llama70_user = all_params[4]['user']
llama70_assistant = all_params[4]['assistant']

def rotowire_templates(model: str, criterion: str) -> list:
    all_prompts = []
    for item in rotowire_lst:
        a = item['sum1']
        b = item['sum2']
        if model == 'llama3':
            if criterion == 'coherence':
                prompt = f"{llama3_bos}\n{llama3_sot}{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Coherence}{answer}{best}\n{worst}\n{analysis}{llama3_eot}\n{llama3_eos}\n\nOutput\nBest: \nWorst: \n"
            elif criterion == 'repetition':
                prompt = f"{llama3_bos}\n{llama3_sot}{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Repetition}{answer}{best}\n{worst}\n{analysis}{llama3_eot}\n{llama3_eos}\n\nOutput:\nBest: \nWorst: \n"
            elif criterion == 'grammaticality':
                prompt = f"{llama3_bos}\n{llama3_sot}{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Grammaticality}{answer}{best}\n{worst}\n{analysis}{llama3_eot}\n{llama3_eos}\n\nOutput:\nBest: \nWorst: \n"
        elif model == 'mistral':
            if criterion == 'coherence':
                prompt = f"{mistral_bos}{mistral_sot}{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Coherence}{answer}{best}\n{worst}\n{analysis}{mistral_eot}{mistral_eos}\n\nOutput:\nBest: \nWorst: \n"
            elif criterion == 'repetition':
                prompt = f"{mistral_bos}{mistral_sot}{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Repetition}{answer}{best}\n{worst}\n{analysis}{mistral_eot}{mistral_eos}\n\nOutput:\nBest: \nWorst: \n"
            elif criterion == 'grammaticality':
                prompt = f"{mistral_bos}{mistral_sot}{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Grammaticality}{answer}{best}\n{worst}\n{analysis}{mistral_eot}{mistral_eos}\n\nOutput:\nBest: \nWorst: \n"
        elif model == 'command-r-plus':
            if criterion == 'coherence':
                prompt = f"{commandrplus_instruction}{summaries}{sys_summaries}\n{commandrplus_input}{A}{a}\n{B}{b}\n{commandrplus_criterion}{rank_criteria}{Coherence}\n{commandrplus_output}{answer}{best}\n{worst}\n{analysis}\n\nOutput:\nBest: \nWorst: \n"
            elif criterion == 'repetition':
                prompt = f"{commandrplus_instruction}{summaries}{sys_summaries}\n{commandrplus_input}{A}{a}\n{B}{b}\n{commandrplus_criterion}{rank_criteria}{Repetition}\n{commandrplus_output}{answer}{best}\n{worst}\n{analysis}\n\nOutput:\nBest: \nWorst: \n"
            elif criterion == 'grammaticality':
                prompt = f"{commandrplus_instruction}{summaries}{sys_summaries}\n{commandrplus_input}{A}{a}\n{B}{b}\n{commandrplus_criterion}{rank_criteria}{Grammaticality}\n{commandrplus_output}{answer}{best}\n{worst}\n{analysis}\n\nOutput:\nBest: \nWorst: \n"
        elif model == 'qwen25':
            if criterion == 'coherence':
                prompt = f"{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Coherence}{answer}{best}\n{worst}\n{analysis}\n\nOutput:\nBest: \nWorst: \n"
            elif criterion == 'repetition':
                prompt = f"{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Repetition}{answer}{best}\n{worst}\n{analysis}\n\nOutput:\nBest: \nWorst: \n"
            elif criterion == 'grammaticality':
                prompt = f"{summaries}{sys_summaries}{A}{a}\n{B}{b}\n{rank_criteria}{Grammaticality}{answer}{best}\n{worst}\n{analysis}\n\nOutput:\nBest: \nWorst: \n"
        else:
            raise ValueError("Prompt doesn't exist in the conditional statement for the selected model. Please select the right model.")
        curr_entry = {"code": item['code'], "system1": item['system1'], "system2": item['system2'], "worker_id": model, "prompt": prompt}
        all_prompts.append(curr_entry)

    return all_prompts

def atanasova_templates(model: str) -> list:
    all_prompts = []
    for item in atanasova_lst:
        item_claim = item['claim']
        item_label = item['LABEL']
        item_justification_1 = item['justification 1']
        item_justification_2 = item['justification 2']
        item_justification_3 = item['justification 3']
        if model == 'llama3':
            prompt = f"{llama3_bos}\n{llama3_sot}{claim}{item_claim}\n{label}{item_label}\n{justification_1}{item_justification_1}\n{justification_2}{item_justification_2}\n{justification_3}{item_justification_3}{llama3_eot}\n{llama3_eos}\nCoverage rank for Justification 1: \nCoverage rank for Justification 2: \nCoverage rank for Justification 3: \n"
        elif model == 'mistral':
            prompt = f"{mistral_bos}{mistral_sot}{claim}{item_claim}\n{label}{item_label}\n{justification_1}{item_justification_1}\n{justification_2}{item_justification_2}\n{justification_3}{item_justification_3}{mistral_eot}{mistral_eos}\nCoverage rank for Justification 1: \nCoverage rank for Justification 2: \nCoverage rank for Justification 3: \n"
        elif model == 'command-r-plus':
            prompt = f"{commandrplus_input}{claim}{item_claim}\n{label}{item_label}\n{justification_1}{item_justification_1}\n{justification_2}{item_justification_2}\n{justification_3}{item_justification_3}\n{commandrplus_output}\nCoverage rank for Justification 1: \nCoverage rank for Justification 2: \nCoverage rank for Justification 3: \n"
        elif model == 'qwen25':
            prompt = f"{claim}{item_claim}\n{label}{item_label}\n{justification_1}{item_justification_1}\n{justification_2}{item_justification_2}\n{justification_3}{item_justification_3}\nCoverage rank for Justification 1: \nCoverage rank for Justification 2: \nCoverage rank for Justification 3: \n"
        elif model == 'deepseek_llama70':
            prompt = f"{llama70_bos}\n{llama70_user}{claim}{item_claim}\n{label}{item_label}\n{justification_1}{item_justification_1}\n{justification_2}{item_justification_2}\n{justification_3}{item_justification_3}{llama70_assistant}\nCoverage rank for Justification 1: \nCoverage rank for Justification 2: \nCoverage rank for Justification 3: \n"
        elif model == 'qwen2_72':
            prompt = f"{claim}{item_claim}\n{label}{item_label}\n{justification_1}{item_justification_1}\n{justification_2}{item_justification_2}\n{justification_3}{item_justification_3}\nCoverage rank for Justification 1: \nCoverage rank for Justification 2: \nCoverage rank for Justification 3: \n"
        elif model == 'llama3_70':
            prompt = f"{llama70_bos}\n{llama70_user}{claim}{item_claim}\n{label}{item_label}\n{justification_1}{item_justification_1}\n{justification_2}{item_justification_2}\n{justification_3}{item_justification_3}{llama70_assistant}\nCoverage rank for Justification 1: \nCoverage rank for Justification 2: \nCoverage rank for Justification 3: \n"
        else:
            raise ValueError("Prompt doesn't exist in the conditional statement for the selected model. Please select the right model.")
        curr_entry = {"id": item['id'], "worker_id": model, "prompt": prompt}
        all_prompts.append(curr_entry)

    return all_prompts

def feng_templates(model: str) -> list:
    all_prompts = []
    for item in feng_lst:
        item_meeting_ID = item['Meeting_ID']
        item_meeting_notes = item['Meeting_notes']
        item_system_ID = item['System_ID']
        item_system_summary = item['System_summary']
        if model == 'llama3':
            prompt = f"{llama3_bos}\n{llama3_sot}{item_meeting_ID}\n{item_meeting_notes}\n{summary}\n{item_system_summary}{llama3_eot}\n{llama3_eos}\nInformativeness rating between 1 to 5: \n"
        elif model == 'mistral':
            prompt = f"{mistral_bos}{mistral_sot}{item_meeting_ID}\n{item_meeting_notes}\n{summary}\n{item_system_summary}{mistral_eot}{mistral_eos}\nInformativeness rating between 1 to 5: \n"
        elif model == 'command-r-plus':
            prompt = f"{commandrplus_input}{item_meeting_ID}\n{item_meeting_notes}\n{summary}\n{item_system_summary}\n{commandrplus_output}\nInformativeness rating between 1 to 5: \n"
        elif model == 'qwen25':
            prompt = f"{item_meeting_ID}\n{item_meeting_notes}\n{summary}\n{item_system_summary}\nInformativeness rating between 1 to 5: \n"
        elif model == 'qwen2_72':
            prompt = f"{item_meeting_ID}\n{item_meeting_notes}\n{summary}\n{item_system_summary}\nInformativeness rating between 1 to 5: \n"
        elif model == 'llama3_70':
            prompt = f"{llama70_bos}\n{llama70_user}{item_meeting_ID}\n{item_meeting_notes}\n{summary}\n{item_system_summary}{llama70_assistant}\nInformativeness rating between 1 to 5: \n"
        elif model == 'granite':
            prompt = f"{item_meeting_ID}\n{item_meeting_notes}\n{summary}\n{item_system_summary}\nInformativeness rating between 1 to 5: \n"
        else:
            raise ValueError("Prompt doesn't exist in the conditional statement for the selected model. Please select the right model.")
        curr_entry = {"meeting_ID": item_meeting_ID, "system_ID": item_system_ID, "worker_id": model, "prompt": prompt}
        all_prompts.append(curr_entry)
    return all_prompts