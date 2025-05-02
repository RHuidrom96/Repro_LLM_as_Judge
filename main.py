#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================================================
# Created By  : Rudali Huidrom
# Created Date: Fri Jan 30 2025
# ================================================================================================
"""
    The module has been build for inferencing LLMs to perform the WebNLG 2020 human evaluation.
"""
# =================================================================================================
# Imports
# =================================================================================================
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights
from dotenv import load_dotenv, find_dotenv
from instruction import rotowire_templates, atanasova_templates, feng_templates
from expt_params import params
from preprocess import *

_ = load_dotenv(find_dotenv())
hf_token = os.getenv("ADD YOUR HF TOKEN VAR")
device = "cuda" if torch.cuda.is_available() else "cpu" # the device to load the model onto

all_params = params()
gen_instr_rotowire = all_params[0]['gen_instr_rotowire']
task_head_rotowire = all_params[0]['task_head_rotowire']
task_instr_rotowire = all_params[0]['task_instr_rotowire']
task_desc_rotowire = all_params[0]['task_desc_rotowire']

instr_head = all_params[0]['instr_head']
task_1 = all_params[0]['task_1']
main_instr_1 = all_params[0]['main_instr_1']
coverage_def = all_params[0]['coverage_def']
main_instr_2 = all_params[0]['main_instr_2']
example_1 = all_params[0]['example_1']
note = all_params[0]['note']
example_2 = all_params[0]['example_2']

feng_instr = all_params[0]['feng_instr']

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_quant_type='nf4',
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.bfloat16)


###################### Mistral ############################

def inference_mistral(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"
  
    all_outputs = []

    model_id = "mistralai/Mistral-7B-Instruct-v0.2" 

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=quantization_config,
                                                 trust_remote_code=True,
                                                 token=hf_token,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 low_cpu_mem_usage=True
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    for prompt in prompts:
        if dataset == 'rotowire':
            messages = [{"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'atanasova':
            messages = [{"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'feng':
            messages = [{"role": "system", "content": f"{feng_instr}"}, {"role": "user", "content": prompt['prompt']}]

        encodeds = tokenizer.apply_chat_template(messages, truncation=True, padding=True, return_tensors="pt")

        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs,
                                        do_sample=True,
                                        max_new_tokens=all_params[1]['maximum_length'],
                                        temperature=all_params[1]['temperature'],
                                        top_p=all_params[1]['top_p'],
                                        pad_token_id=tokenizer.eos_token_id
                                        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        output = decoded[0]
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)
    
    return all_outputs
 

###################### Llama3 ###########################

def inference_llama(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"

    all_outputs = []

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    for prompt in prompts:
        if dataset == 'rotowire':
            messages = [
                    {"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        elif dataset == 'atanasova':
            messages = [
                    {"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        elif dataset == 'feng':
            messages = [
                    {"role": "system", "content": f"{feng_instr}"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            truncation=True, 
            padding=True,
            add_generation_prompt=True,
            return_tensors="pt").to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            eos_token_id=terminators,
            do_sample=True,
            max_new_tokens=all_params[2]['maximum_length'],
            temperature=all_params[2]['temperature'],
            top_p=all_params[2]['top_p'],
            pad_token_id=tokenizer.eos_token_id
            )
        
        response = outputs[0][input_ids.shape[-1]:]
        output = tokenizer.decode(response, skip_special_tokens=True)
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)
    
    return all_outputs

###################### Command-R-Plus ######################

def inference_commandrplus(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"
  
    all_outputs = []

    model_id = "CohereForAI/c4ai-command-r-plus-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    for prompt in prompts:

        if dataset == 'rotowire':
            messages = [{"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'atanasova':
            messages = [{"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'feng':
            messages = [{"role": "system", "content": f"{feng_instr}"}, {"role": "user", "content": prompt['prompt']}]

        input_ids = tokenizer.apply_chat_template(messages, truncation=True, padding=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=all_params[3]['maximum_length'],
            temperature=all_params[3]['temperature'],
            top_p=all_params[3]['top_p'],
            pad_token_id = tokenizer.eos_token_id
            )

        output = tokenizer.decode(gen_tokens[0])
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)
    
    return all_outputs

###################### Qwen2.5 ######################

def inference_qwen25(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"
  
    all_outputs = []

    model_id = "Qwen/Qwen2.5-7B-Instruct-1M"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    for prompt in prompts:
        if dataset == 'rotowire':
            messages = [{"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'atanasova':
            messages = [{"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'feng':
            messages = [{"role": "system", "content": f"{feng_instr}"}, {"role": "user", "content": prompt['prompt']}]

        text = tokenizer.apply_chat_template(
                messages, 
                truncation=True, 
                padding=True, 
                add_generation_prompt=True, 
                tokenize=False)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            max_new_tokens=all_params[5]['maximum_length'],
            temperature=all_params[5]['temperature'],
            top_p=all_params[5]['top_p'],
            pad_token_id = tokenizer.eos_token_id
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)
    
    return all_outputs

################## Deepseek LLaMa70 ##################

def inference_deepseek_llama3_70(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"

    all_outputs = []

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    for prompt in prompts:
        if dataset == 'rotowire':
            messages = [
                    {"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        elif dataset == 'atanasova':
            messages = [
                    {"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        elif dataset == 'feng':
            messages = [
                    {"role": "system", "content": f"{feng_instr}"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        input_text = tokenizer.apply_chat_template(
            messages,
            truncation=True, 
            padding=True,
            tokenize=False,
            add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").to(device)

        outputs = model.generate(
            input_ids["input_ids"],
            do_sample=True,
            max_new_tokens=all_params[2]['maximum_length'],
            temperature=all_params[2]['temperature'],
            top_p=all_params[2]['top_p'],
            pad_token_id=tokenizer.eos_token_id
            )
        
        output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)
    
    return all_outputs

###################### Qwen2_72 ####################

def inference_qwen2_72(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"
  
    all_outputs = []

    model_id = "Qwen/Qwen2-72B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", cache_dir="/add/your/folder/path/")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        cache_dir="/add/your/folder/path/"
    )

    for prompt in prompts:

        if dataset == 'rotowire':
            messages = [{"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'atanasova':
            messages = [{"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'feng':
            messages = [{"role": "system", "content": f"{feng_instr}"}, {"role": "user", "content": prompt['prompt']}]

        text = tokenizer.apply_chat_template(messages, truncation=True, padding=True, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            do_sample=True,
            max_new_tokens=all_params[5]['maximum_length'],
            temperature=all_params[5]['temperature'],
            top_p=all_params[5]['top_p'],
            pad_token_id = tokenizer.eos_token_id
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)
    
    return all_outputs

###################### LLaMa3_70 ######################

def inference_llama3_70(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"

    all_outputs = []

    model_id = "meta-llama/Llama-3.3-70B-Instruct" 

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", cache_dir="/add/your/folder/path/")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        cache_dir="/add/your/folder/path/"
    )

    for prompt in prompts:
        if dataset == 'rotowire':
            messages = [
                    {"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        elif dataset == 'atanasova':
            messages = [
                    {"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"},
                    {"role": "user", "content": prompt['prompt'] }
                    ]
        elif dataset == 'feng':
            messages = [
                    {"role": "system", "content": f"{feng_instr}"},
                    {"role": "user", "content": prompt['prompt'] }
            ]
        input_text = tokenizer.apply_chat_template(
            messages,
            truncation=True, 
            padding=True,
            tokenize=False,
            add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").to(device)

        outputs = model.generate(
            input_ids["input_ids"],
            do_sample=True,
            max_new_tokens=all_params[4]['maximum_length'],
            temperature=all_params[4]['temperature'],
            top_p=all_params[4]['top_p'],
            pad_token_id=tokenizer.eos_token_id
            )
        
        output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)
    
    return all_outputs

###################### Granite-3.2-8B ###########################

def inference_granite(prompts: list, dataset: str, seed=None) -> list:

    if seed is not None:
        SEED = seed
        set_seed(SEED)
    else:
        SEED = "no_seed"

    all_outputs = []

    model_id = "ibm-granite/granite-3.2-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", cache_dir="/add/your/folder/path/")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        cache_dir="/add/your/folder/path/"
    )

    for prompt in prompts:
        if dataset == 'rotowire':
            messages = [{"role": "system", "content": f"{gen_instr_rotowire}\n{task_head_rotowire}\n{task_instr_rotowire}\n{task_desc_rotowire}\n"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'atanasova':
            messages = [{"role": "system", "content": f"{instr_head}{task_1}{main_instr_1}{coverage_def}{main_instr_2}{example_1}{note}{example_2}"}, {"role": "user", "content": prompt['prompt']}]
        elif dataset == 'feng':
            messages = [{"role": "system", "content": f"{feng_instr}"}, {"role": "user", "content": prompt['prompt']}]

        model_inputs = tokenizer.apply_chat_template(messages, thinking=True, return_dict=True, truncation=True, padding=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            max_new_tokens=all_params[5]['maximum_length'],
            temperature=all_params[5]['temperature'],
            top_p=all_params[5]['top_p'],
            pad_token_id = tokenizer.eos_token_id
            )

        output = tokenizer.decode(generated_ids[0, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if dataset == 'rotowire':
            curr_entry = {"code": prompt['code'], "system1": prompt['system1'], "system2": prompt['system2'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'atanasova':
            curr_entry = {"id": prompt['id'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        elif dataset == 'feng':
            curr_entry = {"meeting_ID": prompt['meeting_ID'], "system_ID": prompt['system_ID'], "worker_id": prompt['worker_id'], "prompt": prompt['prompt'], "output": output}
        all_outputs.append(curr_entry)

    return all_outputs

##############################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = parser.add_argument_group("Select one of the following models: llama3, mistral, command-r-plus, qwen25, hunyuan, deepseek_llama70, qwen2_72, llama3_70, deepseek_r1, granite, internlm")
    model.add_argument("--model", action="store", required=True)
    seed = parser.add_argument_group("Select seed: 42, 1234, 1738, 198, 2338, 3104, 3677, 7188, 7582, 7736, 8578, 9191")
    parser.add_argument("--seed", type=int)
    dataset = parser.add_argument_group("Select one of the following dataset: rotowire, atanasova, feng")
    dataset.add_argument("--dataset", action="store", required=True)
    criteria = parser.add_argument_group("Select one of the following models: coherence, repetition, grammaticality, other")
    criteria.add_argument("--criteria", action="store", required=True)
    args = parser.parse_args()


    if args.dataset == 'rotowire':
        prompt = rotowire_templates(args.model, args.criterion)
    elif args.dataset == 'atanasova':
        prompt = atanasova_templates(args.model)
    elif args.dataset == 'feng':
        prompt = feng_templates(args.model)

    if args.model == 'llama3':
        if args.dataset == 'rotowire':
            result = inference_llama(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_llama3_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_llama(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_llama3_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_llama(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_llama3_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    elif args.model == 'mistral':
        if args.dataset == 'rotowire':
            result = inference_mistral(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_mistral_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_mistral(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_mistral_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_mistral(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_mistral_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    elif args.model == 'command-r-plus':
        if args.dataset == 'rotowire':
            result = inference_commandrplus(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_commandrplus_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_commandrplus(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_commandrplus_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_commandrplus(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_commandrplus_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    elif args.model == 'qwen25':
        if args.dataset == 'rotowire':
            result = inference_qwen25(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_qwen25_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_qwen25(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_qwen25_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_qwen25(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_qwen25_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    elif args.model == 'deepseek_llama70':
        if args.dataset == 'rotowire':
            result = inference_deepseek_llama3_70(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_deepseek_llama3_70_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_deepseek_llama3_70(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_deepseek_llama3_70_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_deepseek_llama3_70(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_deepseek_llama3_70_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    elif args.model == 'qwen2_72':
        if args.dataset == 'rotowire':
            result = inference_qwen2_72(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_qwen2_72_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_qwen2_72(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_qwen2_72_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_qwen2_72(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_qwen2_72_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    elif args.model == 'llama3_70':
        if args.dataset == 'rotowire':
            result = inference_llama3_70(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_llama3_70_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_llama3_70(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_llama3_70_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_llama3_70(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_llama3_70_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    elif args.model == 'granite':
        if args.dataset == 'rotowire':
            result = inference_granite(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.criterion}_{args.dataset}_granite_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'atanasova':
            result = inference_granite(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_granite_out.json', 'w') as f:
                json.dump(result, f, indent=4)
        elif args.dataset == 'feng':
            result = inference_granite(prompt, seed=args.seed, dataset=args.dataset)
            with open(f'{args.seed}_{args.dataset}_granite_out.json', 'w') as f:
                json.dump(result, f, indent=4)
    else:
        raise ValueError("Please enter a valid model name.")