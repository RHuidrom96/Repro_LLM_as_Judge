#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================================================
# Created By  : Rudali Huidrom
# Created Date: Fri Jan 31 2025
# ================================================================================================
"""
    The module lists out the required experiment parameters.
"""
# =================================================================================================
# Imports
# =================================================================================================

def params() -> list:

    all_params = [
        {
            'gen_instr_rotowire': "You are a native speaker of English or a near-native speaker who can comfortably comprehend  summary of NBA basketball games written in English.\n",
            'task_head_rotowire': "Evaluate Sports Summaries of (NBA) basketball games\n",
            'task_instr_rotowire': "Your task is to read two short texts which have been produced by different automatic systems. These systems typically take a large table as input which contains statistics of a basketball game  and  produce a document which  summarizes the table in natural langauge (e.g., talks about what happened in the game, who scored, who won and so on).  Please read the two summaries carefully and judge how good each is according to the following criterion:\n",
            'task_desc_rotowire': "This task contains validation instances (for which answers are known) that will be used for an automatic quality assessment of submissions. Therefore, please read the summaries carefully.\n",
            'summaries': "Summaries\n\n",
            'sys_summaries': "System Summaries\n",
            'A': "A: ",
            'B': "B: ",
            'rank_criteria': "Ranking Criteria\n",
            'Coherence': "Coherence:  How coherent is the summary? How natural is the ordering of the facts? The summary should be well structured and well organized and have a natural ordering of the facts.\n",
            'Repetition': "Avoid repetition:  Which summary is better at avoiding unnecessary repetition? Unnecessary repetition might take the form of whole sentences that are repeated, or repeated facts, or the repeated use of a phrase\n",
            'Grammaticality': "Grammaticality: Are the sentences grammatical and well-formed? The summary sentences should be grammatically correct. You should not rate the document as whole but rather whether the sentences could be written by a native speaker or by someone who is a learner and makes mistakes. Choose the more grammatical summary.\n",
            'answer': "Answers\n",
            'best': "Best: ",
            'worst': "Worst: ",
            'analysis': "Analysis\n",
            'instr_head': "Instructions to Participants\n",
            'task_1': "Task 1\n\n",
            'main_instr_1': "Evaluate the outputs of the three different systems.\nEach row contains a claim, its veracity label and three different explanations/reasons for the veracity label.\nYour task is to rank the three different explanations with the ranks 1, 2, 3, (first, second and third place) according to the following criterion:\n\n",
            'coverage_def': "Coverage. The explanation contains important, salient information and doesn't miss any important points that contribute to the fact-check.\n\n",
            'main_instr_2': "You are presented with a column for the criterion.\n\nThe nth column for each criterion should contain the rank for the nth justification.\n\n",
            'example_1': "Example:\nFor a particular claim, you find that justification3 was the best w.r.t. coverage, then you put 1 in the third column.\n\n",
            'note': "Note:  If there is a tie and two justifications seem to have the same rank, then assign the same rank to them.\n",
            'example_2': "Example:\nIf you think that justification1 and justification3 were both the best w.r.t. coverage, then the ranks for coverage should be: 1 2 1\n",
            'claim': "Claim: ",
            'label': "Label: ",
            'justification_1': "Justification 1: ",
            'justification_2': "Justification 2: ",
            'justification_3': "Justification 3: ",
            'feng_instr': "Hi everyone, thanks for helping to do the human evaluation, there is one dataset, AMI, long conversation, 10 data items in total. Please mark each based on the indicator: Informativeness, ranging from 1 to 5, 1 being the worst 5 being the best. Informativeness measures whether the abstract contains the key information from the original conversation.  Everyone’s document is the same, a total of 4 people will evaluate the same data, and we will then calculate the kappa value to measure the consistency.",
            'summary': "Summary: "
        },
        {
            'model': 'mistral',
            'BOS': '<s>',
            'EOS': '</s>',
            'start_token': '[INST]',
            'end_token': '[/INST]',
            'temperature': 0.001,
            'maximum_length': 1500,
            'top_p': 1
        },
        {
            'model': 'llama3',
            'BOS': '<|begin_of_text|>',
            'EOS': '<|end_of_text|>',
            'start_token': '{{',
            'end_token': '}}',
            'temperature': 0.001,
            'maximum_length': 1500,
            'top_p': 1
        },
        {
            'model': 'command-r-plus',
            'Instruction': '## Instructions\n',
            'Input': '## Input\n',
            'Output': '## Output\n',
            'Criterion': '## Criterion\n',
            'temperature': 0.001,
            'maximum_length': 1500,
            'top_p': 1
        },
        {
            'model': 'llama70',
            'BOS': '<|begin_of_sentence|>',
            'EOS': '<|end_of_sentence|>',
            'user': '<|User|>',
            'assistant': '<|Assistant|>',
            'temperature': 0.001,
            'maximum_length': 1500,
            'top_p': 1
        },
        {
            'model': 'qwen2.5',
            'temperature': 0.001,
            'maximum_length': 1500,
            'top_p': 1
        }
    ]

    return all_params