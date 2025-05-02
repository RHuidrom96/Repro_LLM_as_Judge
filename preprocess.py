#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================================================
# Created By  : Rudali Huidrom
# Created Date: Fri Jan 31 2025
# ================================================================================================
"""
    The module has been build to summaries from rotowire reprohum evaluation samples.
"""
# =================================================================================================
# Imports
# =================================================================================================

import os
import json
import pandas as pd

def extract_rotowire(file_path: str) -> list:
    df = pd.read_csv(file_path)
    dict_list = df.to_dict(orient='records')
    
    return dict_list

def extract_atanasova(file_path: str) -> list:
    df = pd.read_excel(file_path, sheet_name=0)
    dict_list = df.to_dict(orient='records')

    return dict_list

def extract_feng(file_path: str) -> list:
    df = pd.read_excel(file_path, sheet_name=0)
    dict_list = df.to_dict(orient='records')

    return dict_list