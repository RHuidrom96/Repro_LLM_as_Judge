#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================================================
# Created By  : Rudali Huidrom
# Created Date: Fri Jan 31 2025
# ================================================================================================
"""
    The module has been build for bringing in all the preprocessed data together.
"""
# =================================================================================================
# Imports
# =================================================================================================
import os
import json
import argparse
from collections import defaultdict
from preprocess import *

def rotowire_data() -> list:
    project_dirpath = os.path.join('/', 'add', 'your', 'folder', 'path')
    rotowire_filepath = os.path.join(project_dirpath, 'rotowire.csv')

    rotowire_list = extract_rotowire(rotowire_filepath)

    return rotowire_list

def atanasova_data() -> list:
    project_dirpath = os.path.join('/', 'add', 'your', 'folder', 'path')
    atanasova_filepath = os.path.join(project_dirpath, 'atanasova.xlsx')

    atanasova_list = extract_atanasova(atanasova_filepath)

    return atanasova_list

def feng_data() -> list:
    project_dirpath = os.path.join('/', 'add', 'your', 'folder', 'path')
    feng_filepath = os.path.join(project_dirpath, 'feng.xlsx')

    feng_list = extract_feng(feng_filepath)

    return feng_list