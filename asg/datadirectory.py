# -*- coding: utf-8 -*-
"""Read the local data.directory.txt to point to local VIST dataset"""

import os

with open("data.directory.txt") as f:
    data_directory = f.read().strip()

if not os.path.isdir(data_directory):
    raise Exception(
        "Data directory: {} does not exist.".format(data_directory))
