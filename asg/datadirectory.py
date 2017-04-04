import os

with open("data.directory.txt") as f:
    DATA_DIRECTORY = f.read().strip()

if not os.path.isdir(DATA_DIRECTORY):
    raise Exception(
        "Data directory: {} does not exist.".format(DATA_DIRECTORY))
