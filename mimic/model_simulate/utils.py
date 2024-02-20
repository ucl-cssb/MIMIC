import json
import os


def read_parameters(json_file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, json_file)
    with open(file_path, 'r') as f:
        parameters = json.load(f)
    return parameters
