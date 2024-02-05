import json

def read_parameters(json_file):
    with open(json_file, 'r') as f:
        parameters = json.load(f)
    return parameters