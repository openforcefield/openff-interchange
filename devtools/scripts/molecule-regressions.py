import json

file_name = "regression_tests/differences.json"
input_path = "regression_tests/small-molecule/input-topologies.json"

differences = dict()

try:
    with open(file_name) as file:
        differences.update(json.loads(file.read()))

except FileNotFoundError:
    print(f"File {file_name} not found.")

if len(differences) > 0:
    with open(input_path) as file:
        inputs = json.loads(file.read())

    for molecule in differences.keys():
        print(inputs[int(molecule)])

    exit(1)
