import json

def get_dummy_data():
    dummy_file = open("././dummy.json")
    dummy_dict = json.load(dummy_file)
    return dummy_dict