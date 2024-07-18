import json 
import os
def add_id(data):
    for index , intent in enumerate(data['intents'],start=1):
        intent['id'] = index
    return data

current_dir = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(current_dir,'data.json')

with open(json_path,'r') as file:
    json_data = json.load(file)
    
json_data = add_id(json_data)

with open(json_path,'w') as file:
    json.dump(json_data,file,indent = 4)