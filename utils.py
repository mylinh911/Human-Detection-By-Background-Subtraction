import yaml
import json

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def write_json(file,data):
    with open(file,'w') as f:
        json.dump(data,f,indent = 4)

def get_config(yaml_file='./config.yml'):
	with open(yaml_file, 'r') as file:
		cfgs = yaml.load(file, Loader=yaml.FullLoader)
	return cfgs

def get_label(path):
    if path.split('\\')[-2] == 'Human':
        return 1
    else:
        return 0

# if __name__ == '__main__':
#     cfgs = get_config()
    
#     get_json_data(root_name, type_data_name, fold_name, json_name)
#     print(cfgs)
