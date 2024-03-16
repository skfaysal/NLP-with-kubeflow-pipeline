import os
import json
import yaml

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_directory_if_not_exists(directory):
    """
    Checks if a directory exists, and if not, creates it.

    Args:
    - directory (str): The path of the directory to check and potentially create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_label_mapping(label_mapping, output_dir, filename='label_mapping.json'):
    """
    Saves the label mapping to a JSON file.

    Args:
    - label_mapping (dict): The label mapping to be saved.
    - output_dir (str): The directory where the file will be saved.
    - filename (str): The name of the file. Default is 'label_mapping.json'.
    """
    create_directory_if_not_exists(output_dir)
    
    label_mapping_path = os.path.join(output_dir, filename)
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f)


def load_json(label_mapping_path):
    """
    Loads label mapping from a JSON file.

    Args:
    - label_mapping_path (str): Path to the label mapping JSON file.

    Returns:
    - dict: A dictionary containing the label mapping.
    """
    with open(label_mapping_path, 'r') as f:
        json_file = json.load(f)
    return json_file
