import yaml

def read_yaml_file(file_path):
    """
    Reads a YAML file and returns its contents as a Python dictionary.

    :param file_path: Path to the YAML file
    :return: Dictionary containing the YAML file contents
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")