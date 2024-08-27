import json


def update_json(filename, key, val):
    """
    Update a .json file

    :param filename: str
        Path of .json file to update
    :param key:
        Key of entry to update
    :param val:
        Value of entry to update

    :return:
    """

    with open(filename, "r") as file:
        file_data = json.load(file)

    file_data[key] = val

    with open(filename, "w") as file:
        json.dump(file_data, file, indent=4)
