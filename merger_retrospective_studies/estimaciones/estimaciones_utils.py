import random
import json
import os
import numpy as np


def generate_random_sparse_array(shape, start_range, end_range, k):
    """
    Generates a multidimensional array with `k` random elements and the rest set to zero.

    Parameters:
    shape (tuple): Shape of the array, e.g., (3, 4) for a 3x4 array.
    start_range (int): Start of the range (inclusive).
    end_range (int): End of the range (inclusive).
    k (int): Number of elements to assign random values.

    Returns:
    list: Multidimensional sparse array.
    """
    if any(dim <= 0 for dim in shape):
        raise ValueError("All dimensions must be positive integers.")
    if start_range > end_range:
        raise ValueError("Start of the range must be less than or equal to the end of the range.")
    if k > (total_elements := int(eval('*'.join(map(str, shape))))):
        raise ValueError("k cannot exceed the total number of elements in the array.")

    # Create a flattened version of the array with k random values and the rest set to zero
    flat_array = [random.uniform(start_range, end_range) if i < k else 0 for i in range(total_elements)]
    random.shuffle(flat_array)

    # Function to reshape the flat array into the desired multidimensional shape
    def reshape(flat, shape):
        if len(shape) == 1:
            return flat[:shape[0]]
        size = shape[0]
        sub_size = total_elements // size
        return [reshape(flat[i * sub_size:(i + 1) * sub_size], shape[1:]) for i in range(size)]

    return reshape(flat_array, shape)


def generate_random_array(shape, start_range, end_range):
    """
    Generates a multidimensional array of random numbers.

    Parameters:
    shape (tuple): Shape of the array, e.g., (3, 4) for a 3x4 array.
    start_range (int): Start of the range (inclusive).
    end_range (int): End of the range (inclusive).

    Returns:
    list: Multidimensional array of random numbers.
    """
    if any(dim <= 0 for dim in shape):
        raise ValueError("All dimensions must be positive integers.")
    if start_range > end_range:
        raise ValueError("Start of the range must be less than or equal to the end of the range.")
    
    def create_array(shape):
        if len(shape) == 1:  # Base case: 1D array
            return [random.uniform(start_range, end_range) for _ in range(shape[0])]
        # Recursive case: Nested arrays
        return [create_array(shape[1:]) for _ in range(shape[0])]

    return create_array(shape)


def generate_random_floats(x, start_range, end_range):
    """
    Generates a list of random floats.

    Parameters:
    x (int): Number of random floats to generate.
    start_range (int): Start of the range (inclusive).
    end_range (int): End of the range (inclusive).

    Returns:
    list: List of random floats.
    """
    if x <= 0:
        raise ValueError("The number of random floats must be a positive integer.")
    if start_range > end_range:
        raise ValueError("Start of the range must be less than or equal to the end of the range.")
    
    return [random.uniform(start_range, end_range) for _ in range(x)]


def count_non_zero_one_strings(input_string):
    """
    Count the number of substrings different from "0" or "1" in a given string
    split by the '+' delimiter.

    Args:
        input_string (str): The input string to process.

    Returns:
        int: The count of substrings different from "0" and "1".
    """
    # Split the string by '+' and remove whitespace
    elements = [item.strip() for item in input_string.split('+')]
    
    # Count elements that are not equal to "0" or "1"
    return sum(1 for element in elements if element not in {"0", "1"})


def classify_dir_elements(obj):
    """
    Classify elements from dir(obj) into attributes and methods.

    Args:
        obj: The object to inspect.

    Returns:
        dict: A dictionary with two keys: 'attributes' and 'methods'.
              - 'attributes' contains a list of non-callable elements.
              - 'methods' contains a list of callable elements.
    """
    # Get all elements from dir()
    elements = dir(obj)

    # Classify elements
    attributes = []
    methods = []
    for item in elements:
        element = getattr(obj, item)
        if callable(element):
            methods.append(item)
        else:
            attributes.append(item)

    return {"attributes": attributes, "methods": methods}


def save_dict_json(dictionary, folder_path, file_name):
    """
    Save a dictionary as a JSON file in the specified folder, handling NumPy arrays.

    Args:
        dictionary (dict): The dictionary to save.
        folder_path (str): The folder path where the file will be saved.
        file_name (str): The name of the file (e.g., 'data.json').
    """
    # Convert NumPy arrays to lists
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Full file path
    file_path = os.path.join(folder_path, file_name)

    # Save the dictionary
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, indent=4, default=convert_ndarray)
    print(f"Dictionary saved to {file_path}")
