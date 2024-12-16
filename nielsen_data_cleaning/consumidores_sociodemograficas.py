import os
import pandas as pd
import chardet
import re
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
PRODUCT_MODULE = 7460
NROWS = 20000000
YEAR = 2014
WEEKS = [20140125, 20140201]


def read_non_whitespace_lines(filename):
    """
    Reads a file and returns a list of lines that do not start with any type of whitespace.

    :param filename: The path to the file to be read.
    :return: A list containing the lines that do not start with whitespace.
    """
    non_whitespace_lines = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.lstrip() == line:
                # This line starts with whitespace, skip it
                continue
            non_whitespace_lines.append(line.strip())

    return non_whitespace_lines


def read_file_with_guessed_encoding(file_path):
    # First, guess the encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    encoding_guessed = chardet.detect(raw_data).get('encoding')
    print(encoding_guessed)

    # Now, read the file with the detected encoding
    if encoding_guessed:
        try:
            with open(file_path, 'r', encoding=encoding_guessed) as file:
                print(f"File content with guessed encoding ({encoding_guessed}):")
                print(len(file.read()))
                return encoding_guessed

        except UnicodeDecodeError:
            print(f"Could not decode the file with the guessed encoding ({encoding_guessed}).")
    else:
        print("Could not guess the encoding of the file.")
        return None


def process_string_lines(input_string):
    """
    Processes an input string line by line.

    :param input_string: The string to be processed.
    :return: A list of processed lines.
    """
    processed_lines = []
    for line in input_string.splitlines():
        # Process each line here. For now, we'll just add it to the list.
        processed_lines.append(line)

    return processed_lines


def process_non_whitespace_lines(input_string):
    """
    Processes an input string line by line and adds to a list only the lines that do not start with whitespace.

    :param input_string: The string to be processed.
    :return: A list of lines that do not start with whitespace.
    """
    non_whitespace_lines = []
    for line in input_string.splitlines():
        # if line and not line.lstrip() == line:
        #     # This line starts with whitespace, skip it
        #     continue
        # non_whitespace_lines.append(line)
        if line and line[0].isupper():
            non_whitespace_lines.append(line)
    return non_whitespace_lines


def extract_number(s):
    match = re.search(r'\b\d+\b', s)
    if match:
        return match.group()
    else:
        return "No number found"


def extract_first_numbers(strings):
    numbers = []
    for s in strings:
        match = re.search(r'\b\d+\b', s)
        if match:
            numbers.append(int(match.group()))
    return numbers


def extract_number_after_first_tabs(input_string):
    """
    Extracts the number that appears after the first sequence of tabs in an input string.

    :param input_string: The string to be processed.
    :return: The extracted number, or None if no number is found.
    """

    # Split the string on tabs
    parts = input_string.split('\t')

    # Find the first non-empty part after the first tab
    for part in parts[1:]:
        if part:
            # Extract the number from the part
            match = re.search(r'\b\d+\b', part)
            if match:
                return int(match.group())

    # No number found
    return None


def extract_numbers_after_first_tabs(strings):
    """
    Extracts the numbers that appear after the first sequence of tabs in a list of input strings.

    :param strings: The list of strings to be processed.
    :return: A list of extracted numbers.
    """
    numbers = []
    for s in strings:
        number = extract_number_after_first_tabs(s)
        if number:
            numbers.append(number)

    return numbers


# prompt: how to check if a string is all upercase

def is_uppercase(string):
    """
    Checks if a given string is all uppercase.

    :param string: The string to be checked.
    :return: True if the string is all uppercase, False otherwise.
    """

    # Check if the string is empty
    if not string:
        return False

    # Check if every character in the string is uppercase
    for char in string:
        if not char.isupper():
            return False

    # All characters are uppercase
    return True


def append_uppercase_tabbed_elements(input_list, output_list):
    """
    Appends the elements of a list to another list if the element's first word is all in uppercase letters and it is followed by one or more tabs.

    :param input_list: The list of elements to be processed.
    :param output_list: The list to which the elements will be appended.
    """
    for element in input_list:
        print(element)
        # Check if the first word is all in uppercase letters
        if element.split()[0].isupper():
            # Check if the element is followed by one or more tabs
            if re.search(r'\t+', element):
                output_list.append(element)


def extract_number_after_first_tabs(input_string):
    """
    Extracts the number that appears after the first sequence of tabs in an input string.

    :param input_string: The string to be processed.
    :return: The extracted number, or None if no number is found.
    """

    # Split the string on tabs
    parts = input_string.split('\t')

    # Find the first non-empty part after the first tab
    for part in parts[1:]:
        if part:
            # Extract the number from the part
            match = re.search(r'\b\d+\b', part)
            if match:
                return int(match.group())

    # No number found
    return None


def update_dict_with_string(input_string, input_dict):
    elements = input_string.split('\t')

    # Check if the first element is an uppercase word
    if elements and elements[0].isupper():
        key = elements[0]

        # Check for tabs followed by a number
        for element in elements[1:]:
            if element.isdigit():
                input_dict[key] = int(element)
                break


def process_file(file_path):
    # Read the content of the file
    with open(file_path, 'r', encoding = encoding_guessed ) as file:
        lines = file.readlines()

    # Process each line
    new_lines = []
    counter = 1
    for line in lines:
        words = line.split()
        if words and words[0].isupper():
            new_lines.append(f"{line.strip()}{counter}\n")
            counter += 1
        else:
            new_lines.append(line)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)


def process_file(file_path):
    with open(file_path, 'r', encoding=encoding_guessed) as file:
        lines = file.readlines()

    # Process each line
    new_lines = []
    counter = 1
    for line in lines:
        # Split the line by tabs and check the conditions
        parts = line.split()
        if len(parts) < 2:
            continue
        elif parts[0].isupper() and parts[1].strip().isdigit():
            new_line = f"{counter}\t{line}"
            counter += 1
        else:
            new_line = line
        new_lines.append(new_line)
    print(counter)

    # Write the processed lines back to the file
    with open(file_path+ '_modified', 'w') as file:
        file.writelines(new_lines)


def process_file(file_path):
    with open(file_path, 'r', encoding=encoding_guessed) as file:
        lines = file.readlines()

    # Process each line
    output_dict = {}
    new_lines = []
    counter = 0
    for line in lines:
        # Split the line by tabs and check the conditions
        parts = line.split()
        if len(parts) < 2:
            continue
        elif parts[0].isupper() and parts[1].strip().isdigit():
            new_line = f"{counter}\t{line}"
            populate_dictionary(output_dict, parts[0], parts[1])
            # output_dict[parts[0]] = parts[1]
            counter += 1
        else:
            new_line = line
        new_lines.append(new_line)
    print(counter)

    # Write the processed lines back to the file
    with open(file_path+ '_modified', 'w') as file:
        file.writelines(new_lines)

    return output_dict


def populate_dictionary(dictionary, key, value):
    """
    Populates a dictionary with a key and a value.

    :param dictionary: The dictionary to be populated.
    :param key: The key to be added to the dictionary.
    """

    # Check if the key is already in the dictionary
    if key in dictionary:
        # Get the current count of keys with the same root name
        count = len([k for k in dictionary.keys() if k.startswith(key)])

        # Add the new key with the incremented count
        dictionary[f"{key}_{count + 1}"] = int(value)
    else:
        # Add the new key with a count of 1
        dictionary[key] = int(value)


# prompt: Given a regex pattern, find the pattern in a given string and store all the matches in a list.

def find_pattern(pattern, string):
    """
    Finds all occurrences of a pattern in a string and stores them in a list.

    :param pattern: The regular expression pattern to search for.
    :param string: The string to search in.
    :return: A list of all matches found.
    """

    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Find all matches of the pattern in the string
    matches = regex.search(string)

    # Return the list of matches
    return matches


def find_and_append_pattern(pattern, line, match_list):
    """
    Finds a pattern in a line of text and appends the match to a list.

    :param pattern: The regular expression pattern to search for.
    :param line: The line of text to search in.
    :param match_list: The list to append the match to.
    """

    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Find all matches of the pattern in the line
    matches = regex.findall(line)

    # Append the matches to the list
    match_list.extend(matches)

    # Example usage:
    pattern = r'\d+'  # Pattern to find all digits
    line = "The quick brown fox jumps over the lazy dog 1231241"
    match_list = []

    find_and_append_pattern(pattern, line, match_list)

    print(match_list)  # Output: ['123456']


def find_and_append_pattern(pattern, line):
    """
    Finds a pattern in a line of text and appends the match to a list.

    :param pattern: The regular expression pattern to search for.
    :param line: The line of text to search in.
    :param match_list: The list to append the match to.
    """

    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Find all matches of the pattern in the line
    matches = regex.findall(line)

    # return the matches
    return matches


def process_file_regex(file_path, pattern):
    with open(file_path, 'r', encoding=encoding_guessed) as file:
        lines = file.readlines()

    # Process each line
    match_list = []
    output_dict = {}
    new_lines = []
    counter = 1
    for line in lines:
        match_list.append(find_and_append_pattern(pattern, line))

    return match_list


def process_string_regex(string, pattern):
    # Process each line
    match_list = []
    output_dict = {}
    new_lines = []
    counter = 1
    match_list.append(find_pattern(pattern, line))

    return match_list


def count_values(variable):
    """
    Creates a dictionary that returns keys equal to the value of the variable and values equal to the number of times the value appears in the variable.

    :param variable: The variable to be counted.
    :return: A dictionary with keys equal to the value of the variable and values equal to the number of times the value appears in the variable.
    """

    # Create an empty dictionary
    value_counts = {}

    # Iterate over the variable
    for value in variable:
        # Check if the value is already in the dictionary
        if value in value_counts:
        # Increment the count for the value
            value_counts[value] += 1
        else:
        # Add the value to the dictionary with a count of 1
            value_counts[value] = 1

    # Return the dictionary
    return value_counts


def display_tree(root):
    """
    Displays the directory tree rooted at the given path.

    :param root: The path to the root directory.
    """

    # Get the list of directories and files in the root directory
    directories = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    # Print the root directory
    print(root)

    # Iterate over the directories and files
    for directory in directories:
        # Recursively display the subtree rooted at the directory
        display_tree(os.path.join(root, directory))

    for file in files:
        # Print the file
        print(f"\t{file}")


def calculate_intersection_percentage(set1, set2):
    """
    Calculates the percentage of numbers in the first set that are also in the second set.

    :param set1: The first set.
    :param set2: The second set.
    :return: The percentage of numbers in the first set that are also in the second set.
    """

    # Calculate the intersection of the two sets
    intersection = set1.intersection(set2)

    # Calculate the size of the intersection
    intersection_size = len(intersection)

    # Calculate the size of the first set
    set1_size = len(set1)

    # Calculate the percentage of numbers in the first set that are also in the second set
    percentage = (intersection_size / set1_size) * 100

    # Return the percentage
    return percentage


def absolute_values_dataframe(df):
    """
    Takes a pandas dataframe and returns the same dataframe but each column is the absolute value of the original one.

    :param df: The pandas dataframe.
    :return: The pandas dataframe with absolute values.
    """

    # Create a new dataframe with the same columns as the original dataframe
    df_absolute = pd.DataFrame(columns=df.columns)

    # Iterate over each column in the original dataframe
    for column in df.columns:
        # Calculate the absolute values of the column
        absolute_values = abs(df[column])

        # Add the absolute values to the new dataframe
        df_absolute[column] = absolute_values

    # Return the new dataframe
    return df_absolute


def get_random_samples_by_code(df, codes, x):
    """
    Returns a DataFrame with `x` number of random samples of rows for each code in the provided list.

    Parameters:
    df (pd.DataFrame): The original DataFrame to sample from.
    codes (list): A list of codes to filter the DataFrame.
    x (int): The number of random samples to return for each code.
    
    Returns:
    pd.DataFrame: A new DataFrame containing `x` random rows for each code.
    """
    codes = set(codes)
    # Filter DataFrame for the specified codes
    filtered_df = df[df['GESTFIPS'].isin(codes)]
    
    # Sample `x` rows for each code
    sampled_df = filtered_df.groupby('GESTFIPS').apply(lambda group: group.sample(n=x, replace=True)).reset_index(drop=True)
    
    return sampled_df


def impute_demographic_data(demographic_sample):
    knn_imputer = KNNImputer(n_neighbors=2)
    demographic_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(demographic_sample[['HEFAMINC',
                                                                                                'PRTAGE',
                                                                                                'HRNUMHOU',
                                                                                                'PTDTRACE',
                                                                                                'PEEDUCA']]),
                                                    columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 
                                                            'ptdtrace_imputed', 'peeduca_imputed'])
    return demographic_sample_knn_imputed


def add_random_nodes(demographic_sample, mean=0, std_dev=1, num_nodes=5):
    """
    Adds columns with random normal values to the DataFrame.

    :param demographic_sample: The DataFrame to which the columns will be added.
    :param mean: The mean of the normal distribution.
    :param std_dev: The standard deviation of the normal distribution.
    :param num_nodes: The number of columns to add.
    :return: The DataFrame with the new columns added.
    """
    size = len(demographic_sample)
    for i in range(num_nodes):
        demographic_sample[f'nodes{i}'] = np.random.normal(loc=mean, scale=std_dev, size=size)
    return demographic_sample


def main():
    product_data = pd.read_csv('6.product_data_postinst_Reynolds_Lorillard_retailer_2024-12-04 18:37:37.916473.csv')
    encoding_guessed = read_file_with_guessed_encoding('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/otros/January_2014_Record_Layout.txt')
    output = process_file('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/otros/January_2014_Record_Layout.txt')
    agent_data_pop = pd.read_fwf('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/apr14pub.dat', widths= [int(elem) for elem in output.values()] )
    column_names = output.keys()
    agent_data_pop.columns = column_names
    agent_data_pop=agent_data_pop[agent_data_pop['GTCO']!=0]
    agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']
    agent_data_pop.reset_index(inplace=True, drop=True)

    product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})
    
    demographic_sample = get_random_samples_by_code(agent_data_pop, product_data['GESTFIPS'], 200)[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]
    demographic_sample.replace(-1, np.nan, inplace=True)

    knn_imputer = KNNImputer(n_neighbors=2)
    demographic_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(demographic_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]),
                                columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 
                                        'ptdtrace_imputed', 'peeduca_imputed'])

    grouped = demographic_sample.groupby('GESTFIPS').size()

    demographic_sample['weights'] = demographic_sample['GESTFIPS'].map(1 / grouped)
    demographic_sample = pd.concat([demographic_sample[['FIPS', 'GESTFIPS','weights']],demographic_sample_knn_imputed], axis=1)
    demographic_sample = add_random_nodes(demographic_sample)

    demographic_sample = demographic_sample[['FIPS', 'GESTFIPS', 'weights',
                                            'nodes0', 'nodes1', 'nodes2', 'nodes3','nodes4',
                                            'hefaminc_imputed', 'prtage_imputed','hrnumhou_imputed', 
                                            'ptdtrace_imputed', 'peeduca_imputed']]
    
    agent_data = pd.merge(product_data[['market_ids', 'market_ids_string', 'GESTFIPS']].drop_duplicates(),
                                      demographic_sample, 
                                      how='inner', 
                                      left_on='GESTFIPS',
                                      right_on='GESTFIPS')



if __name__ == '__main__':
    main()