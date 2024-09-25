import pandas as pd
import numpy as np
import random
import string
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

def generate_random_password(length):
    """
    Generate a random password of a given length.
    
    Args:
        length (int): The length of the password.
    
    Returns:
        str: The generated password.
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_random_password_batch(batch_size, existing_passwords, max_length):
    """
    Generate a batch of random passwords.
    
    Args:
        batch_size (int): The number of passwords to generate.
        existing_passwords (set): A set of existing passwords to avoid duplicates.
        max_length (int): The maximum length of the passwords.
    
    Returns:
        list: A list of generated passwords.
    """
    batch = []
    while len(batch) < batch_size:
        length = random.randint(1, max_length)
        random_password = generate_random_password(length)
        if random_password not in existing_passwords:
            batch.append(random_password)
    return batch

def fetch_random_word_batch(batch_size, max_length):
    """
    Fetch a batch of random words from an API.
    
    Args:
        batch_size (int): The number of words to fetch.
        max_length (int): The maximum length of the words.
    
    Returns:
        list: A list of fetched words.
    
    Raises:
        RuntimeError: If the API request fails.
    """
    try:
        response = requests.get(f'https://random-word.ryanrk.com/api/en/word/random/{batch_size}?maxLength={max_length}')
        response.raise_for_status()
        words = response.json()
        # Turn the first letter to a capital letter in 25% of the cases
        for i in range(len(words)):
            if random.random() < 0.25:
                words[i] = words[i].capitalize()
        return words
    except requests.RequestException as e:
        raise RuntimeError("Failed to fetch random words from the API. The API might be down.") from e

def fetch_random_word_plus_int_batch(batch_size, max_length):
    """
    Fetch a batch of random words from an API and append 1-9 random digits.
    
    Args:
        batch_size (int): The number of words to fetch.
        max_length (int): The maximum length of the words.
    
    Returns:
        list: A list of modified words.
    """
    words = fetch_random_word_batch(batch_size, max_length)
    modified_words = [word + ''.join(random.choices(string.digits, k=random.randint(1, 9))) for word in words]
    return modified_words

def replace_with_special_chars(word):
    """
    Replace characters in a word with special characters.
    
    Args:
        word (str): The word to modify.
    
    Returns:
        str: The modified word.
    """
    replacements = {'s': '$', 'o': '0', 'a': '@', 'i': '!', 'e': '3', 'c' : 'ç', 't' : '†', 'l' : '£', 'g' : '9', 'b' : 'ß', 'u' : 'µ'}
    chars = list(word)
    num_replacements = max(1, len(chars) // 2)
    indices = random.sample(range(len(chars)), num_replacements)
    
    for i in indices:
        if chars[i] in replacements:
            chars[i] = replacements[chars[i]]
    
    # Turn some random characters to capital letters
    num_capitals = max(1, len(chars) // 4)
    capital_indices = random.sample(range(len(chars)), num_capitals)
    
    for i in capital_indices:
        chars[i] = chars[i].upper()
    
    return ''.join(chars)

def fetch_random_word_plus_int_and_special_char_batch(batch_size, max_length):
    """
    Fetch a batch of random words from an API, append 1-9 random digits, and replace some characters with special characters.
    
    Args:
        batch_size (int): The number of words to fetch.
        max_length (int): The maximum length of the words.
    
    Returns:
        list: A list of modified words.
    """
    words = fetch_random_word_batch(batch_size, max_length)
    modified_words = [replace_with_special_chars(word) + ''.join(random.choices(string.digits, k=random.randint(1, 9))) for word in words]
    return modified_words

def generate_negative_samples(num_samples, existing_passwords, max_length):
    """
    Generate negative samples (random passwords not in the cracked list).
    
    Args:
        num_samples (int): The number of negative samples to generate.
        existing_passwords (set): A set of existing passwords to avoid duplicates.
        max_length (int): The maximum length of the passwords.
    
    Returns:
        list: A list of generated negative samples.
    """
    batch_size = 100
    num_batches = math.ceil(num_samples / 400)
    negative_samples = []

    with ThreadPoolExecutor() as executor:
        futures = []

        for _ in range(num_batches):
            # 1/4 random strings
            futures.append(executor.submit(generate_random_password_batch, batch_size, existing_passwords, max_length))

            # 1/4 words from the Random Word API
            futures.append(executor.submit(fetch_random_word_batch, batch_size, max_length))

            # 1/4 words from the Random Word API followed by 1-9 numbers
            futures.append(executor.submit(fetch_random_word_plus_int_batch, batch_size, max_length))

            # 1/4 words from the Random Word API followed by 1-9 numbers and some letters replaced by special characters
            futures.append(executor.submit(fetch_random_word_plus_int_and_special_char_batch, batch_size, max_length))

        for future in as_completed(futures):
            result = future.result()
            negative_samples.extend(result)

    # Ensure no duplicates between positive and negative samples
    negative_samples = list(set(negative_samples) - set(existing_passwords))

    return negative_samples[:num_samples]

def generate_data(file_path, output_file_path, generated_data_factor):
    """
    Generate data by reading passwords from a CSV file, cleaning the data, generating negative samples,
    and writing the combined data to a new CSV file.
    
    Args:
        file_path (str): The path to the input CSV file.
        output_file_path (str): The path to the output CSV file.
        generated_data_factor (int): The factor of uncracked passwords to cracked passwords.
    
    Raises:
        FileNotFoundError: If the input file does not exist.
        pd.errors.EmptyDataError: If the input file is empty.
        pd.errors.ParserError: If the input file is not a valid CSV.
    """
    try:
        # Read only the 'password' column from the CSV file
        print("Reading CSV file...")
        passwords = pd.read_csv(file_path, usecols=['password'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input data file not found: {file_path}") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Input data file is empty: {file_path}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Input data file is not a valid CSV: {file_path}") from e

    try:
        # Remove entries where the password is empty (represented by '-'), NaN, '<No Pass>', '<Any Pass> or '(none)'
        print("Cleaning data...")
        passwords = passwords[passwords['password'] != '-']
        passwords = passwords[passwords['password'] != '<No Pass>']
        passwords = passwords[passwords['password'] != '<Any Pass>']
        passwords = passwords[passwords['password'] != '(none)']
        passwords = passwords.dropna(subset=['password'])

        # Create a binary target variable where 1 indicates the password is in the list
        passwords['target'] = 1

        # Determine the maximum length of the longest positive password
        max_length = passwords['password'].str.len().max()
        print(f"Maximum length of positive passwords: {max_length}")

        # Generate negative samples (random passwords not in the cracked list)
        num_negative_samples = len(passwords) * generated_data_factor
        print("Generating negative samples...")
        negative_samples = generate_negative_samples(num_negative_samples, passwords['password'].values, max_length)

        print(f"Generated {len(negative_samples)} negative samples.")

        negative_samples_df = pd.DataFrame({
            'password': negative_samples,
            'target': 0
        })

        # Combine positive and negative samples
        print("Combining positive and negative samples...")
        data = pd.concat([passwords, negative_samples_df])

        # Remove any remaining NaN values from the final table
        data = data.dropna()

        # Ensure no duplicates between positive and negative samples
        data = data.drop_duplicates(subset=['password'], keep='first')

        # Write the combined dataset to a new CSV file
        print(f"Writing combined data to {output_file_path}...")
        data.to_csv(output_file_path, index=False)
        print("Data generation completed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def main():
    """
    Main function for manual use.
    """
    file_path = 'data/events.csv'
    output_file_path = 'data/combined_data.csv'
    print("The factor of uncracked passwords to cracked passwords you want is: ")
    generated_data_factor = int(input())
    generate_data(file_path, output_file_path, generated_data_factor)

if __name__ == "__main__":
    main()