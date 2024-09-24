import pandas as pd
import numpy as np
import random
import string
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# Function to generate a random password
def generate_random_password(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Function to generate a batch of random passwords
def generate_random_password_batch(batch_size, existing_passwords, max_length):
    batch = []
    while len(batch) < batch_size:
        length = random.randint(1, max_length)
        random_password = generate_random_password(length)
        if random_password not in existing_passwords:
            batch.append(random_password)
    return batch

# Function to fetch random words from the API
def fetch_random_word_batch(batch_size, max_length):
    response = requests.get(f'https://random-word.ryanrk.com/api/en/word/random/{batch_size}?maxLength={max_length}')
    if response.status_code == 200:
        words = response.json()
        # Turn the first letter to a capital letter in 25% of the cases
        for i in range(len(words)):
            if random.random() < 0.25:
                words[i] = words[i].capitalize()
        return words
    else:
        return []

# Function to fetch random words from the API and append 1-9 random digits
def fetch_random_word_plus_int_batch(batch_size, max_length):
    words = fetch_random_word_batch(batch_size, max_length)
    modified_words = [word + ''.join(random.choices(string.digits, k=random.randint(1, 9))) for word in words]
    return modified_words

# Function to replace characters in a word with special characters
def replace_with_special_chars(word):
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

# Function to fetch random words from the API, append 1-9 random digits, and replace some characters with special characters
def fetch_random_word_plus_int_and_special_char_batch(batch_size, max_length):
    words = fetch_random_word_batch(batch_size, max_length)
    modified_words = [replace_with_special_chars(word) + ''.join(random.choices(string.digits, k=random.randint(1, 9))) for word in words]
    return modified_words

# Function to generate negative samples
def generate_negative_samples(num_samples, existing_passwords, max_length):
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

def main():
    # Specify the path to your CSV file
    file_path = 'data/events.csv'
    output_file_path = 'data/combined_data.csv'

    # Read only the 'password' column from the CSV file
    print("Reading CSV file...")
    passwords = pd.read_csv(file_path, usecols=['password'])

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

    print("The factor of uncracked passwords to cracked passwords you want is: ")
    generated_data_factor = int(input())

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

if __name__ == "__main__":
    main()