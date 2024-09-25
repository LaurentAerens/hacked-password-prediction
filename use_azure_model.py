# This is now included in the use_model.py file. So this file is now mainly for testing only the Azure model.

import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Function to send a string to the Azure model and get the response
def check_password(password):
    # Prepare the data payload
    data = {
        "input_data": {
            "data": [
                {"password": password}
            ]
        }
    }

    body = str.encode(json.dumps(data))

    url = '[your-endpoint-url]'  # Replace this with the endpoint URL
    # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
    api_key = '[your-api-key]'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")
        
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

def main():
    while True:
        password = input("Enter a password to check (or type 'exit' to quit): ")
        if password.lower() == 'exit':
            break
        
        result = check_password(password)
        
        if result is not None:
            print(f"Password is hacked: {result}")
        else:
            print("Failed to get a valid response from the model.")
if __name__ == "__main__":
    main()