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
        "Inputs": {
            "data": [
                {"password": password}
            ]
        }
    }

    body = str.encode(json.dumps(data))

    url = 'http://30e75f18-d937-4d51-a097-deebae87acd3.westeurope.azurecontainer.io/score'
    headers = {'Content-Type': 'application/json'}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None

# Main loop to prompt the user for passwords
def main():
    while True:
        password = input("Enter a password to check (or type 'exit' to go back): ")
        if password.lower() == 'exit':
            break
        result = check_password(password)
        if result:
            is_hacked = result.get('Results', [None])[0]
            if is_hacked is True:
                print(f'The password "{password}" is most likely a HACKED password. Be Warned!')
            elif is_hacked is False:
                print(f'The password "{password}" is most likely NOT a hacked password.')
            else:
                print(f'Unexpected result for "{password}": {result}')

if __name__ == "__main__":
    main()