import requests
from json import dumps

key = "OE75KP9T13V2PB59"

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={key}'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Extract the JSON content from the response
    data = response.json()
    
    # Serialize the JSON content with pretty printing
    pretty_data = dumps(data, indent=4)
    
    # Print the pretty JSON data
    print(pretty_data)
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")