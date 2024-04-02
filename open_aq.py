import requests
"""
# Define the API endpoint for measurements
url = 'https://api.openaq.org/v1/measurements'

# Define the parameters for a more general request
params = {
    'location': 'HAMPTON',  # Country code, e.g., "US" for the United States
    'parameter': 'no2',
    'limit': 10,  # Number of results to return
}

# Make the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    for result in data['results']:
        print(f"Location: {result['location']},Transport: {result['close']}, Parameter: {result['parameter']}, Value: {result['value']} {result['unit']}, Date: {result['date']['utc']}")
else:
    print(f"Failed to fetch data: {response.status_code}")

"""
def get_allcountries():

    # Define the API endpoint for countries
    url = 'https://api.openaq.org/v1/countries'

    # Make the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        # Extract and print the list of countries
        countries = data['results']
        for country in countries:
            print(f"Code: {country['code']}, Name: {country['name']}")
    else:
        print(f"Failed to fetch data: {response.status_code}")

import requests

def fetch_measurements(page=1, limit=100):
    url = 'https://api.openaq.org/v2/measurements'
    params = {
        'page': page,
        'limit': limit
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['results']
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return []


# Example of iterating over multiple pages
all_measurements = []
page = 1
max_pages = 10  # Set a limit to avoid excessively long requests

while page <= max_pages:
    results = fetch_measurements(page=page, limit=100)
    if not results:
        break
    all_measurements.extend(results)
    page += 1

# Now all_measurements contains data from the first 10 pages
print(f"Total measurements fetched: {len(all_measurements)}")
print(all_measurements)


