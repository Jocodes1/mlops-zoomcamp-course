import requests

ride = {
    "PUlocationID": 130,
    "DOlocationID": 205,
    "trip_distance": 15.5
    }


url = 'http://127.0.0.1:9696/predict'

response = requests.post(url, json=ride)
print(response.json())
