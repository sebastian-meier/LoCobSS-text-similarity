import requests

url = 'http://localhost:8080/similar_new'

body = {
    "question": "Who is god?"
}

response = requests.post(url, data=body)

print(response.json())