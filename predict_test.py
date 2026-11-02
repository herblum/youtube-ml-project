import requests

url = "http://0.0.0.0:8000/predict"

test_sample = {
  "category": "Education",
  "duration_sec": 1201,
  "title_length": 72,
  "thumbnail_face": 0,
  "upload_hour": 13,
  "upload_day": "Sat",
  "tags_count": 15,
  "description_length": 569,
  "subscriber_count": 2789543
}

response = requests.post(url, json=test_sample)
print(response.json())