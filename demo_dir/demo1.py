import json

data = [{"name": "Tom", "age": 11}, {"name": "Sam", "age": 13}]
json_str1 = json.dumps(data, ensure_ascii=False)
print(json_str1)
