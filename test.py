import json

jsonData = '{"name": "Frank", "age": 39}'
jsonToPython = json.loads(jsonData)

with open('data.json', 'r') as f:
    json_data = json.load(f)
print (json_data)

with open('data.json', 'a') as f:
    json.dump(jsonToPython, f)

print ("\n\n NEW\n")
