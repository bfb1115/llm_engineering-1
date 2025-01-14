import requests
from pprint import pprint as print

url = "https://10.0.0.15/process"
param_list = []
param_list.append({"po_number": 589955})
param_list.append({"order_number": 530042})


i = 0
response_list = []
for param in param_list:
    params = param
    response = requests.get(url, params=params, verify=False)
    response_list.append(response.json())

for response in response_list:
    print(response)
    print("\n")
