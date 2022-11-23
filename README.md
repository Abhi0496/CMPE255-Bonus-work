## Setup


1. Clone git repo
```bash
git clone https://github.com/Abhi0496/CMPE255-Bonus-work
cd CMPE255-Bonus-work
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Start flask app
```bash
python app.py
```
The flask app has two end points
 - `/torch` : for pytorch inference
 - `/openvino`: for openvino inference

2. Make HTTP request to the flask app
```python
import requests

url_torch = "http://localhost:5000/torch" # for pytorch inference
url_openvino = "http://localhost:5000/openvino" # for openvino inference

resp = requests.post(url_torch, files={"files": open('cat.jpeg', 'rb')})
print(resp)
print(resp.json())
```

