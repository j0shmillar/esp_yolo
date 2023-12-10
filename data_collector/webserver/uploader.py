import requests

# filename as cli argument
import sys
filename = sys.argv[1]
print(f'Uploading {filename}...')

url = 'http://localhost:5000/upload'
files = {'file': open(filename, 'rb')}
response = requests.post(url, files=files)

print(response.text)

