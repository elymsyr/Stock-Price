# %%
from requests import get
from json import loads

# %%
url = 'http://127.0.0.1:8000/tomorrow'
# %%
response = get(url)
# %%
print(loads(response.text))

# %%
