import polygon
from pretty_dict import pretty
import matplotlib.pyplot as plt
import numpy as np

stocks_client = polygon.StocksClient('w2BPqZGZRpCn_Hr_r8HO_22S_CXMeYdp')  # for usual sync client
async_stock_client = polygon.StocksClient('w2BPqZGZRpCn_Hr_r8HO_22S_CXMeYdp', True)  # for an async client

data = stocks_client.get_sma('app')

xpoints = np.array([])
ypoints = np.array([])

for values in data["results"]["values"]:
    print(float(values["timestamp"]))
    np.append(xpoints, float(values["timestamp"]))
    np.append(ypoints, float(values["value"]))

print(xpoints)

plt.plot(xpoints, ypoints)
plt.show()


# # client = RESTClient("API_KEY")
    
# aggs = []
# for a in stocks_client.list_aggs(
#     "AAPL",
#     1,
#     "minute",
#     "2022-01-01",
#     "2023-02-03",
#     limit=50000,
# ):
#     aggs.append(a)

# print(aggs)
