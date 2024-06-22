from polygon import RESTClient

client = RESTClient(api_key="w2BPqZGZRpCn_Hr_r8HO_22S_CXMeYdp")

ticker = "AAPL"

# List Aggregates (Bars)
aggs = []
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_="2023-01-01", to="2023-06-13", limit=50000):
    aggs.append(a)

print(aggs)

# # Get Last Trade
# trade = client.get_last_trade(ticker=ticker)
# print(trade)

# # List Trades
# trades = client.list_trades(ticker=ticker, timestamp="2022-01-04")
# for trade in trades:
#     print(trade)

# # Get Last Quote
# quote = client.get_last_quote(ticker=ticker)
# print(quote)

# # List Quotes
# quotes = client.list_quotes(ticker=ticker, timestamp="2022-01-04")
# for quote in quotes:
#     print(quote)


# #polygon.exceptions.AuthError: Your plan doesn't include websocket access. Visit https://polygon.io/pricing to upgrade.
# from polygon import WebSocketClient
# from polygon.websocket.models import WebSocketMessage
# from typing import List
# # Note: Multiple subscriptions can be added to the array 
# # For example, if you want to subscribe to AAPL and META,
# # you can do so by adding "T.META" to the subscriptions array. ["T.AAPL", "T.META"]
# # If you want to subscribe to all tickers, place an asterisk in place of the symbol. ["T.*"]
# ws = WebSocketClient(api_key="w2BPqZGZRpCn_Hr_r8HO_22S_CXMeYdp", subscriptions=["T.AAPL"])

# def handle_msg(msg: List[WebSocketMessage]):
#     for m in msg:
#         print(m)

# ws.run(handle_msg=handle_msg)