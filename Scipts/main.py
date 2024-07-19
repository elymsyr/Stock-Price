from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def a():
    return {"message": "Hello World"}