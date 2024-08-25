from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World from FAST!"}

@app.post("/post")
def post_something():
    return {"message": "I am posting something"}