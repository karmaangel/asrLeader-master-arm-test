from main import app


if __name__ == "__main__":
    import uvicorn
    import config

    uvicorn.run("server:app", host=config.HOST, port=config.PORT, workers=1)
