# Sentiment Insights
Developed by Ahmed Sheta. A Technical Challenge of Sentiment Analysis using FastAPI

![Sentiment Insights App Logo](sentiment_insights_logo.png)

Sentiment Insights is a simple **FastAPI** service that provides sentiment analysis based on **TextBlob**.  
The service exposes REST endpoints to classify text into **negative**, **neutral**, or **positive**, with polarity values.

---
## Creating the Environment

To create the virtual environment in this project we must have `pipenv`
installed on our machine. Then run the following commands:

```bash
pip install pipenv
# for development environment
pipenv install --dev
# for production environment
pipenv install
```

To work within the environment we can now run:

```bash
# to activate the virtual environment
pipenv shell
# to run a single command
pipenv run <COMMAND>
```

## Build Process

This application is built and tested on every push and pull request creation
through Github actions. For this, the `pipenv` environment is installed and then
the code style is checked using `flake8`. Finally, the `tests/` directory is
executed using `pytest` and a test coverage report is created using `coverage`.
The test coverage report can be found in the Github actions output.

## Running the app

To run the application the `pipenv` environment must be installed. Then the application can
be started in multiple ways:

### Running locally

For the development purposes, the real time debugging helps indicate the script functionality,
rather than building a full image or an API. For this, we can run:

```bash
python src\sentiment.py
```

### Inference on FAST API

We can run the inference through FAST API on a local port by carrying out these steps:

```bash
uvicorn src.app_textblob:app --host localhost --port 8000
```
We can either then go to (http://127.0.0.1:8000/docs) insert the desired feedback for sentiment analysis,
or run these commands on an additional terminal:

```bash
curl -Method POST http://127.0.0.1:8000/predict `
   -Headers @{ "Content-Type" = "application/json" } `
   -Body '{ "text": "I absolutely love this!"}'
# or in case of multiple feedbacks
curl -Method POST http://127.0.0.1:8000/predict `
   -Headers @{ "Content-Type" = "application/json" } `
   -Body '{ "texts": ["this is bad", "meh", "pretty good"] }'

```

Then, we would receive a response with the classification and the polarity score ranging from -1 to 1.

## Building a docker image and running inference
We can build a docker image with all the code and dependencies using the following command line:

```bash
docker build -t sentiment-api .
```

After building the image, we can create and run a container based on this image using the following command line:

```bash
docker run --rm -p 8000:8000 sentiment-api
```



