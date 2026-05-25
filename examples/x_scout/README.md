# X Scout

Scout candidate X posts from an Apify dataset, rank each post with Gemini structured output, and print ranked JSON.

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `GEMINI_API_KEY`. If you also set `APIFY_TOKEN` and `APIFY_DATASET_ID`, the example reads real dataset items. Without Apify settings it uses sample posts.

## Run

```bash
python main.py
```

