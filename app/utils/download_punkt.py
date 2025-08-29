import nltk

def download_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
        print("punkt is already downloaded. proceeding further...")
    except LookupError:
        print("punkt is not found. downloading now...")
        nltk.download("punkt")