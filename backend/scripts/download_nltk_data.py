"""One-time setup: download the NLTK corpora used by
app.pipeline.preprocessing.preprocess_email_text().

Run this during dev/deploy setup -- NOT something the app does for itself at
request time. It used to: the first request touching email preprocessing
would try to download these corpora inline, adding 1-2+ seconds of network
latency to that request (see preprocessing.py's _load_nltk() docstring).
Without this run, the app still works, just with a smaller built-in
stop-word list instead of NLTK's.

Usage:
    python -m scripts.download_nltk_data
"""

import nltk


def main() -> None:
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(resource)
        # nltk.download() can report "already up-to-date" and skip extraction
        # when a stale/partial .zip is already present but was never
        # unpacked (observed on this machine's Windows nltk_data directory:
        # wordnet.zip and omw-1.4.zip existed but had no extracted folder
        # next to them, so nltk.data.find() failed even after a "successful"
        # download). Verify the resource actually resolves, and force a
        # fresh download if it doesn't.
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"{resource} did not resolve after download; forcing a fresh download")
            nltk.download(resource, force=True)
            nltk.data.find(f"corpora/{resource}")  # raise loudly if still broken


if __name__ == "__main__":
    main()
