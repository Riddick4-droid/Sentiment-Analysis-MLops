#!/usr/bin/env python3
"""
Inference script for sentiment analysis.
Loads trained model, vectorizer, and label encoder to predict on new text.
"""

import joblib
import re
import argparse
import glob
from pathlib import Path
from typing import List, Union

def clean_text(text: str) -> str:
    """Simple text cleaning matching training preprocessing."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
    text = re.sub(r'\d+', '', text)        # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_artifacts(model_name: str = None):
    """
    Load vectorizer, label encoder, and model.
    If model_name is None, uses the most recent model found in trained_models/
    """
    # Load vectorizer
    vec_path = Path("artifacts/preprocessing/tfidf_vectorizer.pkl")
    if not vec_path.exists():
        vec_path = Path("artifacts/preprocessing/count_vectorizer.pkl")
    if not vec_path.exists():
        raise FileNotFoundError("No vectorizer found in artifacts/preprocessing/")
    vectorizer = joblib.load(vec_path)
    print(f"Loaded vectorizer from {vec_path}")

    # Load label encoder
    le_path = Path("artifacts/preprocessing/label_encoder.pkl")
    if not le_path.exists():
        raise FileNotFoundError("Label encoder not found in artifacts/preprocessing/")
    label_encoder = joblib.load(le_path)
    print(f"Loaded label encoder with classes: {label_encoder.classes_}")

    # Load model - use glob to find matching *_model.pkl
    if model_name:
        # Search for any file ending with {model_name}_model.pkl in trained_models recursively
        pattern = f"trained_models/**/{model_name}_model.pkl"
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(f"No model file found for '{model_name}' in trained_models/")
        model_path = Path(matches[0])  # take the first match
        print(f"Found model for {model_name}: {model_path}")
    else:
        # Auto-select any model: pick the first *_model.pkl found
        matches = glob.glob("trained_models/**/*_model.pkl", recursive=True)
        if not matches:
            raise FileNotFoundError("No model.pkl found in trained_models/")
        model_path = Path(matches[0])
        model_name = model_path.stem.replace("_model", "")  # e.g., logistic_regression_model -> logistic_regression
        print(f"Auto-selected model: {model_name} from {model_path}")

    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return vectorizer, label_encoder, model

def predict(text: Union[str, List[str]], vectorizer, label_encoder, model):
    """Predict sentiment for single text or list of texts."""
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    pred_ids = model.predict(X)
    pred_labels = label_encoder.inverse_transform(pred_ids)
    return pred_labels

def main():
    parser = argparse.ArgumentParser(description="Run sentiment inference")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., logistic_regression)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    try:
        vectorizer, label_encoder, model = load_artifacts(args.model)
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    if args.interactive:
        print("\nSentiment Analysis Interactive (type 'quit' to exit)")
        while True:
            user_input = input("\nEnter text: ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            if not user_input.strip():
                continue
            pred = predict(user_input, vectorizer, label_encoder, model)[0]
            print(f"Predicted sentiment: {pred}")
    elif args.text:
        pred = predict(args.text, vectorizer, label_encoder, model)[0]
        print(f"Text: {args.text}")
        print(f"Sentiment: {pred}")
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        preds = predict(lines, vectorizer, label_encoder, model)
        for text, sent in zip(lines, preds):
            print(f"{text} -> {sent}")
    else:
        # Default example
        test_texts = [
            "I absolutely love this product!",
            "Terrible service, very disappointed.",
            "It's okay, nothing special."
        ]
        print("Running example predictions:")
        for txt in test_texts:
            pred = predict(txt, vectorizer, label_encoder, model)[0]
            print(f"'{txt}' -> {pred}")

if __name__ == "__main__":
    main()