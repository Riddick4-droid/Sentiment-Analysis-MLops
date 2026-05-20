import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Union, Dict
from pathlib import Path

from src.exceptions.custom_exceptions import PreprocessingError
from src.utils.logger import setup_logger
from src.utils.tracer import trace

logger = setup_logger("data_preprocessing")

@trace(log_args=True, log_return=False)
def clean_text(
    text:str,
    lowercase:bool=True,
    remove_punctuation:bool=True,
    remove_numbers:bool=True,
    remove_stopwords:bool=True,
    remove_urls:bool=True,
    remove_emojis:bool=True,
    remove_html_tags:bool=True,
    custom_stopwords:Optional[List[str]]=None,
    remove_extra_whitespace:bool=True
) -> str:
    """
    Clean and preprocess text data.

    Args:
        text: The input text string to clean.
        lowercase: Whether to convert text to lowercase.
        remove_punctuation: Whether to remove punctuation characters.
        remove_numbers: Whether to remove numeric characters.
        remove_stopwords: Whether to remove common stopwords.
        remove_urls: Whether to remove URLs from the text.
        remove_emojis: Whether to remove emojis from the text.
        remove_html_tags: Whether to remove HTML tags from the text.
        custom_stopwords: Optional list of additional stopwords to remove.
        remove_extra_whitespace: Whether to collapse multiple spaces into one."""
    
    if pd.isna(text) or text is None:
        logger.warning("Received empty or NaN text input for cleaning")
        return ""
    
    #convert to string (in case we get non-string inputs)
    text = str(text)

    if lowercase:
        text = text.lower()
    if remove_urls:
        url_pattern = r"http\S+|www\S+|https\S+"
        text = re.sub(url_pattern, "", text)
    if remove_emojis:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)
    if remove_html_tags:
        html_pattern = r"<.*?>"
        text = re.sub(html_pattern, "", text)
    if remove_punctuation:
        punctuation_pattern = r"[^\w\s]"
        text = re.sub(punctuation_pattern, "", text)
    if remove_numbers:
        number_pattern = r"\d+"
        text = re.sub(number_pattern, "", text)
    if remove_stopwords:
        from sklearn.feature_extraction import text
        stop_words = set(text.ENGLISH_STOP_WORDS)
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        stopword_pattern = r"\b(" + "|".join(re.escape(word) for word in stop_words) + r")\b"
        text = re.sub(stopword_pattern, "", text) 
    if remove_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text

@trace
def clean_texts_series(
    series:pd.Series,
    **cleaning_kwargs
) -> pd.Series:
    """
    Apply text cleaning to a pandas Series of text data.

    Args:
        series: A pandas Series containing text data to clean.
        **cleaning_kwargs: Keyword arguments to pass to the clean_text function.

    Returns:
        A pandas Series with cleaned text.
    """
    logger.info(f"Cleaning {len(series)} text entries with kwargs: {cleaning_kwargs}")
    cleaned = series.apply(lambda x: clean_text(x, **cleaning_kwargs))
    empty_count = (cleaned == "").sum()
    if empty_count > 0:
        logger.warning(f"Cleaning resulted in {empty_count} empty entries out of {len(series)}")
    return cleaned

@trace(log_args=True, log_return=False)
def encode_labels(
    labels:pd.Series
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode categorical labels into numeric format.

    Args:
        labels: A pandas Series containing categorical labels.

    Returns:
        A tuple of (encoded_labels, label_encoder) where:
        - encoded_labels: A numpy array of encoded numeric labels.
        - label_encoder: The fitted LabelEncoder instance for inverse transformation.
    """
    if labels.isna().any():
        logger.warning("Label series contains NaN values. These will be treated as a separate category.")
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels.astype(str))
    logger.info(f"Encoded {len(labels)} labels into {len(label_encoder.classes_)} classes")
    return encoded_labels, label_encoder


@trace(log_args=True, log_return=False)
def create_vectorizer(
    method:str="tfidf",
    max_features:Optional[int]=None,
    ngram_range:Tuple[int, int]=(1, 1),
    stop_words:Optional[str]="english"
) -> TfidfVectorizer:
    """
    Create a text vectorizer based on the specified method.

    Args:
        method: The vectorization method to use ("tfidf" or "count").
        max_features: The maximum number of features to keep (most frequent).
        ngram_range: The range of n-grams to consider (e.g., (1, 2) for unigrams and bigrams).
        stop_words: Stop words to remove ("english" for built-in list, or a list of custom stop words).

    Returns:
        An instance of TfidfVectorizer or CountVectorizer based on the specified method.
    """
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, 
                                     ngram_range=ngram_range, 
                                     stop_words=stop_words)
        logger.info(f"Created TfidfVectorizer with max_features={max_features}, ngram_range={ngram_range}, stop_words={stop_words}")
    elif method == "count":
        vectorizer = CountVectorizer(max_features=max_features, 
                                     ngram_range=ngram_range, 
                                     stop_words=stop_words)
        logger.info(f"Created CountVectorizer with max_features={max_features}, ngram_range={ngram_range}, stop_words={stop_words}")
    else:
        error_msg = f"Unsupported vectorization method: {method}. Use 'tfidf' or 'count'."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return vectorizer

@trace(log_args=True, log_return=False)
def fit_vectorizer(
    vectorizer:Union[TfidfVectorizer, CountVectorizer],
    texts:pd.Series
) -> np.ndarray:
    """
    Fit the vectorizer to the text data and transform it into a feature matrix.

    Args:
        vectorizer: An instance of TfidfVectorizer or CountVectorizer.
        texts: A pandas Series containing the text data to vectorize."""
    logger.info(f"Fitting vectorizer to {len(texts)} text entries")
    try:
        features = vectorizer.fit(texts)
        logger.info(f"Vectorizer fitted successfully. Feature matrix shape: {features.shape}")
        return features
    except Exception as e:
        error_msg = f"Failed to fit vectorizer: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise PreprocessingError(error_msg, original_exception=e)
    
@trace
def transform_texts(
    vectorizer:Union[TfidfVectorizer, CountVectorizer],
    texts:pd.Series
) -> np.ndarray:
    """
    Transform text data into a feature matrix using the fitted vectorizer.

    Args:
        vectorizer: A fitted instance of TfidfVectorizer or CountVectorizer.
        texts: A pandas Series containing the text data to transform.

    Returns:
        A sparse matrix representing the transformed text features.
    """
    logger.info(f"Transforming {len(texts)} text entries using the fitted vectorizer")
    try:
        features = vectorizer.transform(texts)
        logger.info(f"Text transformation successful. Feature matrix shape: {features.shape}")
        return features
    except Exception as e:
        error_msg = f"Failed to transform texts: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise PreprocessingError(error_msg, original_exception=e)
    
@trace
def preprocess_pipeline(
    df:pd.DataFrame,
    text_column:str,
    label_column:str,
    vectorizer_method:str="tfidf",
    vectorizer_kwargs:Optional[dict]=None,
    cleaning_kwargs:Optional[dict]=None,
    random_state:int=42,
    test_size:float=0.2
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, Union[TfidfVectorizer, CountVectorizer]]:
    """
    Run the full preprocessing pipeline: clean text, encode labels, vectorize features, and split data.
    """
    if cleaning_kwargs is None:
        cleaning_kwargs = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_numbers": True,
            "remove_stopwords": True,
            "remove_urls": True,
            "remove_emojis": True,
            "remove_html_tags": True,
            "remove_extra_whitespace": True
        }

    #1. extract text and labels
    raw_texts = df[text_column].astype(str).tolist()  #ensure text is string type
    raw_labels = df[label_column].values

    #2. clean text
    cleaned_texts = clean_texts_series(pd.Series(raw_texts), **cleaning_kwargs).tolist()

    #3. encode labels
    encoded_labels, label_encoder = encode_labels(pd.Series(raw_labels))

    #4. create vectorizer and fit-transform text
    vectorizer_kwargs = vectorizer_kwargs or {}
    vectorizer = create_vectorizer(method=vectorizer_method, **vectorizer_kwargs)
    fitted_vectorizer = fit_vectorizer(vectorizer, pd.Series(cleaned_texts))
    X = transform_texts(fitted_vectorizer, pd.Series(cleaned_texts))

    #5. split data
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=test_size, random_state=random_state)
    logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    logger.info(f"Label distribution in train set: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    logger.info(f"Label distribution in test set: {pd.Series(y_test).value_counts(normalize=True).to_dict()}")
    return X_train, X_test, y_train, y_test, label_encoder, fitted_vectorizer

@trace
def get_vectorizer_vocabulary(vectorizer:Union[TfidfVectorizer, CountVectorizer]) -> dict:
    """
    Get the vocabulary mapping from the fitted vectorizer.

    Args:
        vectorizer: A fitted instance of TfidfVectorizer or CountVectorizer."""
    if not hasattr(vectorizer, "vocabulary_"):
        error_msg = "Vectorizer is not fitted yet. Please fit the vectorizer before accessing the vocabulary."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return vectorizer.vocabulary_

@trace
def save_preprocessing_artifacts(
    vectorizer:Union[TfidfVectorizer, CountVectorizer],
    label_encoder:LabelEncoder,
    output_dir:str = "artifacts/preprocessing"
) -> Dict[str,str]:
    """
    Save the fitted vectorizer and label encoder to disk for later use.

    Args:
        vectorizer: A fitted instance of TfidfVectorizer or CountVectorizer.
        label_encoder: A fitted LabelEncoder instance.
        output_dir: Directory to save the artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #the vectorizer will be saved based on its type (tfidf or count) to avoid confusion
    if isinstance(vectorizer, TfidfVectorizer):
        vec_path = output_path / "tfidf_vectorizer.pkl"
    else:
        vec_path = output_path / "count_vectorizer.pkl"

    le_path = output_path / "label_encoder.pkl"

    try:
        import joblib
        joblib.dump(vectorizer, vec_path)
        joblib.dump(label_encoder, le_path)
        logger.info(f"Saved vectorizer to {vec_path} and label encoder to {le_path}")
        return {"vectorizer_path": str(vec_path), "label_encoder_path": str(le_path)}
    except Exception as e:
        error_msg = f"Failed to save preprocessing artifacts: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise PreprocessingError(error_msg, original_exception=e)