import os
import re
import pandas as pd
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

VIETNAMESE_STOPWORDS = {
    'và', 'là', 'của', 'có', 'không', 'cho', 'được', 'với', 'một', 'nhiều', 'rất', 'những',
    'đã', 'tôi', 'chúng', 'ta', 'mà', 'đó', 'còn', 'này', 'kia', 'vì', 'theo', 'khi', 'thì',
    'hay', 'trong', 'ngoài', 'trên', 'dưới', 'về', 'để', 'bị', 'như', 'đây', 'sẽ', 'đang',
    'nhưng', 'vẫn', 'cũng', 'đã', 'sau', 'trước', 'nên', 'nếu', 'chỉ', 'ra', 'vào', 'mà',
    'lại', 'cả', 'vì', 'như', 'giữa', 'theo', 'để', 'đến', 'ở', 'được', 'nên'
}

LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}

VALID_CHAR_PATTERN = re.compile(r"[^0-9a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]")


def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản tiếng Việt: chuẩn hóa, tách từ, loại stopwords."""
    if text is None:
        text = ""
    text = str(text).strip().lower()
    text = VALID_CHAR_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = word_tokenize(text, format="text")
    tokens = [token for token in text.split() if token not in VIETNAMESE_STOPWORDS]
    return " ".join(tokens)


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'sentence' in df.columns:
        df = df.rename(columns={'sentence': 'text'})
    if 'sentiment' in df.columns:
        df['label'] = df['sentiment'].map(LABEL_MAPPING)
    return df.dropna(subset=['text', 'label'])


def get_dataset_stats() -> dict:
    stats = {}
    for name, path in {
        'train': 'data_csv/train.csv',
        'test': 'data_csv/test.csv',
        'validation': 'data_csv/validation.csv'
    }.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            stats[name] = {'rows': len(df), 'cols': len(df.columns)}
        else:
            stats[name] = {'rows': 0, 'cols': 0}
    return stats


def load_and_train_model(train_path: str = 'data_csv/train.csv', test_path: str = 'data_csv/test.csv'):
    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train = _load_dataset(train_path)
        df_test = _load_dataset(test_path)
    else:
        df_train = pd.DataFrame([
            ("Thầy dạy rất nhiệt tình, giảng bài dễ hiểu.", "Positive"),
            ("Phòng học quá nóng, điều hòa hỏng.", "Negative"),
            ("Môn học bình thường, không có gì đặc sắc.", "Neutral")
        ], columns=["text", "label"])
        df_test = df_train.copy()

    df_train['processed_text'] = df_train['text'].apply(preprocess_text)
    df_test['processed_text'] = df_test['text'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(df_train['processed_text'])
    y_train = df_train['label']
    X_test = vectorizer.transform(df_test['processed_text'])
    y_test = df_test['label']

    model = LogisticRegression(random_state=42, C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'cm': confusion_matrix(y_test, y_pred, labels=['Negative', 'Neutral', 'Positive']),
        'labels': ['Negative', 'Neutral', 'Positive']
    }

    return vectorizer, model, metrics, df_train
