from model import preprocess_text, load_and_train_model


def test_preprocess_text():
    sample = "Giảng viên dạy rất nhiệt tình và hỗ trợ sinh viên."
    result = preprocess_text(sample)
    assert isinstance(result, str)
    assert "giảng_vien" in result or "giảng" in result
    assert "và" not in result


def test_model_training_and_prediction():
    vectorizer, model, metrics, df_train = load_and_train_model()
    assert 'accuracy' in metrics
    assert metrics['accuracy'] >= 0

    sample = "Thầy cô giảng dễ hiểu, sinh viên thấy hài lòng."
    processed = preprocess_text(sample)
    prediction = model.predict(vectorizer.transform([processed]))[0]
    assert prediction in ['Positive', 'Neutral', 'Negative']
    assert len(df_train) > 0


if __name__ == '__main__':
    test_preprocess_text()
    test_model_training_and_prediction()
    print('✅ Kiểm thử tự động đã chạy thành công.')
