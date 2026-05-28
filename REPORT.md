# Báo cáo Dự án: Hệ thống Phân tích Cảm xúc Phản hồi Sinh viên

## 1. Giới thiệu

Đề tài xây dựng một hệ thống sử dụng kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) để phân tích cảm xúc từ các phản hồi của sinh viên về môn học, giảng viên hoặc cơ sở vật chất. Hệ thống phân loại nội dung thành ba nhóm cảm xúc chính: tích cực, tiêu cực và trung lập.

## 2. Mục tiêu

- Nghiên cứu và áp dụng kỹ thuật NLP cho tiếng Việt.
- Tiền xử lý dữ liệu bằng thư viện `underthesea`.
- Xây dựng mô hình phân tích cảm xúc sử dụng `TF-IDF` và `Logistic Regression`.
- Triển khai giao diện demo bằng `Streamlit` để nhập phản hồi và hiển thị kết quả real-time.
- Hỗ trợ xuất báo cáo kết quả phân tích ra file CSV và PDF.

## 3. Dữ liệu sử dụng

- Dataset chính: **UIT-VSFC**.
- Thư mục dữ liệu: `data_csv/`.
- Kích thước bộ dữ liệu:
  - `train.csv`: 11.426 mẫu
  - `test.csv`: 3.166 mẫu
  - `validation.csv`: 1.583 mẫu
- Mỗi mẫu bao gồm câu phản hồi (`sentence`), nhãn cảm xúc (`sentiment`) và chủ đề (`topic`).

## 4. Phương pháp

### 4.1. Tiền xử lý dữ liệu

- Chuyển văn bản về chữ thường.
- Loại bỏ các ký tự đặc biệt không phải chữ tiếng Việt và số.
- Chuẩn hóa khoảng trắng.
- Tách từ tiếng Việt bằng `underthesea.word_tokenize`.
- Loại bỏ stopwords tiếng Việt cơ bản để giảm nhiễu cho đại diện đặc trưng.

### 4.2. Trích xuất đặc trưng

- Sử dụng `TfidfVectorizer` với `ngram_range=(1, 2)`.
- Mô hình tính trọng số từ quan trọng trên toàn văn bản.

### 4.3. Huấn luyện mô hình

- Thuật toán: `LogisticRegression`.
- Sử dụng tập huấn luyện `train.csv` để fit mô hình.
- Kiểm thử trên `test.csv` với 3 nhãn: `Negative`, `Neutral`, `Positive`.

## 5. Kết quả thực nghiệm

- Mô hình được đánh giá bằng các chỉ số:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Kết quả đánh giá chi tiết được hiển thị trong ứng dụng Streamlit tại tab **Báo Cáo Hiệu Suất AI**.

## 6. Phân tích

- Mô hình `TF-IDF + Logistic Regression` phù hợp với bài toán phân loại văn bản ngắn, xử lý nhanh và dễ triển khai.
- Sử dụng tokenization và stopwords giúp giảm nhiễu, đặc biệt với văn bản tiếng Việt.
- Giới hạn của mô hình:
  - Chưa xử lý tốt mỉa mai, biểu cảm phức tạp hoặc câu dài.
  - Chưa áp dụng mô hình ngôn ngữ sâu như PhoBERT.
  - Dữ liệu chưa đảm bảo độ cân bằng hoàn hảo giữa các nhãn.

## 7. Kết luận

Dự án đã hoàn thành phần code cơ bản cho đề tài:

- Giao diện Streamlit tương tác.
- Tiền xử lý tiếng Việt bằng `underthesea`.
- Mô hình phân tích cảm xúc `TF-IDF + Logistic Regression`.
- Đánh giá bằng Accuracy, Precision, Recall, F1-score.
- Hỗ trợ xuất báo cáo CSV và PDF.

## 8. Hướng dẫn chạy

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 9. Kiểm thử

- Chạy kiểm thử tự động với:

```bash
python tests.py
```
