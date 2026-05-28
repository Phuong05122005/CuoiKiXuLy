# Đại Học Lạc Hồng — Bài Cuối Môn Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

**Thông tin sinh viên:**

| STT | Mã Số Sinh Viên |     Họ và Tên    |
|-----|-----------------|------------------|
| 1   |    123000846    |Nguyễn Thái Phương|
| 2   |    123000384    |   Phan Gia Huy   |

**Đề tài đã chọn:** _(Xây dựng hệ thống phân tích cảm xúc phản hồi sinh viên)_

---

## Mô tả dự án

_(Mô tả ngắn gọn:

Đề tài xây dựng một hệ thống sử dụng kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) để phân tích cảm xúc từ các phản hồi của sinh viên về môn học, giảng viên hoặc cơ sở vật chất. Hệ thống sẽ phân loại nội dung phản hồi thành các nhóm cảm xúc như tích cực, tiêu cực và trung lập nhằm hỗ trợ nhà trường đánh giá mức độ hài lòng của sinh viên.

Mục tiêu:
Tìm hiểu và áp dụng các kỹ thuật NLP cho tiếng Việt.
Xây dựng mô hình phân tích cảm xúc sử dụng TF-IDF + Machine Learning hoặc PhoBERT.
Tiền xử lý dữ liệu tiếng Việt bằng thư viện underthesea.
Xây dựng giao diện demo bằng Streamlit cho phép người dùng nhập phản hồi và xem kết quả phân tích cảm xúc.
Phạm vi dự án:
Dữ liệu sử dụng chủ yếu là phản hồi sinh viên tiếng Việt từ dataset UIT-VSFC hoặc dữ liệu thu thập thủ công.
Hệ thống chỉ tập trung phân loại cảm xúc văn bản ở mức cơ bản: tích cực, tiêu cực, trung lập.
Chưa xử lý các trường hợp phức tạp như mỉa mai, đa nghĩa hoặc hội thoại dài.
Demo hoạt động trên môi trường local bằng Python và Streamlit.)_

---

## Cấu trúc project

```
d:
├── app.py              # Ứng dụng Streamlit chính
├── model.py            # Module tách riêng logic train/eval và tiền xử lý
├── tests.py            # Kiểm thử tự động mẫu
├── REPORT.md           # Báo cáo luận văn/môn học súc tích
├── requirements.txt    # Danh sách thư viện
├── DE-TAI.md           # Chi tiết các đề tài gợi ý
├── SETUP.md            # Hướng dẫn cài đặt
├── README.md           # File này
└── LICENSE
```

---

## Dữ liệu dự án

- `data_csv/train.csv`: 11.426 mẫu
- `data_csv/test.csv`: 3.166 mẫu
- `data_csv/validation.csv`: 1.583 mẫu
- Dữ liệu hiện có sẵn trong thư mục `data_csv/` và ứng dụng sẽ tải mô hình bằng dữ liệu này khi chạy.

---

## Hướng dẫn cài đặt & chạy

```bash
# 1. Tạo môi trường ảo
python3 -m venv venv

# 2. Kích hoạt môi trường ảo
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate       # Windows

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Chạy ứng dụng demo
streamlit run app.py

# 5. Chạy kiểm thử tự động
python tests.py
```

---

## Báo cáo hoàn chỉnh

- Xem file `REPORT.md` để có phần mô tả mục tiêu, phương pháp, dữ liệu, kết quả thực nghiệm, phân tích và kết luận.
- `REPORT.md` đã bổ sung nội dung phù hợp yêu cầu luận văn/báo cáo môn học.

---

## Các nhiệm vụ cần hoàn thành trong `app.py`

_(Sinh viên liệt kê các TODO / nhiệm vụ cụ thể theo đề tài đã chọn)_

- [ ] Nhiệm vụ 1: Thu thập và tìm hiểu dataset phản hồi sinh viên tiếng Việt (UIT-VSFC)
- [ ] Nhiệm vụ 2: Tiền xử lý dữ liệu tiếng Việt bằng underthesea (tokenize, loại bỏ stopwords, chuẩn hóa văn bản)
- [ ] Nhiệm vụ 3: Xây dựng mô hình phân tích cảm xúc bằng TF-IDF và Logistic Regression
- [ ] Nhiệm vụ 4: Xử lý lỗi khi input rỗng hoặc văn bản không hợp lệ
- [ ] Nhiệm vụ 5: Export kết quả phân tích cảm xúc ra file CSV hoặc PDF
- [ ] Nhiệm vụ 6: Trực quan hoá kết quả bằng biểu đồ, confidence score và word cloud
- [ ] Nhiệm vụ 7: Xây dựng giao diện Streamlit cho phép nhập văn bản và hiển thị kết quả dự đoán
- [ ] Nhiệm vụ 8: Đánh giá mô hình bằng Accuracy, Precision, Recall và F1-score
- [ ] Nhiệm vụ 9: Kiểm thử hệ thống với nhiều loại phản hồi khác nhau
- [ ] Nhiệm vụ 10: Hoàn thiện báo cáo

---

## Công nghệ sử dụng

- Python 3.x
- Streamlit (giao diện web)
- underthesea (tiền xử lý tiếng Việt)
- scikit-learn (TF-IDF, Logistic Regression, đánh giá mô hình)
- pandas (xử lý dữ liệu)
- numpy (tính toán số học)
- matplotlib / seaborn / plotly (trực quan hoá)
- wordcloud (tạo word cloud)
- fpdf (xuất báo cáo PDF)

---

## Demo

Ứng dụng chạy local bằng lệnh sau:

```bash
streamlit run app.py
```

Mở trình duyệt và truy cập địa chỉ được Streamlit cung cấp để nhập phản hồi sinh viên và xem kết quả phân tích cảm xúc.

> Nếu muốn, bạn có thể deploy lên Streamlit Cloud hoặc Hugging Face Spaces và cập nhật link tại đây sau khi hoàn thành.

---

## Tài liệu tham khảo

- Dataset UIT-VSFC (bộ dữ liệu phản hồi sinh viên tiếng Việt)
- Thư viện `underthesea` cho xử lý tiếng Việt
- Thư viện `scikit-learn` cho TF-IDF và Logistic Regression
- Thư viện `Streamlit` cho giao diện demo
- Tài liệu hướng dẫn `WordCloud`, `matplotlib`, `seaborn`, và `plotly` cho trực quan hoá
- Các bài viết, blog và tài liệu NLP tiếng Việt liên quan đến phân tích cảm xúc
