import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from wordcloud import WordCloud
from model import load_and_train_model, preprocess_text, get_dataset_stats

# --- 1. SETUP THÔNG SỐ TRANG ---
st.set_page_config(
    page_title="Phân Tích Cảm Xúc Sinh Viên",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh cho UI/UX hiện đại
st.markdown("""
<style>
    /* Tổng quan */
    body {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        color: #0D47A1;
        text-align: center;
        font-weight: 800;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Box kết quả cảm xúc */
    .sentiment-box {
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .sentiment-box:hover {
        transform: translateY(-5px);
    }
    .pos-box { background: linear-gradient(135deg, #4CAF50, #81C784); }
    .neg-box { background: linear-gradient(135deg, #F44336, #E57373); }
    .neu-box { background: linear-gradient(135deg, #9E9E9E, #BDBDBD); }
    
    /* Tùy chỉnh Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 10px;
        text-align: center;
    }
    .sidebar-info {
        background-color: #F3F6F9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1976D2;
        margin-bottom: 20px;
    }
    
    /* Button Tải xuống */
    .btn-download {
        display: inline-block;
        padding: 10px 20px;
        background: linear-gradient(135deg, #1976D2, #42A5F5);
        color: white !important;
        text-decoration: none;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .btn-download:hover {
        background: linear-gradient(135deg, #1565C0, #1E88E5);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR THÔNG TIN ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Logo-Lac-Hong-University.png/320px-Logo-Lac-Hong-University.png", use_container_width=True)
    st.markdown('<div class="sidebar-title">Đồ Án Cuối Kỳ</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info">
        <b>Đề tài:</b> Xây dựng hệ thống phân tích cảm xúc phản hồi sinh viên<br><br>
        <b>Trường:</b> Đại Học Lạc Hồng<br>
        <b>Sinh viên thực hiện:</b>
        <ul>
            <li>Nguyễn Thái Phương</li>
            <li>Phan Gia Huy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🔍 Giới thiệu:**")
    st.caption("Ứng dụng AI (NLP) để tự động phân loại các phản hồi của sinh viên về chất lượng giảng dạy, cơ sở vật chất thành 3 nhóm: Tích cực, Tiêu cực, Trung lập.")

# --- 3. DATASET & PREPROCESSING & MODEL (CACHED) ---
@st.cache_resource
def load_and_train_resource():
    return load_and_train_model()

vectorizer, model, metrics, df_train = load_and_train_resource()

# --- 4. TIỆN ÍCH UI ---
def get_download_link(df, filename="ket_qua_phan_tich.csv", text="Tải xuống Excel/CSV"):
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn-download">📥 {text}</a>'
    return href

def create_pdf_report(df):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Báo cáo Phân tích Cảm xúc Sinh viên", ln=True)
    pdf.ln(4)

    for _, row in df.tail(15).iterrows():
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 8, f"{row['Thời gian']} | {row['Cảm xúc']} | {row['Độ tin cậy']}", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, f"Văn bản gốc: {row['Văn bản gốc']}")
        pdf.multi_cell(0, 6, f"Văn bản tiền xử lý: {row['Văn bản tiền xử lý']}")
        pdf.ln(2)

    pdf_output = pdf.output(dest='S').encode('latin-1', errors='replace')
    return pdf_output


def plot_confidence_gauge(confidence, label):
    color = "#4CAF50" if label == "Positive" else "#F44336" if label == "Negative" else "#9E9E9E"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        number = {'suffix': "%", 'font': {'size': 40, 'color': color}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#f3f3f3"},
                {'range': [50, 80], 'color': "#e0e0e0"},
                {'range': [80, 100], 'color': "#fafafa"}],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# --- 5. GIAO DIỆN CHÍNH ---
st.markdown('<p class="main-header">Phân Tích Cảm Xúc Sinh Viên</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hệ thống AI tự động phân loại đánh giá chất lượng dạy & học</p>', unsafe_allow_html=True)

# Khởi tạo tabs với icon
tab1, tab2, tab3 = st.tabs(["🚀 Phân Tích Phản Hồi", "📊 Báo Cáo Hiệu Suất AI", "💡 Kiến Trúc Hệ Thống"])

# ================= TAB 1: PHÂN TÍCH =================
with tab1:
    st.markdown("### Nhập Dữ Liệu Cần Phân Tích")
    
    # Text input
    user_input = st.text_area(
        "Mời nhập nội dung (Gợi ý: Càng chi tiết mô hình phân tích càng chuẩn xác)", 
        height=120,
        placeholder="Ví dụ: Giảng viên dạy rất tận tâm, nhưng wifi phòng học hơi yếu..."
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_btn = st.button("🔎 PHÂN TÍCH BẰNG AI", type="primary", use_container_width=True)
        
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=["Thời gian", "Văn bản gốc", "Văn bản tiền xử lý", "Cảm xúc", "Độ tin cậy"])

    if analyze_btn:
        if not user_input.strip():
            st.error("⚠️ Vui lòng nhập nội dung văn bản trước khi nhấn nút!")
        else:
            with st.spinner("Đang sử dụng Mô hình Logistic Regression xử lý ngôn ngữ..."):
                processed_text = preprocess_text(user_input)
                X_input = vectorizer.transform([processed_text])
                
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                confidence = np.max(probabilities)
                
                # Render Kết quả to rõ
                st.markdown("---")
                if prediction == "Positive":
                    st.markdown('<div class="sentiment-box pos-box">😊 ĐÁNH GIÁ TÍCH CỰC (POSITIVE)</div>', unsafe_allow_html=True)
                elif prediction == "Negative":
                    st.markdown('<div class="sentiment-box neg-box">😠 ĐÁNH GIÁ TIÊU CỰC (NEGATIVE)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sentiment-box neu-box">😐 ĐÁNH GIÁ TRUNG LẬP (NEUTRAL)</div>', unsafe_allow_html=True)
                    
                # Hai cột cho Chart & WordCloud
                st.markdown("### Chi Tiết Phân Tích")
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    st.markdown("**Mức độ tin cậy của AI:**")
                    st.plotly_chart(plot_confidence_gauge(confidence, prediction), use_container_width=True)
                    
                with c2:
                    st.markdown("**Từ khóa trọng tâm (Word Cloud):**")
                    try:
                        wordcloud = WordCloud(width=500, height=300, background_color='white', colormap='viridis', max_words=50).generate(processed_text)
                        fig_wc, ax = plt.subplots(figsize=(5, 3))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc)
                    except ValueError:
                        st.info("Câu quá ngắn để trích xuất Word Cloud.")

                # Lưu lịch sử
                from datetime import datetime
                new_row = {
                    "Thời gian": datetime.now().strftime("%H:%M:%S"),
                    "Văn bản gốc": user_input, 
                    "Văn bản tiền xử lý": processed_text, 
                    "Cảm xúc": prediction, 
                    "Độ tin cậy": f"{confidence*100:.2f}%"
                }
                st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)

    # Lịch sử dạng Expander
    st.markdown("---")
    with st.expander("📂 XEM LỊCH SỬ PHÂN TÍCH & XUẤT BÁO CÁO", expanded=False):
        if not st.session_state.history.empty:
            st.dataframe(st.session_state.history, use_container_width=True)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(get_download_link(st.session_state.history), unsafe_allow_html=True)
            with c2:
                pdf_data = create_pdf_report(st.session_state.history)
                if pdf_data is not None:
                    st.download_button(
                        label="📄 Tải xuống báo cáo PDF",
                        data=pdf_data,
                        file_name="bao_cao_phan_tich_cam_xuc.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.info("PDF export chưa khả dụng. Cài thêm thư viện `fpdf` nếu cần.")
        else:
            st.write("Chưa có phiên phân tích nào.")

# ================= TAB 2: ĐÁNH GIÁ =================
with tab2:
    st.markdown("### Đánh Giá Năng Lực Của Mô Hình")
    st.success("✅ Mô hình được kiểm thử trên tập dữ liệu chưa từng thấy (Test set - 3166 dòng) thuộc bộ UIT-VSFC, đảm bảo tính khách quan hoàn toàn.")
    
    # Metrics hiển thị sang trọng
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="Độ chính xác (Accuracy)", value=f"{metrics['accuracy']*100:.2f}%", delta="Tổng quan")
    c2.metric(label="Độ chuẩn xác (Precision)", value=f"{metrics['precision']*100:.2f}%", delta="Nhiễu thấp")
    c3.metric(label="Độ phủ (Recall)", value=f"{metrics['recall']*100:.2f}%", delta="Nhận diện tốt")
    c4.metric(label="F1-Score", value=f"{metrics['f1']*100:.2f}%", delta="Độ cân bằng")
    
    st.markdown("---")
    
    col_cm, col_report = st.columns([1.2, 1])
    
    with col_cm:
        st.markdown("**Ma Trận Nhầm Lẫn (Confusion Matrix)**")
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics["cm"], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=metrics["labels"], yticklabels=metrics["labels"],
                    annot_kws={"size": 12}, linewidths=.5)
        plt.ylabel('Nhãn Thực Tế (True Label)', fontweight='bold')
        plt.xlabel('Nhãn Dự Đoán (Predicted Label)', fontweight='bold')
        st.pyplot(fig_cm)
        st.caption("Trục chéo chính hiển thị số lượng mẫu dự đoán ĐÚNG. Có thể thấy mô hình bắt cực kỳ tốt phản hồi Tích Cực (Positive).")
        
    with col_report:
        st.markdown("**Báo Cáo Chi Tiết (Classification Report)**")
        df_report = pd.DataFrame(metrics["report"]).transpose()
        st.dataframe(df_report.style.format("{:.3f}").background_gradient(cmap='Blues'), use_container_width=True)
        st.caption("Bảng báo cáo bóc tách chi tiết khả năng nhận diện cho từng nhãn riêng biệt.")

    st.markdown("---")
    st.markdown("### Kiểm thử mẫu với dữ liệu đầu vào khác nhau")
    sample_texts = [
        "Giảng viên rất nhiệt tình, bài giảng dễ hiểu.",
        "Phòng học nóng và wifi yếu, làm việc khó chịu.",
        "Môn học bình thường, không có gì nổi bật.",
        "Tôi cảm thấy hài lòng với cách tổ chức của giảng viên.",
        "Cơ sở vật chất cũ kỹ, cần được cải thiện nhanh.",
    ]

    sample_df = pd.DataFrame(sample_texts, columns=["Mẫu phản hồi"])
    sample_df["Tiền xử lý"] = sample_df["Mẫu phản hồi"].apply(preprocess_text)
    sample_X = vectorizer.transform(sample_df["Tiền xử lý"])
    sample_df["Dự đoán"] = model.predict(sample_X)
    sample_df["Độ tin cậy"] = np.max(model.predict_proba(sample_X), axis=1).round(4)
    st.dataframe(sample_df, use_container_width=True)

# ================= TAB 3: KIẾN TRÚC =================
with tab3:
    st.markdown("### Kiến Trúc Hệ Thống & Quá Trình Tiền Xử Lý")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.info("**1. Thu thập dữ liệu (Dataset):**")
        st.write("- Sử dụng bộ dữ liệu chuẩn **UIT-VSFC**.")
        st.write("- Triển khai trên tập `'train.csv'`, `'test.csv'`, `'validation.csv'` trong thư mục `data_csv`.")
        dataset_stats = get_dataset_stats()
        if dataset_stats:
            st.write(f"- Train: {dataset_stats['train']['rows']} mẫu | Test: {dataset_stats['test']['rows']} mẫu | Validation: {dataset_stats['validation']['rows']} mẫu")
        else:
            st.write("- Dữ liệu chưa có sẵn, ứng dụng sẽ dùng dữ liệu dự phòng nhỏ để demo.")
        st.write("- Phân loại 3 nhãn: Tích cực, Tiêu cực, Trung lập.")
        
        st.info("**2. Tiền xử lý ngôn ngữ tự nhiên (NLP):**")
        st.write("- Lowercasing: Chuyển toàn bộ văn bản về chữ thường.")
        st.write("- Chuẩn hóa ký tự, loại bỏ ký tự đặc biệt không cần thiết.")
        st.write("- Tokenization: Sử dụng thư viện `underthesea` để tách từ tiếng Việt chính xác.")
        st.write("- Loại bỏ stopwords tiếng Việt chuẩn, giúp giảm nhiễu cho TF-IDF.")
        
    with c2:
        st.info("**3. Trích xuất đặc trưng (Feature Extraction):**")
        st.write("- Áp dụng thuật toán **TF-IDF Vectorizer** (Term Frequency - Inverse Document Frequency).")
        st.write("- N-gram range: (1, 2) giúp mô hình học cả các cụm 2 từ (Bigram) để bảo toàn ngữ nghĩa.")
        
        st.info("**4. Huấn luyện Machine Learning:**")
        st.write("- Thuật toán: **Logistic Regression**.")
        st.write("- Ưu điểm: Đơn giản, huấn luyện cực nhanh nhưng lại rất hiệu quả đối với bài toán phân loại văn bản thưa thớt (sparse text classification).")