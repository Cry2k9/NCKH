import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Chẩn đoán Sốt xuất huyết",
    page_icon="🩸",
    layout="wide"
)

# ---------------------------------------------------------
# 1. Hàm load dữ liệu & Huấn luyện mô hình (Caching)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dengue_dataset_FINAL_clean.csv")
    return df

@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Không thể đọc file dataset 'Dengue_dataset_FINAL_clean.csv'. Vui lòng kiểm tra lại đường dẫn file. Lỗi: {e}")
    st.stop()

# ---------------------------------------------------------
# Tiền xử lý dữ liệu cơ bản (An toàn tuyệt đối)
# ---------------------------------------------------------
df_model = df.copy()

if 'Result' in df_model.columns:
    # 1. Chuyển cột Result về dạng chuỗi viết thường & xóa khoảng trắng thừa
    s_result = df_model['Result'].astype(str).str.strip().str.lower()
    
    # 2. Ánh xạ toàn bộ các dạng biểu diễn của Positive / Negative về 1 và 0
    map_dict = {
        'positive': 1, 'pos': 1, '1': 1, '1.0': 1,
        'negative': 0, 'neg': 0, '0': 0, '0.0': 0
    }
    mapped = s_result.map(map_dict)
    
    # 3. Với các giá trị đã ở sẵn dạng số, dùng to_numeric ép kiểu an toàn
    df_model['Result'] = mapped.fillna(pd.to_numeric(df_model['Result'], errors='coerce'))
    
    # 4. Loại bỏ các dòng bị mâu thuẫn/thiếu nhãn Result
    df_model = df_model.dropna(subset=['Result'])
    
    # 5. Ép kiểu chuẩn int cho XGBoost và sklearn
    df_model['Result'] = df_model['Result'].astype(int)

# Tách features và target
X_raw = df_model.drop(columns=['Result'])
y = df_model['Result']

# Mã hóa các cột dữ liệu phân loại (categorical)
X = pd.get_dummies(X_raw, drop_first=True)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Huấn luyện các mô hình
models = train_models(X_train, y_train)

# ---------------------------------------------------------
# 2. Sidebar Navigation (Mục lục)
# ---------------------------------------------------------
st.sidebar.title("📌 Mục lục")
menu = st.sidebar.radio(
    "Chọn nội dung:",
    [
        "1. Giới thiệu đề tài",
        "2. Tổng quan Dataset",
        "3. Trực quan hóa dữ liệu",
        "4. Công thức & Lý thuyết",
        "5. Baseline & Đánh giá Models",
        "6. Chẩn đoán Sốt xuất huyết"
    ]
)

# ---------------------------------------------------------
# Trang 1. GIỚI THIỆU ĐỀ TÀI
# ---------------------------------------------------------
if menu == "1. Giới thiệu đề tài":
    st.title("🩸 Đề tài: Dự đoán & Chẩn đoán Bệnh Sốt xuất huyết (Dengue Fever)")
    st.markdown("""
    ---
    ### 📝 Thông tin & Mục tiêu đề tài:

    - **Mục tiêu đề tài:** Xây dựng ứng dụng học máy hỗ trợ chẩn đoán sớm khả năng nhiễm virus Sốt xuất huyết dựa trên các chỉ số lâm sàng và xét nghiệm máu của bệnh nhân.
    - **Ý nghĩa thực tiễn:** Hỗ trợ bác sĩ và nhân viên y tế phân loại nhanh bệnh nhân, nâng cao hiệu quả điều trị và giảm thiểu rủi ro biến chứng nguy hiểm.
    - **Phương pháp thực hiện:** Sử dụng các thuật toán Học máy phổ biến như **Logistic Regression**, **Random Forest**, và **XGBoost** trên bộ dữ liệu xét nghiệm lâm sàng Dengue.
    """)

# ---------------------------------------------------------
# Trang 2. TỔNG QUAN DATASET
# ---------------------------------------------------------
elif menu == "2. Tổng quan Dataset":
    st.title("📊 Tổng quan Dataset")
    st.write(f"Bộ dữ liệu gồm có **{df.shape[0]}** dòng và **{df.shape[1]}** cột/đặc trưng.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bản xem trước dữ liệu (Data Preview)")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Thống kê mô tả (Descriptive Statistics)")
        st.dataframe(df.describe())

    st.subheader("Phân bố nhãn mục tiêu (Result)")
    result_counts = df['Result'].value_counts()
    st.bar_chart(result_counts)

# ---------------------------------------------------------
# Trang 3. TRỰC QUAN HÓA DỮ LIỆU
# ---------------------------------------------------------
elif menu == "3. Trực quan hóa dữ liệu":
    st.title("📈 Trực quan hóa Dữ liệu")
    
    col_select = st.selectbox("Chọn thuộc tính để xem phân bố theo kết quả chẩn đoán:", X_raw.columns)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=col_select, hue='Result', kde=True, ax=ax, palette='Set1')
    ax.set_title(f"Phân bố thuộc tính {col_select} theo nhãn Result")
    st.pyplot(fig)

    st.subheader("Ma trận tương quan (Correlation Matrix)")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_model.corr(numeric_only=True), annot=False, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# ---------------------------------------------------------
# Trang 4. CÔNG THỨC & LÝ THUYẾT
# ---------------------------------------------------------
elif menu == "4. Công thức & Lý thuyết":
    st.title("🧮 Công thức & Thuật toán sử dụng")
    
    st.subheader("1. Logistic Regression")
    st.latex(r"P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_n X_n)}}")
    st.write("Sử dụng hàm Sigmoid để biến đổi đầu ra thành xác suất trong khoảng [0, 1].")

    st.subheader("2. Random Forest")
    st.write("Mô hình Ensemble dựa trên kỹ thuật Bagging (Bootstrap Aggregating), kết hợp nhiều cây quyết định để đưa ra dự đoán trung bình/đa số:")
    st.latex(r"\hat{y} = \text{mode}\{T_1(x), T_2(x), \dots, T_B(x)\}")

    st.subheader("3. XGBoost (Extreme Gradient Boosting)")
    st.write("Mô hình Ensemble dựa trên kỹ thuật Boosting, tối ưu hóa hàm mục tiêu gồm hàm mất mát và thành phần điều hòa (Regularization):")
    st.latex(r"\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)")

    st.subheader("4. Đánh giá Mô hình (Metrics)")
    st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
    st.latex(r"\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}")

# ---------------------------------------------------------
# Trang 5. BASELINE & ĐÁNH GIÁ MODELS
# ---------------------------------------------------------
elif menu == "5. Baseline & Đánh giá Models":
    st.title("⚖️ Baseline & So sánh hiệu năng các Models")
    
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

    df_results = pd.DataFrame(results)
    st.table(df_results.style.highlight_max(axis=0, color='lightgreen'))

    st.subheader("So sánh Tỷ lệ Dự đoán giữa các Models (Accuracy & F1-Score)")
    fig_metric, ax_metric = plt.subplots(figsize=(8, 4))
    df_results.plot(x='Model', y=['Accuracy', 'F1-Score', 'ROC-AUC'], kind='bar', ax=ax_metric)
    plt.xticks(rotation=0)
    plt.ylim(0, 1.1)
    st.pyplot(fig_metric)

    st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
    cols_cm = st.columns(3)
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        with cols_cm[i]:
            st.markdown(f"**{name}**")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
            ax_cm.set_xlabel('Dự đoán')
            ax_cm.set_ylabel('Thực tế')
            st.pyplot(fig_cm)

# ---------------------------------------------------------
# Trang 6. CHẨN ĐOÁN SỐT XUẤT HUYẾT
# ---------------------------------------------------------
elif menu == "6. Chẩn đoán Sốt xuất huyết":
    st.title("🩺 Chẩn đoán Sốt xuất huyết")
    st.write("Nhập thông số xét nghiệm của bệnh nhân để dự đoán xác suất bị bệnh:")

    user_input_raw = {}
    cols = st.columns(3)
    
    # Tạo giao diện nhập liệu từ dữ liệu gốc
    for i, col_name in enumerate(X_raw.columns):
        col_idx = i % 3
        with cols[col_idx]:
            if np.issubdtype(X_raw[col_name].dtype, np.number):
                min_val = float(X_raw[col_name].min())
                max_val = float(X_raw[col_name].max())
                mean_val = float(X_raw[col_name].mean())
                
                user_input_raw[col_name] = st.number_input(
                    f"{col_name}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )
            else:
                unique_vals = list(X_raw[col_name].dropna().unique())
                user_input_raw[col_name] = st.selectbox(
                    f"{col_name}",
                    options=unique_vals
                )

    # Xử lý chuẩn hóa input_df sao cho khớp chuẩn các cột của X_train
    input_df_raw = pd.DataFrame([user_input_raw])
    input_df_encoded = pd.get_dummies(input_df_raw)
    
    # Đồng bộ số cột đúng như tập huấn luyện X
    input_df = input_df_encoded.reindex(columns=X.columns, fill_value=0)

    if st.button("🔍 Tiến hành Chẩn đoán"):
        st.subheader("📋 Kết quả dự đoán từ các Mô hình:")
        
        diag_cols = st.columns(3)
        for i, (name, model) in enumerate(models.items()):
            proba = model.predict_proba(input_df)[0][1] * 100
            pred = model.predict(input_df)[0]
            
            with diag_cols[i]:
                st.markdown(f"#### **{name}**")
                if pred == 1:
                    st.error(f"🚨 **Dương tính (Bị bệnh)**")
                else:
                    st.success(f"✅ **Âm tính (Không bị bệnh)**")
                
                st.metric(label="Tỷ lệ/Xác suất nguy cơ", value=f"{proba:.2f}%")
