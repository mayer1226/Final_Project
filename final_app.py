"""
HỆ THỐNG TÌM KIẾM VÀ GỢI Ý XE MÁY CŨ
=====================================

Ứng dụng web tìm kiếm và gợi ý xe máy cũ thông minh sử dụng Machine Learning.

Author: Hoàng Phúc & Bích Thủy
Version: 2.0.1 (Optimized)
Date: 2025-11-29
Python: 3.9+
Framework: Streamlit 1.31.0

Features:
- Hybrid Search (TF-IDF + Content-based)
- K-Means Clustering (K=5)
- Similar Bike Recommendations
- Advanced Analytics Dashboard
- Admin Panel with Export

Performance Improvements:
- Fixed duplicate tab keys
- Removed redundant code
- Enhanced error handling
- Memory optimization with gc
- Better caching strategy
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.sparse import csr_matrix, hstack
import gc
import warnings

import json
from pathlib import Path
import uuid

# ==============================
# 📁 DATA STORAGE CONFIG
# ==============================
USER_LISTINGS_FILE = "user_listings.parquet"
USER_LISTINGS_BACKUP = "user_listings_backup.parquet"

def init_user_listings():
    """Initialize user listings file if not exists"""
    if not Path(USER_LISTINGS_FILE).exists():
        # Create empty dataframe with same structure as main df
        empty_df = pd.DataFrame(columns=[
            'listing_id', 'brand', 'model', 'price', 'km_driven', 'age',
            'vehicle_type', 'vehicle_type_display', 'engine_capacity_num',
            'engine_capacity', 'origin_num', 'origin', 'location',
            'description', 'cluster', 'created_at', 'user_name', 'user_phone'
        ])
        empty_df.to_parquet(USER_LISTINGS_FILE, index=False)
        return empty_df
    return pd.read_parquet(USER_LISTINGS_FILE)

def load_all_bikes():
    """Load both original + user listings"""
    original_df = load_data()  # Your existing function
    user_df = init_user_listings()
    
    if len(user_df) > 0:
        # Combine and reset index
        combined = pd.concat([original_df, user_df], ignore_index=True)
        return combined
    return original_df

def predict_cluster_for_new_bike(bike_data):
    """
    Predict cluster for new bike using trained clustering model
    
    Args:
        bike_data: dict with keys:
            - price (float)
            - km_driven (int)
            - age (int)
            - vehicle_type (int): 0=Xe số, 1=Tay ga, 2=Côn tay, 3=Điện
            - engine_capacity_num (int): 0=100-175cc, 1=50-100cc, 2=<50cc, 3=>175cc
            - origin_num (int): 0=Việt Nam, 1=Nhật, 2=Thái, 3=Trung Quốc, 4=Other
    
    Returns:
        cluster_id (int)
    """
    try:
        if cluster_model is None or cluster_scaler is None:
            return predict_cluster_rule_based(bike_data)
        
        # Extract basic features
        price = float(bike_data.get('price', 0))
        km_driven = int(bike_data.get('km_driven', 0))
        age = int(bike_data.get('age', 0))
        vehicle_type = int(bike_data.get('vehicle_type', 1))  # Default: Tay ga
        engine_capacity_num = int(bike_data.get('engine_capacity_num', 0))  # Default: 100-175cc
        origin_num = int(bike_data.get('origin_num', 0))  # Default: Việt Nam
        
        # Calculate log_km (as in training)
        log_km = np.log1p(km_driven)
        
        # Create vehicle type one-hot (only 2 columns in model: Tay ga, Xe số)
        vtype_tay_ga = 1 if vehicle_type == 1 else 0
        vtype_xe_so = 1 if vehicle_type == 0 else 0
        
        # Create engine capacity one-hot (3 columns: 50-100cc, <50cc, >175cc)
        engine_50_100 = 1 if engine_capacity_num == 1 else 0
        engine_duoi_50 = 1 if engine_capacity_num == 2 else 0
        engine_tren_175 = 1 if engine_capacity_num == 3 else 0
        
        # Create origin one-hot (9 columns)
        origin_my = 1 if origin_num == 5 else 0  # Mỹ
        origin_nhat = 1 if origin_num == 1 else 0  # Nhật Bản
        origin_other = 1 if origin_num == 4 else 0  # Other
        origin_thai = 1 if origin_num == 2 else 0  # Thái Lan
        origin_trung = 1 if origin_num == 3 else 0  # Trung Quốc
        origin_viet = 1 if origin_num == 0 else 0  # Việt Nam
        origin_dai_loan = 1 if origin_num == 6 else 0  # Đài Loan
        origin_duc = 1 if origin_num == 7 else 0  # Đức
        origin_an_do = 1 if origin_num == 8 else 0  # Ấn Độ
        
        # Build DataFrame with exact column names as in training
        bike_df = pd.DataFrame([{
            'price': price,
            'log_km': log_km,
            'age': age,
            'vtype_Tay ga': vtype_tay_ga,
            'vtype_Xe số': vtype_xe_so,
            'engine_capacity_num': engine_capacity_num,
            'engine_50 - 100 cc': engine_50_100,
            'engine_Dưới 50 cc': engine_duoi_50,
            'engine_Trên 175 cc': engine_tren_175,
            'origin_num': origin_num,
            'origin_Mỹ': origin_my,
            'origin_Nhật Bản': origin_nhat,
            'origin_Other': origin_other,
            'origin_Thái Lan': origin_thai,
            'origin_Trung Quốc': origin_trung,
            'origin_Việt Nam': origin_viet,
            'origin_Đài Loan': origin_dai_loan,
            'origin_Đức': origin_duc,
            'origin_Ấn Độ': origin_an_do
        }])
        
        # Scale features using ColumnTransformer
        features_scaled = cluster_scaler.transform(bike_df)
        
        # Predict cluster
        cluster_id = cluster_model.predict(features_scaled)[0]
        
        return int(cluster_id)
    
    except Exception as e:
        st.warning(f"⚠️ Lỗi dự đoán cluster: {e}. Sử dụng rule-based.")
        return predict_cluster_rule_based(bike_data)

def predict_cluster_rule_based(bike_data):
    """
    Fallback rule-based prediction (khi model fail)
    """
    price = bike_data.get('price', 0)
    km = bike_data.get('km_driven', 0)
    age = bike_data.get('age', 0)
    
    # Simple rules based on 5 clusters
    if price < 15 and km > 30000:
        return 0  # Xe Cũ Giá Rẻ - Km Cao
    elif price > 80:
        return 1  # Hạng Sang Cao Cấp
    elif age <= 2 and km < 5000:
        return 4  # Xe Mới - Ít Sử Dụng
    elif 30 <= price <= 80 and km < 20000:
        return 3  # Trung Cao Cấp
    else:
        return 2  # Phổ Thông Đại Trà

def save_new_listing(listing_data):
    """
    Save new bike listing to file
    
    Args:
        listing_data: dict with all bike info
    
    Returns:
        success (bool), message (str)
    """
    try:
        # Load existing listings
        user_df = init_user_listings()
        
        # Create backup
        if len(user_df) > 0:
            user_df.to_parquet(USER_LISTINGS_BACKUP, index=False)
        
        # Add new listing
        new_row = pd.DataFrame([listing_data])
        updated_df = pd.concat([user_df, new_row], ignore_index=True)
        
        # Save
        updated_df.to_parquet(USER_LISTINGS_FILE, index=False)
        
        return True, "✅ Đăng tin thành công!"
    
    except Exception as e:
        return False, f"❌ Lỗi lưu dữ liệu: {str(e)}"

def delete_listing(listing_id):
    """
    Delete a bike listing by ID
    
    Args:
        listing_id: unique listing ID
    
    Returns:
        success (bool), message (str)
    """
    try:
        # Load existing listings
        user_df = init_user_listings()
        
        if len(user_df) == 0:
            return False, "❌ Không có tin đăng nào để xóa"
        
        # Check if listing exists
        if listing_id not in user_df['listing_id'].values:
            return False, "❌ Không tìm thấy tin đăng này"
        
        # Create backup
        user_df.to_parquet(USER_LISTINGS_BACKUP, index=False)
        
        # Remove listing
        updated_df = user_df[user_df['listing_id'] != listing_id]
        
        # Save
        updated_df.to_parquet(USER_LISTINGS_FILE, index=False)
        
        # Clear cache to reload data
        st.cache_data.clear()
        
        return True, "✅ Đã xóa tin đăng thành công!"
    
    except Exception as e:
        return False, f"❌ Lỗi xóa dữ liệu: {str(e)}"   
# ==============================
# 🔧 OPTIMIZATION SETTINGS
# ==============================
warnings.filterwarnings('ignore')
plt.ioff()  # Turn off interactive mode
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['agg.path.chunksize'] = 10000

# ==============================
# 📱 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Hệ Thống Xe Máy Cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🤖 HYBRID RECOMMENDER CLASS
# ==============================

class HybridBikeRecommender:
    """
    PHIÊN BẢN TÁCH MODEL / DATAFRAME
    - Model không chứa DataFrame trong file .joblib
    - DataFrame sẽ được load sau và nạp vào model bằng set_dataframe()
    """

    def __init__(self, 
                 tfidf_max_features=5000,
                 brand_model_boost=5,
                 weights=None,
                 verbose=False):

        self.df = None  
        self.tfidf_max_features = tfidf_max_features
        self.brand_model_boost = brand_model_boost
        self.verbose = verbose

        self.weights = weights or {
            "text": 0.35,
            "numeric": 0.45,
            "binary": 0.20
        }

        self.tfidf = None
        self.numeric_scaler = None
        self.text_features = None
        self.numeric_features = None
        self.binary_features = None
        self.combined_features = None

    def set_dataframe(self, df: pd.DataFrame):
        """Gán DataFrame sau khi load model."""
        self.df = df.reset_index(drop=True)

    def build_features(self):
        """Build tất cả features sau khi có DataFrame."""
        if self.df is None:
            raise ValueError("Bạn phải gọi set_dataframe(df) trước khi build features.")

        self.text_features, self.tfidf = self._build_text_features()
        self.numeric_features, self.numeric_scaler = self._build_numeric_features()
        self.binary_features = self._build_binary_features()
        self.combined_features = self._build_combined_matrix()

    def _build_text_features(self):
        df = self.df.copy()
        df["brand_model"] = df["brand"].fillna("") + " " + df["model"].fillna("")
        brand_model_boosted = (df["brand_model"] + " ") * self.brand_model_boost

        col_list = df.columns.tolist()
        vtype_col = "vehicle_type_display" if "vehicle_type_display" in col_list else "vehicle_type"
        engine_col = "engine_capacity_num" if "engine_capacity_num" in col_list else "engine_capacity"
        origin_col = "origin_num" if "origin_num" in col_list else "origin"
        
        text_parts = [brand_model_boosted, df["description"].fillna("")]
        
        if vtype_col in col_list:
            text_parts.append(df[vtype_col].fillna("").astype(str))
        if engine_col in col_list:
            text_parts.append(df[engine_col].fillna("").astype(str))
        if origin_col in col_list:
            text_parts.append(df[origin_col].fillna("").astype(str))
        if "location" in col_list:
            text_parts.append(df["location"].fillna(""))
        
        df["clean_text"] = " ".join([""]*len(text_parts))
        for i, part in enumerate(text_parts):
            if i == 0:
                df["clean_text"] = part
            else:
                df["clean_text"] = df["clean_text"] + " " + part

        tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            dtype=np.float32
        )

        X = tfidf.fit_transform(df["clean_text"])
        return X, tfidf

    def _build_numeric_features(self):
        numeric_cols = ["price", "km_driven", "age"]
        numeric_data = self.df[numeric_cols].fillna(0).astype(np.float32)
        scaler = RobustScaler()
        scaled = scaler.fit_transform(numeric_data).astype(np.float32)
        return scaled, scaler

    def _build_binary_features(self):
        df = self.df
        bool_cols = [
            c for c in df.columns
            if df[c].dropna().isin([0,1,True,False]).all()
        ]

        if not bool_cols:
            return np.zeros((len(df), 0), dtype=np.float32)

        critical = ["xe_chinh_chu"] if "xe_chinh_chu" in bool_cols else []
        normal = [c for c in bool_cols if c not in critical]

        parts = []
        if critical:
            parts.append(df[critical].astype(float).values * 3.0)
        if normal:
            parts.append(df[normal].astype(float).values)

        return np.hstack(parts).astype(np.float32)

    def _build_combined_matrix(self):
        X_text = self.text_features.multiply(self.weights["text"])
        X_num = csr_matrix(self.numeric_features * self.weights["numeric"])
        X_bin = csr_matrix(self.binary_features * self.weights["binary"])
        return hstack([X_text, X_num, X_bin], format="csr")

    def recommend(self, item_id, top_k=5, filter_by_segment=True):
        if self.df is None:
            raise ValueError("Model chưa có DataFrame. Gọi set_dataframe(df) trước.")

        input_vec = self.combined_features[item_id]
        sim = cosine_similarity(input_vec, self.combined_features).flatten()

        vtype_col = "vehicle_type_display" if "vehicle_type_display" in self.df.columns else "vehicle_type"
        if filter_by_segment and vtype_col in self.df.columns:
            seg = self.df.iloc[item_id][vtype_col]
            mask = (self.df[vtype_col] == seg).values
            sim[~mask] = -10

        sim[item_id] = -10
        top_idx = np.argsort(sim)[::-1][:top_k]

        out = self.df.iloc[top_idx].copy()
        out["similarity_score"] = sim[top_idx]
        out["position"] = top_idx
        return out.reset_index(drop=True)
    
    def search(self, query, top_k=10):
        """Search using hybrid features"""
        if self.df is None or self.combined_features is None:
            raise ValueError("Model chưa sẵn sàng. Gọi set_dataframe() và build_features() trước.")
        
        query_text = query.lower()
        query_tfidf = self.tfidf.transform([query_text])
        
        query_numeric = np.zeros((1, self.numeric_features.shape[1]), dtype=np.float32)
        query_binary = np.zeros((1, self.binary_features.shape[1]), dtype=np.float32)
        
        query_vec = hstack([
            query_tfidf.multiply(self.weights["text"]),
            csr_matrix(query_numeric * self.weights["numeric"]),
            csr_matrix(query_binary * self.weights["binary"])
        ], format="csr")
        
        similarities = cosine_similarity(query_vec, self.combined_features).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results['search_score'] = similarities[top_indices]
        results['position'] = top_indices
        
        return results.reset_index(drop=True)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

# ==============================
# 🎨 APPLE DESIGN CSS
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apple-style Global Font */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', 'SF Pro Display', sans-serif !important;
        letter-spacing: -0.01em;
    }
    
    /* Frosted Glass Header */
    .frosted-header {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        padding: 24px 32px;
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        margin-bottom: 32px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Apple Metric Card with Frosted Glass */
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border: 0.5px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }
    
    /* Apple-style Bike Card */
    .bike-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border: 0.5px solid rgba(0, 0, 0, 0.08);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .bike-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
        border-color: rgba(0, 122, 255, 0.3);
    }
    
    /* Apple Blue Accent */
    .apple-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Breadcrumb Navigation */
    .breadcrumb {
        font-size: 13px;
        color: #6e6e73;
        margin-bottom: 16px;
        font-weight: 400;
    }
    
    .breadcrumb a {
        color: #007aff;
        text-decoration: none;
        transition: opacity 0.2s;
    }
    
    .breadcrumb a:hover {
        opacity: 0.7;
    }
    
    .breadcrumb-separator {
        margin: 0 8px;
        color: #d2d2d7;
    }
    
    /* Active Filter Pills */
    .filter-pill {
        display: inline-block;
        background: rgba(0, 122, 255, 0.1);
        color: #007aff;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 500;
        margin: 4px;
        border: 1px solid rgba(0, 122, 255, 0.2);
    }
    
    .filter-pill .close-btn {
        margin-left: 6px;
        cursor: pointer;
        font-weight: 600;
        opacity: 0.6;
    }
    
    .filter-pill .close-btn:hover {
        opacity: 1;
    }
    
    /* Apple-style Buttons - Subtle Colors */
    .stButton > button {
        background: linear-gradient(180deg, #5E9FFF 0%, #4A8FEE 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: -0.02em !important;
        box-shadow: 0 2px 8px rgba(94, 159, 255, 0.2) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(180deg, #4A8FEE 0%, #3A7FDE 100%) !important;
        box-shadow: 0 4px 16px rgba(94, 159, 255, 0.3) !important;
        transform: translateY(-1px);
    }
    
    .stButton > button:disabled {
        background: #E5E5EA !important;
        color: #8E8E93 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }
    
    /* Price Highlight (Apple Orange) */
    .price-highlight {
        color: #ff9500;
        font-weight: 700;
        font-size: 20px;
        letter-spacing: -0.03em;
    }
    
    /* Smooth Section Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 0, 0, 0.06), transparent);
        margin: 32px 0;
    }
    
    /* Clean Header Typography */
    h1, h2, h3, h4 {
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #1d1d1f;
    }
    
    h1 { font-size: 40px; line-height: 1.1; }
    h2 { font-size: 32px; line-height: 1.2; }
    h3 { font-size: 24px; line-height: 1.3; }
    h4 { font-size: 18px; line-height: 1.4; }
    
    /* Soft Text Colors */
    p, span, div {
        color: #1d1d1f;
    }
    
    .secondary-text {
        color: #6e6e73;
    }
    
    /* Search Bar Enhancement */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007aff !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1) !important;
    }
    
    /* Sidebar Apple Style */
    section[data-testid="stSidebar"] {
        background: rgba(247, 247, 247, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Hide the Material Icon text in hamburger button */
    span[data-testid="stIconMaterial"] {
        font-size: 0 !important;
        color: transparent !important;
    }
    
    /* Add hamburger icon */
    span[data-testid="stIconMaterial"]::before {
        content: "☰" !important;
        font-size: 24px !important;
        color: #1d1d1f !important;
        font-weight: 300 !important;
        display: inline-block !important;
    }
    
    /* Style hamburger button */
    button[kind="header"] {
        width: 40px !important;
        height: 40px !important;
        background: transparent !important;
        border: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    button[kind="header"]:hover {
        background: rgba(0, 0, 0, 0.05) !important;
        border-radius: 8px !important;
    }
    
    /* Streamlit Metric Override */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #1d1d1f !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        color: #6e6e73 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 📥 LOAD MODELS & DATA
# ==============================
@st.cache_resource(show_spinner=False, ttl=3600)
def load_clustering_model():
    """Load clustering model (K-Means K=5)"""
    try:
        model = joblib.load('clustering_model.joblib')
        scaler = joblib.load('clustering_scaler.joblib')
        info = joblib.load('clustering_info.joblib')
        return model, scaler, info
    except Exception as e:
        st.error(f"❌ Không thể load clustering model: {e}")
        return None, None, None

@st.cache_resource(show_spinner=False)
def load_hybrid_model():
    """Load hybrid recommender model"""
    try:
        hybrid = HybridBikeRecommender.load('hybrid_model.joblib')
        return hybrid
    except (FileNotFoundError, AttributeError, ModuleNotFoundError) as e:
        st.info(f"ℹ️ Tạo hybrid model mới (không load được model cũ: {type(e).__name__})")
        hybrid = HybridBikeRecommender(
            tfidf_max_features=5000,
            brand_model_boost=5,
            weights={"text": 0.35, "numeric": 0.45, "binary": 0.20},
            verbose=False
        )
        return hybrid
    except Exception as e:
        st.error(f"❌ Lỗi không xác định khi load hybrid model: {e}")
        return None

# Thay thế hàm load_data() hiện tại
@st.cache_data(show_spinner=False, ttl=300)  # Cache 5 phút
def load_data():
    """Load main dataset + user listings"""
    try:
        # Load original data
        df_original = pd.read_parquet('df_clustering.parquet')
        
        # Add missing columns if needed
        if 'engine_capacity' not in df_original.columns and 'engine_capacity_num' in df_original.columns:
            engine_capacity_map = {
                0: "100 - 175 cc",
                1: "50 - 100 cc",
                2: "Dưới 50 cc",
                3: "Trên 175 cc"
            }
            df_original['engine_capacity'] = df_original['engine_capacity_num'].map(engine_capacity_map)
        
        if 'vehicle_type_display' not in df_original.columns and 'vehicle_type' in df_original.columns:
            vehicle_type_map = {
                0: "Xe số",
                1: "Xe tay ga",
                2: "Xe côn tay",
                3: "Xe đạp điện"
            }
            df_original['vehicle_type_display'] = df_original['vehicle_type'].map(vehicle_type_map)
        
        # Load user listings
        user_listings = init_user_listings()
        
        # Combine if user listings exist
        if len(user_listings) > 0:
            # Ensure same columns
            for col in df_original.columns:
                if col not in user_listings.columns:
                    user_listings[col] = None
            
            # Select only common columns
            common_cols = [col for col in df_original.columns if col in user_listings.columns]
            
            df_combined = pd.concat([
                df_original[common_cols],
                user_listings[common_cols]
            ], ignore_index=True)
            
            return df_combined
        
        return df_original
    
    except Exception as e:
        st.error(f"❌ Không thể load dữ liệu: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def initialize_hybrid_model(_hybrid_model, _df):
    """Initialize and build features for hybrid model (cached)"""
    if _hybrid_model is not None and len(_df) > 0:
        _hybrid_model.set_dataframe(_df)
        _hybrid_model.build_features()
    return _hybrid_model

# Load models
cluster_model, cluster_scaler, cluster_info = load_clustering_model()
hybrid_model_raw = load_hybrid_model()
df = load_data()

# Initialize hybrid model with caching
hybrid_model = initialize_hybrid_model(hybrid_model_raw, df)

# Cluster labels and colors
if cluster_info and 'cluster_labels' in cluster_info:
    cluster_labels = cluster_info['cluster_labels']
else:
    cluster_labels = {
        0: "Xe Cũ Giá Rẻ",
        1: "Xe Hạng Sang",
        2: "Xe Phổ Thông",
        3: "Xe Trung-Cao Cấp",
        4: "Xe Ít Dùng"
    }

cluster_colors = {
    0: "#3498db",
    1: "#e74c3c",
    2: "#2ecc71",
    3: "#f39c12",
    4: "#9b59b6"
}

# ==============================
# 🔧 HELPER FUNCTIONS
# ==============================

def search_items(query, df_search, top_k=10):
    """Text search using TF-IDF"""
    if len(df_search) == 0:
        return pd.DataFrame()
    
    try:
        # Use simple TF-IDF search for accuracy
        search_parts = []
        
        if 'brand' in df_search.columns:
            search_parts.append(df_search['brand'].fillna(''))
        
        if 'model' in df_search.columns:
            search_parts.append(df_search['model'].fillna(''))
        
        if 'vehicle_type_display' in df_search.columns:
            search_parts.append(df_search['vehicle_type_display'].fillna(''))
        elif 'vtype_display' in df_search.columns:
            search_parts.append(df_search['vtype_display'].fillna(''))
        
        # Add location for better search
        if 'location' in df_search.columns:
            search_parts.append(df_search['location'].fillna(''))
        
        if 'description' in df_search.columns:
            search_parts.append(df_search['description'].fillna(''))
        
        # Add cluster label if exists
        if 'cluster_label' in df_search.columns:
            search_parts.append(df_search['cluster_label'].fillna(''))
        
        if not search_parts:
            return df_search.head(top_k).copy()
        
        # Combine all text fields
        search_text = search_parts[0]
        for part in search_parts[1:]:
            search_text = search_text + ' ' + part
        
        # TF-IDF vectorization with optimized parameters
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),      # Unigrams + Bigrams
            max_features=3000,       # Increased for better coverage
            min_df=1,                # Keep all terms
            lowercase=True,          # Case insensitive
            token_pattern=r'(?u)\b\w+\b'  # Include all words
        )
        tfidf_matrix = vectorizer.fit_transform(search_text)
        query_vec = vectorizer.transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top results (get more first, then filter)
        top_indices = similarities.argsort()[::-1][:min(top_k * 5, len(df_search))]
        results = df_search.iloc[top_indices].copy()
        results['search_score'] = similarities[top_indices]
        
        # Filter only relevant results (score > 0.01 to avoid very low matches)
        results = results[results['search_score'] > 0.01].head(top_k)
        
        # Keep original index - DO NOT reset
        return results
    except Exception as e:
        return df_search.head(top_k).copy()

def apply_filters(df_filter, brands, models, price_range, vehicle_types, locations, engine_capacities=None, km_range=None, age_range=None):
    """Apply multi-criteria filters"""
    filtered = df_filter.copy()
    
    if brands and 'Tất cả' not in brands:
        filtered = filtered[filtered['brand'].isin(brands)]
    
    if models and 'Tất cả' not in models:
        filtered = filtered[filtered['model'].isin(models)]
    
    if vehicle_types and 'Tất cả' not in vehicle_types:
        if 'vehicle_type_display' in filtered.columns:
            filtered = filtered[filtered['vehicle_type_display'].isin(vehicle_types)]
    
    if locations and 'Tất cả' not in locations:
        filtered = filtered[filtered['location'].isin(locations)]
    
    if engine_capacities and 'Tất cả' not in engine_capacities:
        if 'engine_capacity_num' in filtered.columns:
            filtered = filtered[filtered['engine_capacity_num'].astype(str).isin(engine_capacities)]
    
    if price_range:
        min_price, max_price = price_range
        filtered = filtered[(filtered['price'] >= min_price) & (filtered['price'] <= max_price)]
    
    if km_range:
        min_km, max_km = km_range
        filtered = filtered[(filtered['km_driven'] >= min_km) & (filtered['km_driven'] <= max_km)]
    
    if age_range:
        min_age, max_age = age_range
        filtered = filtered[(filtered['age'] >= min_age) & (filtered['age'] <= max_age)]
    
    return filtered

def get_similar_bikes(bike_idx, df, top_k=5):
    """Get similar bikes using hybrid model or fallback to numerical similarity"""
    try:
        if hybrid_model is not None and hybrid_model.combined_features is not None:
            similar = hybrid_model.recommend(bike_idx, top_k=top_k, filter_by_segment=True)
            if 'position' not in similar.columns:
                similar['position'] = similar.index
            return similar
        
        else:
            features = df[['price', 'km_driven', 'age']].copy()
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            similarities = cosine_similarity([features_scaled[bike_idx]], features_scaled)[0]
            
            similarities[bike_idx] = -1
            
            top_indices = similarities.argsort()[::-1][:top_k]
            similar_bikes = df.iloc[top_indices].copy()
            similar_bikes['similarity'] = similarities[top_indices]
            similar_bikes['position'] = top_indices
            
            return similar_bikes
    except Exception as e:
        return pd.DataFrame()

def get_cluster_badge(cluster_id, cluster_name, cluster_color):
    """Generate cluster badge HTML"""
    return f"""
<div style="
    background-color:{cluster_color};
    display:inline-block;
    color:white;
    padding:8px 15px;
    border-radius:6px;
    font-weight:bold;
    margin:10px 0;">
    🚀 {cluster_name}
</div>
"""

def format_price(price):
    """Format giá tiền"""
    return f"{price:.1f} triệu VNĐ"

def format_km(km):
    """Format số km"""
    return f"{int(km):,} km"

def display_bike_card(bike, view_mode="grid"):
    """Display bike card in grid or list view - Apple E-commerce style"""
    cluster_id = bike['cluster']
    cluster_name = cluster_labels.get(cluster_id, 'N/A')
    cluster_color = cluster_colors.get(cluster_id, '#007aff')
    
    bike_position = bike.get('position', bike.name)
    
    description = ""
    desc_col = None
    
    if 'description_norm' in bike.index:
        desc_col = 'description_norm'
    elif 'description' in bike.index:
        desc_col = 'description'
    
    if desc_col and pd.notna(bike[desc_col]):
        desc_text = str(bike[desc_col]).strip()
        if desc_text and desc_text.lower() != 'nan':
            max_len = 60 if view_mode == "grid" else 150
            description = desc_text[:max_len] + "..." if len(desc_text) > max_len else desc_text
    
    if view_mode == "grid":
        st.markdown(f"""
<div class="bike-card">
    <span class="apple-badge" style="background: linear-gradient(135deg, {cluster_color} 0%, {cluster_color}dd 100%);">
        {cluster_name}
    </span>
    <h4 style="margin:12px 0 8px 0; color:#1d1d1f; font-size:17px; font-weight:600; line-height:1.3; height:46px; overflow:hidden;">
        {bike['brand']} {bike['model']}
    </h4>
    <div style="margin:10px 0;">
        <div class="price-highlight">{format_price(bike['price'])}</div>
    </div>
    <div style="font-size:13px; color:#6e6e73; line-height:1.6; margin:8px 0;">
        📏 {format_km(bike['km_driven'])} • 📅 {int(bike['age'])} năm
    </div>
    <div style="font-size:12px; color:#86868b; font-style:italic; line-height:1.4; height:36px; overflow:hidden; margin-top:8px;">
        {description if description else ''}
    </div>
</div>
""", unsafe_allow_html=True)
        
        if st.button("Xem chi tiết", key=f"card_{bike_position}", use_container_width=True, type="primary"):
            st.session_state.selected_bike_idx = int(bike_position)
            st.session_state.page = "detail"
            st.rerun()
    
    else:
        col_img, col_info, col_action = st.columns([1, 4, 1])
        
        with col_img:
            st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {cluster_color}22 0%, {cluster_color}11 100%);
    border-radius: 12px;
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48px;
    border: 1px solid {cluster_color}33;
">
    🏍️
</div>
""", unsafe_allow_html=True)
        
        with col_info:
            st.markdown(f"""
<div style="padding: 0 16px;">
    <span class="apple-badge" style="background: linear-gradient(135deg, {cluster_color} 0%, {cluster_color}dd 100%);">
        {cluster_name}
    </span>
    <h3 style="margin:8px 0; color:#1d1d1f; font-size:20px; font-weight:600;">
        {bike['brand']} {bike['model']}
    </h3>
    <div class="price-highlight" style="margin:8px 0;">
        {format_price(bike['price'])}
    </div>
    <div style="font-size:14px; color:#6e6e73; line-height:1.6; margin:8px 0;">
        📏 {format_km(bike['km_driven'])} • 📅 {int(bike['age'])} năm • 📍 {bike.get('location', 'N/A')}
    </div>
    <div style="font-size:13px; color:#86868b; line-height:1.5; margin-top:8px;">
        {description if description else 'Thông tin chi tiết sẽ được cập nhật'}
    </div>
</div>
""", unsafe_allow_html=True)
        
        with col_action:
            st.markdown("<div style='padding-top: 40px;'>", unsafe_allow_html=True)
            if st.button("Xem chi tiết", key=f"card_list_{bike_position}", use_container_width=True, type="primary"):
                st.session_state.selected_bike_idx = int(bike_position)
                st.session_state.page = "detail"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)

def show_banner():
    """Display banner for all pages except About"""
    if 'page' in st.session_state and st.session_state.page != "about":
        # Nếu không load được banner thì bỏ qua
        try:
            st.markdown("""
            <style>
                .stImage {
                    margin-top: -6rem !important;
                    margin-bottom: 1rem !important;
                }
            </style>
            """, unsafe_allow_html=True)
            st.image(
                "https://raw.githubusercontent.com/mayer1226/Final_Project/refs/heads/main/banner.jpg",
                use_container_width=True
            )
        except:
            # Không hiển thị gì nếu không load được
            pass

# ==============================
# 📄 PAGE FUNCTIONS
# ==============================
def show_sell_page():
    """Trang đăng bán xe - Apple Style Form"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 16px;
            color: white;
        ">📝 Đăng Tin Bán Xe</h1>
        <p style="
            font-size: 20px;
            opacity: 0.95;
        ">Hệ thống đăng ký bán và hỗ trợ tự động phân loại nhóm xe</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Form container
    with st.form("sell_bike_form", clear_on_submit=True):
        st.markdown("### 📋 Thông Tin Xe")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Brand selection
            all_brands = sorted(df['brand'].unique().tolist())
            brand = st.selectbox(
                "🏢 Hãng xe *",
                options=all_brands,
                help="Chọn hãng xe của bạn"
            )
            
            # Model input
            model = st.text_input(
                "🏍️ Tên model *",
                placeholder="VD: SH Mode, Vision, Wave Alpha...",
                help="Nhập tên model xe"
            )
            
            # Price
            price = st.number_input(
                "💰 Giá bán (triệu VNĐ) *",
                min_value=1.0,
                max_value=500.0,
                value=25.0,
                step=0.5,
                help="Giá bán mong muốn"
            )
            
            # Km driven
            km_driven = st.number_input(
                "📏 Số km đã đi *",
                min_value=0,
                max_value=500000,
                value=10000,
                step=1000,
                help="Tổng số km đã đi"
            )
        
        with col2:
            # Age
            current_year = datetime.now().year
            year_options = list(range(current_year, 1990, -1))
            manufacture_year = st.selectbox(
                "📅 Năm đăng ký *",
                options=year_options,
                help="Năm xe được sản xuất"
            )
            age = current_year - manufacture_year
            
            # Vehicle type
            vehicle_types = {
                "Xe số": 0,
                "Xe tay ga": 1,
                "Xe côn tay": 2,
                "Xe đạp điện": 3
            }
            vehicle_type_display = st.selectbox(
                "🏷️ Loại xe *",
                options=list(vehicle_types.keys())
            )
            vehicle_type = vehicle_types[vehicle_type_display]
            
            # Engine capacity
            engine_options = {
                "Dưới 50 cc": 2,
                "50 - 100 cc": 1,
                "100 - 175 cc": 0,
                "Trên 175 cc": 3
            }
            engine_capacity = st.selectbox(
                "⚙️ Dung tích động cơ *",
                options=list(engine_options.keys())
            )
            engine_capacity_num = engine_options[engine_capacity]
            
            # Origin
            origin_options = {
                "Việt Nam": 0,
                "Nhật Bản": 1,
                "Thái Lan": 2,
                "Trung Quốc": 3,
                "Khác": 4
            }
            origin = st.selectbox(
                "🌍 Xuất xứ *",
                options=list(origin_options.keys())
            )
            origin_num = origin_options[origin]
        
        # Location
        all_locations = sorted(df['location'].unique().tolist())
        location = st.selectbox(
            "📍 Khu vực *",
            options=all_locations,
            help="Chọn khu vực bạn đang ở"
        )
        
        # Description
        description = st.text_area(
            "📝 Mô tả chi tiết *",
            placeholder="VD: Xe chính chủ, bảo dưỡng định kỳ, không tai nạn, phanh đĩa...",
            height=150,
            help="Mô tả chi tiết tình trạng xe"
        )
        
        st.markdown("---")
        st.markdown("### 👤 Thông Tin Liên Hệ")
        
        col3, col4 = st.columns(2)
        
        with col3:
            user_name = st.text_input(
                "👤 Họ tên *",
                placeholder="Nguyễn Văn A"
            )
        
        with col4:
            user_phone = st.text_input(
                "📞 Số điện thoại *",
                placeholder="0912345678"
            )
        
        st.markdown("---")
        
        # Submit button
        col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
        with col_submit2:
            submitted = st.form_submit_button(
                "🚀 Đăng Tin Ngay",
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            # Validation
            errors = []
            
            if not brand:
                errors.append("Vui lòng chọn hãng xe")
            if not model or len(model.strip()) < 2:
                errors.append("Vui lòng nhập tên model (tối thiểu 2 ký tự)")
            if price <= 0:
                errors.append("Giá bán phải lớn hơn 0")
            if not description or len(description.strip()) < 20:
                errors.append("Mô tả phải có tối thiểu 20 ký tự")
            if not user_name or len(user_name.strip()) < 2:
                errors.append("Vui lòng nhập họ tên")
            if not user_phone or len(user_phone.strip()) < 10:
                errors.append("Số điện thoại không hợp lệ")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                # Predict cluster
                bike_features = {
                    'price': float(price),
                    'km_driven': int(km_driven),
                    'age': int(age),
                    'vehicle_type': int(vehicle_type),  # From form
                    'engine_capacity_num': int(engine_capacity_num),  # From form
                    'origin_num': int(origin_num)  # From form
                }
                predicted_cluster = predict_cluster_for_new_bike(bike_features)
                cluster_name = cluster_labels.get(predicted_cluster, f"Nhóm {predicted_cluster}")
                cluster_color = cluster_colors.get(predicted_cluster, "#667eea")
                
                # Show prediction
                st.success("🎯 Hệ thống đã phân tích xe của bạn!")
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {cluster_color} 0%, {cluster_color}dd 100%);
                    padding: 24px;
                    border-radius: 16px;
                    color: white;
                    text-align: center;
                    margin: 20px 0;
                    box-shadow: 0 8px 24px {cluster_color}40;
                ">
                    <h3 style="margin: 0 0 12px 0; color: white;">🚀 Phân khúc xe của bạn</h3>
                    <h2 style="margin: 0; font-size: 32px; color: white;">{cluster_name}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Prepare listing data
                listing_data = {
                    'listing_id': str(uuid.uuid4()),
                    'brand': brand,
                    'model': model.strip(),
                    'price': float(price),
                    'km_driven': int(km_driven),
                    'age': int(age),
                    'vehicle_type': int(vehicle_type),
                    'vehicle_type_display': vehicle_type_display,
                    'engine_capacity_num': int(engine_capacity_num),
                    'engine_capacity': engine_capacity,
                    'origin_num': int(origin_num),
                    'origin': origin,
                    'location': location,
                    'description': description.strip(),
                    'cluster': int(predicted_cluster),
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'user_name': user_name.strip(),
                    'user_phone': user_phone.strip()
                }
                
                # Save to file
                success, message = save_new_listing(listing_data)
                
                if success:
                    st.success(message)
                    # st.balloons()
                    
                    # Show similar bikes
                    st.markdown("### 🎯 Xe tương tự đang bán")
                    similar = df[df['cluster'] == predicted_cluster].head(3)
                    
                    cols = st.columns(3)
                    for idx, (_, bike) in enumerate(similar.iterrows()):
                        with cols[idx]:
                            st.markdown(f"""
                            <div style="
                                background: white;
                                padding: 16px;
                                border-radius: 12px;
                                border: 1px solid #e0e0e0;
                            ">
                                <h4 style="margin: 0 0 8px 0;">{bike['brand']} {bike['model']}</h4>
                                <p style="margin: 4px 0; color: #667eea; font-weight: 600;">
                                    💰 {format_price(bike['price'])}
                                </p>
                                <p style="margin: 4px 0; font-size: 13px; color: #666;">
                                    📏 {format_km(bike['km_driven'])} • 📅 {int(bike['age'])} năm
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.info("💡 **Mẹo:** Xe của bạn đã được lưu và có thể tìm kiếm ngay trong trang Tìm Kiếm!")
                    
                    # Clear cache to reload data
                    st.cache_data.clear()
                    
                else:
                    st.error(message)
    
    # ===== DANH SÁCH XE ĐÃ ĐĂNG =====
    st.markdown("---")
    st.markdown("## 📋 Danh Sách Xe Đã Đăng Bán")
    
    # Load user listings
    user_df = init_user_listings()
    
    if len(user_df) > 0:
        st.info(f"📊 **Tổng số xe đã đăng:** {len(user_df)} xe")
        
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            filter_brand = st.selectbox(
                "🏢 Lọc theo hãng",
                options=["Tất cả"] + sorted(user_df['brand'].unique().tolist()),
                key="filter_brand_sell"
            )
        
        with col_f2:
            filter_cluster = st.selectbox(
                "🎯 Lọc theo phân khúc",
                options=["Tất cả"] + [f"{k}: {v}" for k, v in cluster_labels.items()],
                key="filter_cluster_sell"
            )
        
        with col_f3:
            sort_by = st.selectbox(
                "📊 Sắp xếp theo",
                options=["Mới nhất", "Giá thấp → cao", "Giá cao → thấp", "Km ít nhất"],
                key="sort_sell"
            )
        
        # Apply filters
        filtered_df = user_df.copy()
        
        if filter_brand != "Tất cả":
            filtered_df = filtered_df[filtered_df['brand'] == filter_brand]
        
        if filter_cluster != "Tất cả":
            cluster_id = int(filter_cluster.split(":")[0])
            filtered_df = filtered_df[filtered_df['cluster'] == cluster_id]
        
        # Apply sorting
        if sort_by == "Mới nhất":
            filtered_df = filtered_df.sort_values('created_at', ascending=False)
        elif sort_by == "Giá thấp → cao":
            filtered_df = filtered_df.sort_values('price', ascending=True)
        elif sort_by == "Giá cao → thấp":
            filtered_df = filtered_df.sort_values('price', ascending=False)
        elif sort_by == "Km ít nhất":
            filtered_df = filtered_df.sort_values('km_driven', ascending=True)
        
        if len(filtered_df) == 0:
            st.warning("⚠️ Không tìm thấy xe nào phù hợp với bộ lọc.")
        else:
            st.success(f"✅ Tìm thấy **{len(filtered_df)}** xe")
            
            # Display bikes in grid (3 columns)
            for i in range(0, len(filtered_df), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(filtered_df):
                        bike = filtered_df.iloc[i + j]
                        cluster_id = int(bike['cluster'])
                        cluster_name = cluster_labels.get(cluster_id, f"Nhóm {cluster_id}")
                        cluster_color = cluster_colors.get(cluster_id, "#667eea")
                        listing_id = bike['listing_id']
                        
                        with cols[j]:
                            # Build HTML card
                            card_html = f"""
<div style="background: white; border-radius: 16px; padding: 20px; border: 2px solid #f0f0f0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
    <div style="background: {cluster_color}; color: white; padding: 6px 12px; border-radius: 8px; font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 12px;">{cluster_name}</div>
    <h3 style="margin: 0 0 12px 0; font-size: 20px; color: #1a1a1a;">{bike['brand']} {bike['model']}</h3>
    <div style="font-size: 24px; font-weight: 700; color: {cluster_color}; margin-bottom: 16px;">💰 {format_price(bike['price'])}</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; font-size: 13px; color: #666;">
        <div>📏 {format_km(bike['km_driven'])}</div>
        <div>📅 {int(bike['age'])} năm</div>
        <div>⚙️ {bike.get('engine_capacity', 'N/A')}</div>
        <div>🏍️ {bike.get('vehicle_type_display', 'N/A')}</div>
    </div>
    <div style="font-size: 13px; color: #888; margin-bottom: 12px;">📍 {bike['location']}</div>
    <div style="font-size: 13px; color: #666; font-style: italic; line-height: 1.5; margin-bottom: 16px; max-height: 60px; overflow: hidden;">"{bike['description'][:100]}..."</div>
    <div style="border-top: 1px solid #f0f0f0; padding-top: 12px; font-size: 13px;">
        <div style="color: #333; margin-bottom: 4px;">👤 <strong>{bike['user_name']}</strong></div>
        <div style="color: #667eea; font-weight: 600;">📞 {bike['user_phone']}</div>
    </div>
    <div style="font-size: 11px; color: #999; margin-top: 12px; text-align: right;">🕒 {bike['created_at']}</div>
</div>
"""
                            st.markdown(card_html, unsafe_allow_html=True)
                            
                            # Delete button
                            if st.button("🗑️ Xóa tin", key=f"delete_{listing_id}", use_container_width=True, type="secondary"):
                                success, message = delete_listing(listing_id)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
    else:
        st.info("📭 Chưa có xe nào được đăng bán. Hãy là người đầu tiên!")
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📊 Thống Kê Tin Đăng")
    
    user_listings = init_user_listings()
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("📝 Tổng tin đăng", f"{len(user_listings):,}")
    
    with col_stat2:
        if len(user_listings) > 0:
            avg_price_user = user_listings['price'].mean()
            st.metric("💰 Giá TB", f"{avg_price_user:.1f}M")
        else:
            st.metric("💰 Giá TB", "N/A")
    
    with col_stat3:
        today_listings = 0
        if len(user_listings) > 0:
            today = datetime.now().strftime('%Y-%m-%d')
            today_listings = len(user_listings[user_listings['created_at'].str.startswith(today)])
        st.metric("🆕 Hôm nay", f"{today_listings}")
    
    with col_stat4:
        if len(user_listings) > 0:
            top_brand = user_listings['brand'].value_counts().index[0]
            st.metric("🏆 Hãng phổ biến", top_brand)
        else:
            st.metric("🏆 Hãng phổ biến", "N/A")
def show_home_page():
    """Trang chủ - E-commerce Style"""
    
    # ===== HERO SECTION =====
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
    ">
        <h1 style="
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 12px;
            color: white;
        ">🏍️ Tìm Xe Máy Cũ Như Ý</h1>
        <p style="
            font-size: 16px;
            margin-bottom: 24px;
            opacity: 0.9;
        ">Hệ thống gợi ý xe thông minh với AI - Nhanh chóng, Chính xác, Miễn phí</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Buttons (functional)
    col_cta1, col_cta2, col_cta3 = st.columns([1, 1, 1])
    
    with col_cta1:
        pass
    
    with col_cta2:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔍 Tìm Xe Ngay", use_container_width=True, type="primary", key="hero_search"):
                st.session_state.page = "search"
                st.rerun()
        
        with col_btn2:
            if st.button("📝 Đăng Bán Xe", use_container_width=True, key="hero_sell"):
                st.session_state.page = "sell"
                st.rerun()
    
    with col_cta3:
        pass
    
    st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
    
    # ===== TRUST SIGNALS =====
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 24px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 16px;">
            <div style="font-size: 48px; margin-bottom: 12px;">🏍️</div>
            <div style="font-size: 32px; font-weight: 700; color: #667eea; margin-bottom: 8px;">{len(df):,}</div>
            <div style="font-size: 14px; color: #666;">Xe đang bán</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        user_listings = init_user_listings()
        st.markdown(f"""
        <div style="text-align: center; padding: 24px; background: linear-gradient(135deg, #2ecc7115 0%, #27ae6015 100%); border-radius: 16px;">
            <div style="font-size: 48px; margin-bottom: 12px;">📝</div>
            <div style="font-size: 32px; font-weight: 700; color: #2ecc71; margin-bottom: 8px;">{len(user_listings):,}</div>
            <div style="font-size: 14px; color: #666;">Tin đăng mới</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        n_brands = df['brand'].nunique()
        st.markdown(f"""
        <div style="text-align: center; padding: 24px; background: linear-gradient(135deg, #f39c1215 0%, #e74c3c15 100%); border-radius: 16px;">
            <div style="font-size: 48px; margin-bottom: 12px;">🏢</div>
            <div style="font-size: 32px; font-weight: 700; color: #f39c12; margin-bottom: 8px;">{n_brands}+</div>
            <div style="font-size: 14px; color: #666;">Thương hiệu</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_price = df['price'].mean()
        st.markdown(f"""
        <div style="text-align: center; padding: 24px; background: linear-gradient(135deg, #3498db15 0%, #2980b915 100%); border-radius: 16px;">
            <div style="font-size: 48px; margin-bottom: 12px;">💰</div>
            <div style="font-size: 32px; font-weight: 700; color: #3498db; margin-bottom: 8px;">{avg_price:.1f}M</div>
            <div style="font-size: 14px; color: #666;">Giá trung bình</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 48px 0;'></div>", unsafe_allow_html=True)
    
    # ===== VALUE PROPOSITIONS =====
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px; font-size: 36px;'>✨ Tại Sao Chọn Chúng Tôi?</h2>", unsafe_allow_html=True)
    
    col_v1, col_v2, col_v3 = st.columns(3)
    
    with col_v1:
        st.markdown("""
        <div style="text-align: center; padding: 32px; background: white; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%;">
            <div style="font-size: 64px; margin-bottom: 16px;">🤖</div>
            <h3 style="color: #667eea; margin-bottom: 12px;">AI Thông Minh</h3>
            <p style="color: #666; line-height: 1.8;">
                Hệ thống phân tích và gợi ý xe phù hợp nhất với nhu cầu của bạn bằng Machine Learning
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_v2:
        st.markdown("""
        <div style="text-align: center; padding: 32px; background: white; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%;">
            <div style="font-size: 64px; margin-bottom: 16px;">⚡</div>
            <h3 style="color: #2ecc71; margin-bottom: 12px;">Nhanh Chóng</h3>
            <p style="color: #666; line-height: 1.8;">
                Tìm kiếm và so sánh hàng nghìn xe chỉ trong vài giây. Tiết kiệm thời gian tối đa cho bạn
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_v3:
        st.markdown("""
        <div style="text-align: center; padding: 32px; background: white; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%;">
            <div style="font-size: 64px; margin-bottom: 16px;">💯</div>
            <h3 style="color: #f39c12; margin-bottom: 12px;">Miễn Phí 100%</h3>
            <p style="color: #666; line-height: 1.8;">
                Hoàn toàn miễn phí cho cả người mua và người bán. Không có phí ẩn, không giới hạn
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 48px 0;'></div>", unsafe_allow_html=True)
    
    # ===== HOW IT WORKS =====
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px; font-size: 36px;'>🚀 Cách Thức Hoạt Động</h2>", unsafe_allow_html=True)
    
    col_h1, col_h2, col_h3 = st.columns(3)
    
    with col_h1:
        st.markdown("""
        <div style="text-align: center; padding: 32px;">
            <div style="
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 36px;
                font-weight: 700;
                margin: 0 auto 20px;
                box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
            ">1</div>
            <h3 style="margin-bottom: 12px; color: #333;">🔍 Tìm Kiếm</h3>
            <p style="color: #666; line-height: 1.8;">
                Nhập từ khóa hoặc sử dụng bộ lọc để tìm xe phù hợp
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h2:
        st.markdown("""
        <div style="text-align: center; padding: 32px;">
            <div style="
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 36px;
                font-weight: 700;
                margin: 0 auto 20px;
                box-shadow: 0 8px 24px rgba(46, 204, 113, 0.3);
            ">2</div>
            <h3 style="margin-bottom: 12px; color: #333;">📊 So Sánh</h3>
            <p style="color: #666; line-height: 1.8;">
                Xem chi tiết, so sánh giá cả và tính năng các xe tương tự
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h3:
        st.markdown("""
        <div style="text-align: center; padding: 32px;">
            <div style="
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%);
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 36px;
                font-weight: 700;
                margin: 0 auto 20px;
                box-shadow: 0 8px 24px rgba(243, 156, 18, 0.3);
            ">3</div>
            <h3 style="margin-bottom: 12px; color: #333;">📞 Liên Hệ</h3>
            <p style="color: #666; line-height: 1.8;">
                Liên hệ trực tiếp với người bán để xem xe và thương lượng
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 48px 0;'></div>", unsafe_allow_html=True)
    
    # ===== FEATURED BIKES =====
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px; font-size: 36px;'>🔥 Xe Nổi Bật</h2>", unsafe_allow_html=True)
    
    # Get featured bikes (newest, best price, most popular)
    featured_bikes = df.sort_values('age').head(6)
    
    for i in range(0, len(featured_bikes), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(featured_bikes):
                bike = featured_bikes.iloc[i + j]
                cluster_id = int(bike['cluster'])
                cluster_name = cluster_labels.get(cluster_id, 'N/A')
                cluster_color = cluster_colors.get(cluster_id, '#667eea')
                
                with cols[j]:
                    card_html = f"""
<div style="background: white; border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 24px; height: 100%;">
    <div style="background: {cluster_color}; color: white; padding: 6px 12px; border-radius: 8px; font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 12px;">{cluster_name}</div>
    <h3 style="margin: 0 0 12px 0; font-size: 18px; color: #1a1a1a;">{bike['brand']} {bike['model']}</h3>
    <div style="font-size: 24px; font-weight: 700; color: {cluster_color}; margin-bottom: 16px;">💰 {format_price(bike['price'])}</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; color: #666;">
        <div>📏 {format_km(bike['km_driven'])}</div>
        <div>📅 {int(bike['age'])} năm</div>
    </div>
    <div style="font-size: 13px; color: #888; margin-top: 12px;">📍 {bike['location']}</div>
</div>
"""
                    st.markdown(card_html, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 48px 0;'></div>", unsafe_allow_html=True)
    
    # ===== CLUSTERS OVERVIEW =====
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px; font-size: 36px;'>🎯 Phân Khúc Xe Máy</h2>", unsafe_allow_html=True)
    
    for cluster_id in sorted(cluster_labels.keys()):
        cluster_name = cluster_labels[cluster_id]
        cluster_color = cluster_colors[cluster_id]
        cluster_data = df[df['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {cluster_color}15 0%, {cluster_color}08 100%);
            border-left: 4px solid {cluster_color};
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 16px;
        ">
            <h3 style="color: {cluster_color}; margin: 0 0 8px 0;">{cluster_name}</h3>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 14px; color: #666;">
                <div>📊 <strong>{len(cluster_data):,}</strong> xe</div>
                <div>💰 Giá TB: <strong>{format_price(cluster_data['price'].mean())}</strong></div>
                <div>📏 Km TB: <strong>{format_km(cluster_data['km_driven'].mean())}</strong></div>
                <div>📅 Tuổi TB: <strong>{cluster_data['age'].mean():.1f} năm</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_search_page():
    """Trang tìm kiếm - Apple E-commerce Style"""
    
    st.markdown("<h2 style='margin-bottom: 24px;'>🔍 Tìm Kiếm Xe Máy</h2>", unsafe_allow_html=True)
    
    col_search1, col_search2, col_search3 = st.columns([6, 1, 1])
    with col_search1:
        query = st.text_input(
            "🔍 Tìm kiếm xe", 
            value="", 
            placeholder="Tìm theo tên xe, hãng, model, hoặc mô tả...", 
            key="search_query",
            label_visibility="collapsed"
        )
    with col_search2:
        search_clicked = st.button("🔍 Tìm", use_container_width=True, type="primary")
    with col_search3:
        filter_expanded = st.button("⚙️ Lọc", use_container_width=True)
    
    if 'show_filters' not in st.session_state:
        st.session_state.show_filters = False
    
    if filter_expanded:
        st.session_state.show_filters = not st.session_state.show_filters
    
    if st.session_state.show_filters:
        with st.container():
            st.markdown("### ⚙️ Bộ Lọc Nâng Cao")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                all_brands = ['Tất cả'] + sorted(df['brand'].unique().tolist())
                selected_brands = st.multiselect("🏢 Hãng", options=all_brands, default=['Tất cả'])
            
            with col2:
                if selected_brands and 'Tất cả' not in selected_brands:
                    available_models = df[df['brand'].isin(selected_brands)]['model'].unique().tolist()
                else:
                    available_models = df['model'].unique().tolist()
                
                all_models = ['Tất cả'] + sorted(available_models)
                selected_models = st.multiselect("📦 Model", options=all_models, default=['Tất cả'])
            
            with col3:
                if 'vehicle_type_display' in df.columns:
                    all_vehicle_types = ['Tất cả'] + sorted(df['vehicle_type_display'].dropna().unique().tolist())
                    selected_vehicle_types = st.multiselect("🏷️ Loại xe", options=all_vehicle_types, default=['Tất cả'])
                else:
                    selected_vehicle_types = ['Tất cả']
            
            with col4:
                if 'engine_capacity_num' in df.columns:
                    all_engine_capacities = ['Tất cả'] + sorted([str(x) for x in df['engine_capacity_num'].dropna().unique().tolist()])
                    selected_engine_capacities = st.multiselect("⚙️ Phân khối", options=all_engine_capacities, default=['Tất cả'])
                else:
                    selected_engine_capacities = ['Tất cả']
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                price_range = st.slider("💰 Khoảng giá (triệu)", 
                                       float(df['price'].min()), 
                                       float(df['price'].max()), 
                                       (float(df['price'].min()), float(df['price'].max())))
            
            with col6:
                km_range = st.slider("🛣️ Km đã đi", 
                                    0.0, 
                                    float(df['km_driven'].max()), 
                                    (0.0, float(df['km_driven'].max())))
            
            with col7:
                age_range = st.slider("📅 Tuổi xe (năm)", 
                                     0.0, 
                                     float(df['age'].max()), 
                                     (0.0, float(df['age'].max())))
            
            with col8:
                all_locations = ['Tất cả'] + sorted(df['location'].unique().tolist())
                selected_locations = st.multiselect("📍 Khu vực", options=all_locations, default=['Tất cả'])
    else:
        selected_brands = ['Tất cả']
        selected_models = ['Tất cả']
        selected_vehicle_types = ['Tất cả']
        selected_engine_capacities = ['Tất cả']
        price_range = (float(df['price'].min()), float(df['price'].max()))
        km_range = (0.0, float(df['km_driven'].max()))
        age_range = (0.0, float(df['age'].max()))
        selected_locations = ['Tất cả']
    
    active_filters = []
    if selected_brands and 'Tất cả' not in selected_brands:
        active_filters.extend([f"Hãng: {b}" for b in selected_brands])
    if selected_models and 'Tất cả' not in selected_models:
        active_filters.extend([f"Model: {m}" for m in selected_models])
    if selected_vehicle_types and 'Tất cả' not in selected_vehicle_types:
        active_filters.extend([f"Loại: {v}" for v in selected_vehicle_types])
    if selected_engine_capacities and 'Tất cả' not in selected_engine_capacities:
        # Map engine capacity numbers to text
        engine_map = {
            "0": "100-175cc",
            "1": "50-100cc", 
            "2": "Dưới 50cc",
            "3": "Trên 175cc"
        }
        engine_texts = [engine_map.get(e, e) for e in selected_engine_capacities]
        active_filters.extend([f"Phân khối: {e}" for e in engine_texts])
    if selected_locations and 'Tất cả' not in selected_locations:
        active_filters.extend([f"Khu vực: {l}" for l in selected_locations])
    if price_range != (float(df['price'].min()), float(df['price'].max())):
        active_filters.append(f"Giá: {price_range[0]:.0f}-{price_range[1]:.0f}M")
    if km_range != (0.0, float(df['km_driven'].max())):
        active_filters.append(f"Km: {km_range[0]:.0f}-{km_range[1]:.0f}")
    if age_range != (0.0, float(df['age'].max())):
        active_filters.append(f"Tuổi: {age_range[0]:.0f}-{age_range[1]:.0f} năm")
    
    if active_filters:
        st.markdown("#### Đang lọc theo:")
        filter_html = " ".join([f'<span class="filter-pill">{f} <span class="close-btn">×</span></span>' for f in active_filters])
        st.markdown(f'<div style="margin-bottom: 16px;">{filter_html}</div>', unsafe_allow_html=True)
    
    if search_clicked and query:
        st.session_state.last_query = query
    
    current_query = query if query else st.session_state.get('last_query', '')
    
    # Step 1: Search first (if there's a query)
    if current_query:
        search_results = search_items(current_query, df, top_k=200)  # Get more results for filtering
        if len(search_results) > 0:
            st.markdown(f"""
            <div style="
                background: rgba(0, 122, 255, 0.08);
                padding: 12px 20px;
                border-radius: 12px;
                border-left: 4px solid #007aff;
                margin: 16px 0;
            ">
                <span style="color: #007aff; font-weight: 600;">🔍 Đang tìm kiếm:</span>
                <span style="color: #1d1d1f; font-weight: 500;"> "{current_query}"</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        search_results = df.copy()
        if 'position' not in search_results.columns:
            search_results['position'] = search_results.index.tolist()
    
    if query:
        st.session_state.last_query = query
    
    # Step 2: Apply filters on search results
    filtered_df = apply_filters(
        search_results, 
        selected_brands, 
        selected_models, 
        price_range, 
        selected_vehicle_types, 
        selected_locations,
        selected_engine_capacities,
        km_range,
        age_range
    )
    
    # Check if no results
    if len(filtered_df) == 0:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 60px 20px;
            background: rgba(247, 247, 247, 0.5);
            border-radius: 20px;
            margin: 40px 0;
        ">
            <div style="font-size: 64px; margin-bottom: 16px;">🔍</div>
            <h3 style="color: #1d1d1f; margin-bottom: 8px;">Không tìm thấy kết quả</h3>
            <p style="color: #6e6e73; font-size: 15px;">
                Thử điều chỉnh bộ lọc hoặc từ khóa tìm kiếm của bạn
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        col_result, col_sort, col_view = st.columns([3, 2, 1])
        
        with col_result:
            st.markdown(f"""
            <div style="padding: 12px 0;">
                <span style="color: #6e6e73; font-size: 14px;">Hiển thị</span>
                <span style="color: #1d1d1f; font-weight: 600; font-size: 16px;"> {len(filtered_df)} </span>
                <span style="color: #6e6e73; font-size: 14px;">kết quả</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_sort:
            sort_options = {
                "Mặc định": "default",
                "Giá: Thấp → Cao": "price_asc",
                "Giá: Cao → Thấp": "price_desc",
                "Mới nhất": "age_asc",
                "Cũ nhất": "age_desc",
                "Km: Thấp → Cao": "km_asc",
                "Km: Cao → Thấp": "km_desc"
            }
            sort_choice = st.selectbox("Sắp xếp theo", list(sort_options.keys()), key="sort_select")
            
            sort_key = sort_options[sort_choice]
            if sort_key == "price_asc":
                filtered_df = filtered_df.sort_values('price', ascending=True)
            elif sort_key == "price_desc":
                filtered_df = filtered_df.sort_values('price', ascending=False)
            elif sort_key == "age_asc":
                filtered_df = filtered_df.sort_values('age', ascending=True)
            elif sort_key == "age_desc":
                filtered_df = filtered_df.sort_values('age', ascending=False)
            elif sort_key == "km_asc":
                filtered_df = filtered_df.sort_values('km_driven', ascending=True)
            elif sort_key == "km_desc":
                filtered_df = filtered_df.sort_values('km_driven', ascending=False)
        
        with col_view:
            view_mode = st.radio("Hiển thị", ["🔲 Grid", "📋 List"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("---")
        
        if 'search_page_num' not in st.session_state:
            st.session_state.search_page_num = 0
        
        items_per_page = 9 if view_mode == "🔲 Grid" else 5
        total_pages = max(1, (len(filtered_df) + items_per_page - 1) // items_per_page)
        
        # ✅ FIX: Ensure page_num is valid
        if st.session_state.search_page_num >= total_pages:
            st.session_state.search_page_num = total_pages - 1
        if st.session_state.search_page_num < 0:
            st.session_state.search_page_num = 0
        
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("◀ Trước", disabled=st.session_state.search_page_num == 0, key="prev_page"):
                st.session_state.search_page_num -= 1
                st.rerun()

        with col_page:
            st.markdown(f"<p style='text-align:center; font-size:15px; color:#1d1d1f; font-weight:500;'>Trang {st.session_state.search_page_num + 1} / {total_pages}</p>", 
                    unsafe_allow_html=True)

        with col_next:
            if st.button("Sau ▶", disabled=st.session_state.search_page_num >= total_pages - 1, key="next_page"):
                st.session_state.search_page_num += 1
                st.rerun()
        
        start_idx = st.session_state.search_page_num * items_per_page
        end_idx = start_idx + items_per_page
        page_bikes = filtered_df.iloc[start_idx:end_idx]
        
        if view_mode == "🔲 Grid":
            cols = st.columns(3)
            for idx, (_, bike) in enumerate(page_bikes.iterrows()):
                col = cols[idx % 3]
                with col:
                    display_bike_card(bike, "grid")
        else:
            for idx, (_, bike) in enumerate(page_bikes.iterrows()):
                display_bike_card(bike, "list")

def show_detail_page():
    """Trang chi tiết xe"""
    
    if st.session_state.get('selected_bike_idx') is None:
        st.error("❌ Không tìm thấy thông tin xe. Vui lòng chọn xe từ trang tìm kiếm.")
        if st.button("← Quay lại tìm kiếm", key="back_error1", type="primary"):
            st.session_state.page = "search"
            st.rerun()
        return
    
    bike_idx = st.session_state.selected_bike_idx
    
    # ✅ Enhanced validation
    if not (0 <= bike_idx < len(df)):
        st.error(f"❌ Index không hợp lệ: {bike_idx} (Max: {len(df)-1})")
        if st.button("← Quay lại tìm kiếm", key="back_error2", type="primary"):
            st.session_state.page = "search"
            st.rerun()
        return
    
    try:
        bike = df.iloc[bike_idx]
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        if st.button("← Quay lại tìm kiếm", key="back_error3", type="primary"):
            st.session_state.page = "search"
            st.rerun()
        return
    
    st.components.v1.html("""
        <script>
            window.parent.document.querySelector('.main').scrollTo({top: 0, behavior: 'smooth'});
        </script>
    """, height=0)
    
    if st.button("← Quay lại tìm kiếm"):
        st.session_state.page = "search"
        st.rerun()
    
    st.markdown("---")
    
    st.title(f"{bike['brand']} {bike['model']}")
    
    cluster_id = bike['cluster']
    cluster_name = cluster_labels.get(cluster_id, 'Chưa phân loại')
    cluster_color = cluster_colors.get(cluster_id, '#667eea')
    
    st.markdown(f"""
<div style="
    background-color:{cluster_color};
    display:inline-block;
    color:white;
    padding:8px 15px;
    border-radius:6px;
    font-weight:bold;
    margin-top:5px;
    margin-bottom:15px;">
    🚀 {cluster_name}
</div>
""", unsafe_allow_html=True)
    
    st.markdown("### 💳 Thông Tin Chính")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Giá bán", format_price(bike['price']))
    col2.metric("📏 Số km đã đi", format_km(bike['km_driven']))
    col3.metric("📅 Tuổi xe", f"{int(bike['age'])} năm")
    
    vtype_col = "vehicle_type_display" if "vehicle_type_display" in bike.index else "vehicle_type"
    if vtype_col in bike.index and pd.notna(bike[vtype_col]):
        col4.metric("🏷️ Loại xe", bike[vtype_col])
    
    st.markdown("""<div style="margin: 24px 0;"></div>""", unsafe_allow_html=True)
    
    st.markdown("### 📋 Thông Tin Chi Tiết")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        info_parts = [f"- **🏢 Thương hiệu:** {bike['brand']}", f"- **🏍️ Model:** {bike['model']}"]
        
        engine_col = "engine_capacity" if "engine_capacity" in bike.index else "engine_capacity_num"
        if engine_col in bike.index and pd.notna(bike[engine_col]):
            info_parts.append(f"- **⚙️ Dung tích động cơ:** {bike[engine_col]}")
        st.markdown('\n'.join(info_parts))
    
    with col_y:
        info_parts2 = []
        
        origin_col = "origin" if "origin" in bike.index else "origin_num"
        if origin_col in bike.index and pd.notna(bike[origin_col]):
            info_parts2.append(f"- **🌍 Xuất xứ:** {bike[origin_col]}")
        
        info_parts2.append(f"- **📍 Địa điểm:** {bike['location']}")
        st.markdown('\n'.join(info_parts2))
    
    st.markdown("""<div style="margin: 24px 0;"></div>""", unsafe_allow_html=True)
    
    st.markdown("### 📝 Mô Tả Chi Tiết")
    
    desc_text = ""
    if 'description_norm' in bike.index and pd.notna(bike['description_norm']) and str(bike['description_norm']).strip():
        desc_text = str(bike['description_norm'])
    elif 'description' in bike.index and pd.notna(bike['description']) and str(bike['description']).strip():
        desc_text = str(bike['description'])
    
    if desc_text:
        st.write(desc_text)
    else:
        st.info("ℹ️ Chưa có mô tả chi tiết cho xe này.")
    
    st.markdown("""<div style="margin: 24px 0;"></div>""", unsafe_allow_html=True)
    
    st.markdown("## 🎯 Xe Tương Tự Bạn Có Thể Quan Tâm")
    
    similar_bikes = get_similar_bikes(bike_idx, df, top_k=5)
    
    if len(similar_bikes) > 0:
        for idx, sim_bike in similar_bikes.iterrows():
            sim_cluster_id = sim_bike.get('cluster', 0)
            sim_cluster_name = cluster_labels.get(sim_cluster_id, 'Chưa phân loại')
            sim_cluster_color = cluster_colors.get(sim_cluster_id, '#667eea')
            similarity_score = sim_bike.get('similarity_score', sim_bike.get('similarity', 0))
            
            similar_idx = int(sim_bike.get('position', idx))
            
            engine_info = ""
            if 'engine_capacity' in sim_bike.index and pd.notna(sim_bike['engine_capacity']):
                engine_info = f"⚙️ {sim_bike['engine_capacity']}"
            elif 'engine_capacity_num' in sim_bike.index and pd.notna(sim_bike['engine_capacity_num']):
                engine_info = f"⚙️ {sim_bike['engine_capacity_num']}"
            
            vehicle_type_info = ""
            if 'vehicle_type_display' in sim_bike.index and pd.notna(sim_bike['vehicle_type_display']):
                vehicle_type_info = f"🏍️ {sim_bike['vehicle_type_display']}"
            elif 'vehicle_type' in sim_bike.index and pd.notna(sim_bike['vehicle_type']):
                vehicle_type_info = f"🏍️ {sim_bike['vehicle_type']}"
            
            description_text = ""
            if 'description_norm' in sim_bike.index and pd.notna(sim_bike['description_norm']) and str(sim_bike['description_norm']).strip():
                desc = str(sim_bike['description_norm'])
                description_text = desc[:150] + "..." if len(desc) > 150 else desc
            elif 'description' in sim_bike.index and pd.notna(sim_bike['description']) and str(sim_bike['description']).strip():
                desc = str(sim_bike['description'])
                description_text = desc[:150] + "..." if len(desc) > 150 else desc
            
            st.markdown(f"""
<div style="
    background: white;
    border-left: 5px solid {sim_cluster_color};
    padding: 20px;
    margin: 15px 0;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12);
">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px;">
            <div style="margin-bottom: 10px;">
                <strong style="font-size: 18px; color:#2c3e50;">{sim_bike['brand']} {sim_bike['model']}</strong>
            </div>
            <div style="margin-bottom: 8px;">
                <span style="
                    background: linear-gradient(135deg, {sim_cluster_color} 0%, {sim_cluster_color}dd 100%);
                    color:white;
                    padding:5px 12px;
                    border-radius:6px;
                    font-size:11px;
                    font-weight:600;
                    display: inline-block;
                    margin-right: 8px;
                ">
                    {sim_cluster_name}
                </span>
                {f'<span style="color: #6e6e73; font-size: 13px; margin-right: 12px;">{engine_info}</span>' if engine_info else ''}
                {f'<span style="color: #6e6e73; font-size: 13px;">{vehicle_type_info}</span>' if vehicle_type_info else ''}
            </div>
            {f'<p style="color: #86868b; font-size: 13px; line-height: 1.5; margin: 8px 0 0 0; font-style: italic;">"{description_text}"</p>' if description_text else ''}
        </div>
        <div style="text-align: right; margin-top: 10px;">
            <div style="
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color:white;
                padding:10px 18px;
                border-radius:10px;
                font-weight:700;
                font-size:16px;
            ">
                {similarity_score*100:.1f}% tương tự
            </div>
        </div>
    </div>
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; font-size:15px; color:#555;">
        <span style="margin-right: 20px;">
            💰 <strong style="color:#667eea;">{format_price(sim_bike['price'])}</strong>
        </span>
        <span style="margin-right: 20px;">
            📏 {format_km(sim_bike['km_driven'])}
        </span>
        <span>
            📅 {int(sim_bike['age'])} năm
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
            
            btn_key = f"similar_detail_{similar_idx}_{idx}"
            if st.button("🔍 Xem chi tiết xe này", key=btn_key, use_container_width=True, type="primary"):
                st.session_state.selected_bike_idx = int(similar_idx)
                st.session_state.page = "detail"
                st.rerun()
    else:
        st.info("ℹ️ Không tìm thấy xe tương tự.")

@st.cache_data(show_spinner=False)
def compute_analysis_metrics(df_input):
    """Cache các metrics cơ bản để tránh tính lại"""
    return {
        'total_bikes': len(df_input),
        'avg_price': df_input['price'].mean(),
        'median_price': df_input['price'].median(),
        'total_value': df_input['price'].sum(),
        'avg_km': df_input['km_driven'].mean(),
        'avg_age': df_input['age'].mean()
    }

@st.cache_data(show_spinner=False)
def generate_market_insights(df_input):
    """Tạo insights thông minh từ dữ liệu"""
    insights = []
    
    price_std = df_input['price'].std()
    if price_std > df_input['price'].mean() * 0.5:
        insights.append({
            'icon': '📈',
            'type': 'warning',
            'title': 'Biến động giá cao',
            'message': f'Thị trường có độ biến động giá lớn (±{price_std:.1f}M).'
        })
    
    cluster_avg_prices = df_input.groupby('cluster')['price'].mean()
    best_value_cluster = cluster_avg_prices.idxmin()
    insights.append({
        'icon': '💰',
        'type': 'success',
        'title': 'Phân khúc giá tốt',
        'message': f'{cluster_labels.get(best_value_cluster, f"Nhóm {best_value_cluster}")} có giá trung bình thấp nhất ({cluster_avg_prices[best_value_cluster]:.1f}M)'
    })
    
    high_km_pct = (df_input['km_driven'] > 50000).sum() / len(df_input) * 100
    if high_km_pct > 30:
        insights.append({
            'icon': '🛑',
            'type': 'info',
            'title': 'Xe đi nhiều km',
            'message': f'{high_km_pct:.1f}% xe đã đi trên 50,000km. Kiểm tra kỹ bảo dưỡng!'
        })
    
    top_brand = df_input['brand'].value_counts().iloc[0]
    top_brand_name = df_input['brand'].value_counts().index[0]
    top_brand_pct = top_brand / len(df_input) * 100
    if top_brand_pct > 15:
        insights.append({
            'icon': '🏆',
            'type': 'info',
            'title': 'Thương hiệu phổ biến',
            'message': f'{top_brand_name} chiếm {top_brand_pct:.1f}% thị trường ({top_brand:,} xe)'
        })
    
    age_price_corr = df_input[['age', 'price']].corr().iloc[0, 1]
    if abs(age_price_corr) > 0.6:
        insights.append({
            'icon': '🔗',
            'type': 'info',
            'title': 'Tương quan tuổi - giá',
            'message': f'Tuổi xe có tương quan {"nghịch" if age_price_corr < 0 else "thuận"} mạnh với giá ({age_price_corr:.2f})'
        })
    
    cheap_bikes = df_input[df_input['price'] < df_input['price'].quantile(0.25)]
    if len(cheap_bikes) > 0:
        avg_km_cheap = cheap_bikes['km_driven'].mean()
        insights.append({
            'icon': '✨',
            'type': 'success',
            'title': 'Cơ hội giá tốt',
            'message': f'Có {len(cheap_bikes):,} xe giá rẻ (dưới {df_input["price"].quantile(0.25):.1f}M) với TB {avg_km_cheap:,.0f}km'
        })
    
    return insights

def show_analysis_page(show_header=True):
    """Trang phân tích chuyên sâu cho quản lý - Optimized"""
    if show_header:
        st.header("📊 Phân Tích Thị Trường Chuyên Sâu")
    
    metrics = compute_analysis_metrics(df)
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    ">
        <h3 style="
            color: white;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
        ">
            🤖 AI Insights - Gợi Ý Thông Minh
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    insights = generate_market_insights(df)
    
    cols_per_row = 3
    for i in range(0, len(insights), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(insights):
                insight = insights[i + j]
                color_map = {
                    'success': '#2ecc71',
                    'warning': '#f39c12',
                    'info': '#3498db',
                    'danger': '#e74c3c'
                }
                border_color = color_map.get(insight['type'], '#667eea')
                
                with col:
                    st.markdown(f"""
                    <div style="
                        background: white;
                        padding: 24px;
                        border-radius: 16px;
                        border-left: 4px solid {border_color};
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        margin-bottom: 20px;
                        min-height: 165px;
                    ">
                        <div style="font-size: 36px; margin-bottom: 12px;">{insight['icon']}</div>
                        <h4 style="
                            font-size: 16px;
                            font-weight: 700;
                            color: {border_color};
                            margin-bottom: 10px;
                        ">{insight['title']}</h4>
                        <p style="
                            font-size: 14px;
                            color: #6e6e73;
                            line-height: 1.6;
                            margin: 0;
                        ">{insight['message']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("""<div style="margin: 40px 0;"></div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="
        font-size: 26px;
        font-weight: 700;
        color: #1d1d1f;
        margin-bottom: 24px;
    ">🎯 Chỉ Số Kinh Doanh Chính</h3>
    """, unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.metric("🏍️ Tổng số xe", f"{metrics['total_bikes']:,}")
    
    with kpi2:
        st.metric("💰 Giá TB", f"{metrics['avg_price']:.1f}M", delta=f"Median: {metrics['median_price']:.1f}M")
    
    with kpi3:
        st.metric("💵 Tổng giá trị", f"{metrics['total_value']:,.0f}M")
    
    with kpi4:
        st.metric("🛣️ Km TB", f"{metrics['avg_km']:,.0f}")
    
    with kpi5:
        st.metric("📅 Tuổi TB", f"{metrics['avg_age']:.1f} năm")
    
    st.markdown("""<div style="margin: 32px 0;"></div>""", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Tổng Quan", "💰 Phân Tích Giá", "🏢 Thương Hiệu", 
        "📍 Khu Vực", "🚀 Phân Khúc", "📊 Ma Trận"
    ])
    
    with tab1:
        st.markdown("""
        <h3 style="
            font-size: 24px;
            font-weight: 700;
            color: #1d1d1f;
            margin: 24px 0;
        ">📈 Tổng Quan Thị Trường</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner('Đang vẽ biểu đồ giá...'):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df['price'], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
                ax.axvline(metrics['avg_price'], color='red', linestyle='--', linewidth=2, label=f"Trung bình: {metrics['avg_price']:.1f}M")
                ax.axvline(metrics['median_price'], color='green', linestyle='--', linewidth=2, label=f"Trung vị: {metrics['median_price']:.1f}M")
                ax.set_xlabel('Giá (triệu VNĐ)')
                ax.set_ylabel('Số lượng xe')
                ax.set_title('Phân Bố Giá Xe', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close('all')
                gc.collect()
        
        with col2:
            with st.spinner('Đang vẽ biểu đồ tuổi xe...'):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df['age'], bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
                ax.axvline(metrics['avg_age'], color='red', linestyle='--', linewidth=2, label=f"Trung bình: {metrics['avg_age']:.1f} năm")
                ax.set_xlabel('Tuổi xe (năm)')
                ax.set_ylabel('Số lượng xe')
                ax.set_title('Phân Bố Tuổi Xe', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close('all')
                gc.collect()
        
        st.markdown("""<div style="margin: 32px 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">🔥 Ma Trận Tương Quan</h4>
        """, unsafe_allow_html=True)
        
        col_heat, col_insight = st.columns([2, 1])
        
        with col_heat:
            numeric_cols = ['price', 'km_driven', 'age']
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(7, 5.5))
            im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
            
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(['Giá', 'Km đã đi', 'Tuổi xe'])
            ax.set_yticklabels(['Giá', 'Km đã đi', 'Tuổi xe'])
            
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=12, fontweight='bold')
            
            plt.colorbar(im, ax=ax)
            ax.set_title('Ma Trận Tương Quan Giữa Các Biến', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        with col_insight:
            st.markdown("""
            <div style="
                background: #f5f5f7;
                padding: 20px;
                border-radius: 12px;
                height: 100%;
            ">
                <h4 style="
                    font-size: 16px;
                    font-weight: 700;
                    color: #1d1d1f;
                    margin-bottom: 16px;
                ">📊 Giải Thích Tương Quan</h4>
                <div style="font-size: 13px; color: #6e6e73; line-height: 1.8;">
            """, unsafe_allow_html=True)
            
            price_km_corr = corr_matrix.loc['price', 'km_driven']
            price_age_corr = corr_matrix.loc['price', 'age']
            km_age_corr = corr_matrix.loc['km_driven', 'age']
            
            if price_km_corr < -0.3:
                st.markdown("🔻 **Giá ↓ khi Km ↑**: Xe đi nhiều km có giá thấp hơn")
            elif price_km_corr > 0.3:
                st.markdown("🔺 **Giá ↑ khi Km ↑**: Xe đi nhiều km lại có giá cao (bất thường!)")
            else:
                st.markdown("➡️ **Giá ≈ Km**: Tương quan yếu giữa giá và km")
            
            st.markdown("---")
            
            if price_age_corr < -0.3:
                st.markdown("🔻 **Giá ↓ khi Tuổi ↑**: Xe càng cũ càng rẻ (bình thường)")
            elif price_age_corr > 0.3:
                st.markdown("🔺 **Giá ↑ khi Tuổi ↑**: Xe cũ lại đắt (có thể xe cổ/hiếm)")
            else:
                st.markdown("➡️ **Giá ≈ Tuổi**: Tuổi xe không ảnh hưởng nhiều đến giá")
            
            st.markdown("---")
            
            if km_age_corr > 0.5:
                st.markdown("🔗 **Km ↑ khi Tuổi ↑**: Xe càng cũ đi càng nhiều km (logic)")
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""<div style="margin: 50px 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">🎯 Thống Kê Nhanh</h4>
        """, unsafe_allow_html=True)
        stat1, stat2, stat3, stat4 = st.columns(4)
        
        with stat1:
            cheap_count = len(df[df['price'] < 20])
            st.metric("💰 Xe giá rẻ (<20M)", f"{cheap_count:,}", 
                     delta=f"{cheap_count/len(df)*100:.1f}%")
        
        with stat2:
            low_km_count = len(df[df['km_driven'] < 10000])
            st.metric("🆕 Xe ít km (<10K)", f"{low_km_count:,}",
                     delta=f"{low_km_count/len(df)*100:.1f}%")
        
        with stat3:
            new_bikes = len(df[df['age'] <= 2])
            st.metric("✨ Xe mới (≤2 năm)", f"{new_bikes:,}",
                     delta=f"{new_bikes/len(df)*100:.1f}%")
        
        with stat4:
            premium_bikes = len(df[df['price'] > 100])
            st.metric("🏆 Xe cao cấp (>100M)", f"{premium_bikes:,}",
                     delta=f"{premium_bikes/len(df)*100:.1f}%")
    
    with tab2:
        st.markdown("""
        <h3 style="
            font-size: 24px;
            font-weight: 700;
            color: #1d1d1f;
            margin: 24px 0;
        ">💰 Phân Tích Giá Chi Tiết</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <h4 style="
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">📦 Phân Bố Giá Theo Phân Khúc</h4>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cluster_prices = [df[df['cluster'] == i]['price'].values for i in sorted(cluster_labels.keys())]
            positions = range(len(cluster_labels))
            
            bp = ax.boxplot(cluster_prices, positions=positions, patch_artist=True, widths=0.6)
            
            for patch, cluster_id in zip(bp['boxes'], sorted(cluster_labels.keys())):
                patch.set_facecolor(cluster_colors.get(cluster_id, '#667eea'))
                patch.set_alpha(0.7)
            
            for whisker in bp['whiskers']:
                whisker.set(linewidth=1.5)
            for cap in bp['caps']:
                cap.set(linewidth=1.5)
            for median in bp['medians']:
                median.set(color='red', linewidth=2)
            
            ax.set_xticks(positions)
            ax.set_xticklabels([cluster_labels.get(i, f'Nhóm {i}')[:20] for i in sorted(cluster_labels.keys())], 
                              rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Giá (triệu VNĐ)', fontsize=11)
            ax.set_title('Phân Bố Giá Theo Phân Khúc', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        with col2:
            st.markdown("""
            <h4 style="
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">📊 Thống Kê Giá Theo Phân Khúc</h4>
            """, unsafe_allow_html=True)
            price_stats = []
            for cluster_id in sorted(cluster_labels.keys()):
                cluster_data = df[df['cluster'] == cluster_id]
                price_stats.append({
                    'Phân khúc': cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:30],
                    'Số xe': len(cluster_data),
                    'Giá TB': f"{cluster_data['price'].mean():.1f}M",
                    'Giá Min': f"{cluster_data['price'].min():.1f}M",
                    'Giá Max': f"{cluster_data['price'].max():.1f}M",
                    'Trung vị': f"{cluster_data['price'].median():.1f}M"
                })
            
            stats_df = pd.DataFrame(price_stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("""<div style="margin: 32px 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">📉 Giá Theo Km Đã Đi</h4>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for cluster_id in sorted(cluster_labels.keys()):
            cluster_data = df[df['cluster'] == cluster_id]
            ax.scatter(cluster_data['km_driven'], cluster_data['price'],
                      color=cluster_colors.get(cluster_id, '#667eea'),
                      label=cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:25],
                      alpha=0.5, s=30)
        
        z = np.polyfit(df['km_driven'], df['price'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df['km_driven'].min(), df['km_driven'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Xu hướng', alpha=0.8)
        
        ax.set_xlabel('Km đã đi', fontsize=11)
        ax.set_ylabel('Giá (triệu VNĐ)', fontsize=11)
        ax.set_title('Mối Quan Hệ Giữa Giá và Km Đã Đi', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close('all')
        gc.collect()
    
    with tab3:
        st.subheader("🏢 Phân Tích Thương Hiệu")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🥧 Top 10 Thương Hiệu")
            top_brands = df['brand'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.Set3(range(len(top_brands)))
            wedges, texts, autotexts = ax.pie(top_brands.values, labels=top_brands.index,
                                               autopct='%1.1f%%', colors=colors, startangle=90,
                                               textprops={'fontsize': 10})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Thị Phần Top 10 Thương Hiệu', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        with col2:
            st.markdown("""
            <h4 style="
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">📊 Thống Kê Thương Hiệu</h4>
            """, unsafe_allow_html=True)
            brand_stats = []
            for brand in top_brands.head(10).index:
                brand_data = df[df['brand'] == brand]
                brand_stats.append({
                    'Thương hiệu': brand,
                    'Số xe': len(brand_data),
                    'Giá TB': f"{brand_data['price'].mean():.1f}M",
                    'Km TB': f"{brand_data['km_driven'].mean():,.0f}",
                    'Tuổi TB': f"{brand_data['age'].mean():.1f}"
                })
            
            brand_df = pd.DataFrame(brand_stats)
            st.dataframe(brand_df, use_container_width=True, hide_index=True)
        
        st.markdown("""<div style="margin: 32px 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">💰 Giá Trung Bình Theo Thương Hiệu</h4>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        avg_prices = df.groupby('brand')['price'].mean().sort_values(ascending=False).head(15)
        bars = ax.barh(range(len(avg_prices)), avg_prices.values, color='#667eea', alpha=0.7)
        ax.set_yticks(range(len(avg_prices)))
        ax.set_yticklabels(avg_prices.index, fontsize=10)
        ax.set_xlabel('Giá trung bình (triệu VNĐ)', fontsize=11)
        ax.set_title('Top 15 Thương Hiệu Theo Giá Trung Bình', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, avg_prices.values)):
            ax.text(val, i, f' {val:.1f}M', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close('all')
        gc.collect()
    
    with tab4:
        st.markdown("""
        <h3 style="
            font-size: 24px;
            font-weight: 700;
            color: #1d1d1f;
            margin: 24px 0;
        ">📍 Phân Tích Theo Khu Vực</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <h4 style="
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">🗺️ Phân Bố Xe Theo Khu Vực</h4>
            """, unsafe_allow_html=True)
            location_counts = df['location'].value_counts().head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(location_counts)), location_counts.values, 
                          color='#2ecc71', alpha=0.7)
            ax.set_yticks(range(len(location_counts)))
            ax.set_yticklabels(location_counts.index, fontsize=10)
            ax.set_xlabel('Số lượng xe', fontsize=11)
            ax.set_title('Top 15 Khu Vực Có Nhiều Xe Nhất', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, (bar, val) in enumerate(zip(bars, location_counts.values)):
                ax.text(val, i, f' {val:,}', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        with col2:
            st.markdown("""
            <h4 style="
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">💰 Giá Trung Bình Theo Khu Vực</h4>
            """, unsafe_allow_html=True)
            location_prices = df.groupby('location')['price'].mean().sort_values(ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(location_prices)), location_prices.values,
                          color='#e74c3c', alpha=0.7)
            ax.set_yticks(range(len(location_prices)))
            ax.set_yticklabels(location_prices.index, fontsize=10)
            ax.set_xlabel('Giá trung bình (triệu VNĐ)', fontsize=11)
            ax.set_title('Top 15 Khu Vực Giá Cao Nhất', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, (bar, val) in enumerate(zip(bars, location_prices.values)):
                ax.text(val, i, f' {val:.1f}M', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        st.markdown("""<div style="margin: 32px 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">📊 Bảng Thống Kê Khu Vực</h4>
        """, unsafe_allow_html=True)
        location_stats = df.groupby('location').agg({
            'price': ['count', 'mean', 'median'],
            'km_driven': 'mean',
            'age': 'mean'
        }).round(1)
        
        location_stats.columns = ['Số xe', 'Giá TB (M)', 'Giá median (M)', 'Km TB', 'Tuổi TB']
        location_stats = location_stats.sort_values('Số xe', ascending=False).head(20)
        st.dataframe(location_stats, use_container_width=True)
    
    with tab5:
        st.markdown("""
        <h3 style="
            font-size: 24px;
            font-weight: 700;
            color: #1d1d1f;
            margin: 24px 0;
        ">🚀 Phân Tích Phân Khúc</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <h4 style="
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">📊 Số Lượng Xe Theo Phân Khúc</h4>
            """, unsafe_allow_html=True)
            cluster_dist = df['cluster'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_list = [cluster_colors.get(i, '#667eea') for i in cluster_dist.index]
            
            bars = ax.bar(range(len(cluster_dist)), cluster_dist.values, color=colors_list, alpha=0.8, edgecolor='black')
            ax.set_xticks(range(len(cluster_dist)))
            ax.set_xticklabels([f'Nhóm {i}' for i in cluster_dist.index], fontsize=10)
            ax.set_ylabel('Số lượng xe', fontsize=11)
            ax.set_title('Phân Bố Xe Theo Phân Khúc', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        with col2:
            st.markdown("""
            <h4 style="
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">🎯 Đặc Điểm Phân Khúc</h4>
            """, unsafe_allow_html=True)
            cluster_char = []
            for cluster_id in sorted(cluster_labels.keys()):
                cluster_data = df[df['cluster'] == cluster_id]
                cluster_char.append({
                    'Cụm': f'{cluster_id}',
                    'Tên': cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:30],
                    'Số xe': f"{len(cluster_data):,}",
                    'Giá TB': f"{cluster_data['price'].mean():.1f}M",
                    'Km TB': f"{cluster_data['km_driven'].mean():,.0f}",
                    'Tuổi TB': f"{cluster_data['age'].mean():.1f}"
                })
            
            cluster_df = pd.DataFrame(cluster_char)
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        
        st.markdown("""<div style="margin: 32px 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">🌐 Mối Quan Hệ 3D: Tuổi - Km - Giá</h4>
        """, unsafe_allow_html=True)
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in sorted(cluster_labels.keys()):
            cluster_data = df[df['cluster'] == cluster_id].sample(min(300, len(df[df['cluster'] == cluster_id])))
            ax.scatter(cluster_data['age'], cluster_data['km_driven'], cluster_data['price'],
                      c=cluster_colors.get(cluster_id, '#667eea'),
                      label=cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:20],
                      alpha=0.6, s=20)
        
        ax.set_xlabel('Tuổi xe (năm)', fontsize=10)
        ax.set_ylabel('Km đã đi', fontsize=10)
        ax.set_zlabel('Giá (triệu)', fontsize=10)
        ax.set_title('Phân Bố 3D Theo Tuổi - Km - Giá', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close('all')
        gc.collect()
    
    with tab6:
        st.markdown("""
        <h3 style="
            font-size: 24px;
            font-weight: 700;
            color: #1d1d1f;
            margin: 24px 0;
        ">📊 Ma Trận Phân Tích</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">💵 Phân Bố Theo Khoảng Giá</h4>
        """, unsafe_allow_html=True)
        price_ranges = pd.cut(df['price'], bins=[0, 10, 20, 30, 50, 100, 500], 
                             labels=['<10M', '10-20M', '20-30M', '30-50M', '50-100M', '>100M'])
        price_range_dist = price_ranges.value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(price_range_dist)), price_range_dist.values, 
                         color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(price_range_dist)))
            ax.set_xticklabels(price_range_dist.index, fontsize=10)
            ax.set_ylabel('Số lượng xe', fontsize=11)
            ax.set_title('Phân Bố Theo Khoảng Giá', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        with col2:
            st.markdown("""
            <h5 style="
                font-size: 16px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 16px 0;
            ">🛣️ Phân Bố Theo Khoảng Km</h4>
            """, unsafe_allow_html=True)
            km_ranges = pd.cut(df['km_driven'], bins=[0, 5000, 10000, 20000, 50000, 100000, 1000000],
                              labels=['<5K', '5-10K', '10-20K', '20-50K', '50-100K', '>100K'])
            km_range_dist = km_ranges.value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 5.6))
            bars = ax.bar(range(len(km_range_dist)), km_range_dist.values,
                         color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(km_range_dist)))
            ax.set_xticklabels(km_range_dist.index, fontsize=10)
            ax.set_ylabel('Số lượng xe', fontsize=11)
            ax.set_title('Phân Bố Theo Khoảng Km', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close('all')
            gc.collect()
        
        st.markdown("""<div style="margin: 32px 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin: 16px 0;
        ">🔀 Ma Trận: Thương Hiệu × Phân Khúc (Top 10)</h4>
        """, unsafe_allow_html=True)
        top_brands_list = df['brand'].value_counts().head(10).index
        cross_tab = pd.crosstab(df[df['brand'].isin(top_brands_list)]['brand'], 
                                df[df['brand'].isin(top_brands_list)]['cluster'])
        
        cross_tab.columns = [f'Nhóm {i}' for i in cross_tab.columns]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(cross_tab.columns)))
        ax.set_yticks(range(len(cross_tab.index)))
        ax.set_xticklabels(cross_tab.columns, fontsize=10)
        ax.set_yticklabels(cross_tab.index, fontsize=10)
        
        for i in range(len(cross_tab.index)):
            for j in range(len(cross_tab.columns)):
                text = ax.text(j, i, f'{cross_tab.iloc[i, j]}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        ax.set_title('Phân Bố Thương Hiệu Theo Phân Khúc', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close('all')
        gc.collect()

def show_help_page():
    """Trang hướng dẫn"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 16px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        ">📘 Hướng Dẫn Sử Dụng</h1>
        <p style="
            font-size: 20px;
            opacity: 0.95;
            margin: 0;
        ">Khám phá mọi tính năng để tìm xe máy cũ hoàn hảo! 🏍️✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_quick, tab_search, tab_detail, tab_cluster, tab_tips = st.tabs([
        "🚀 Bắt Đầu Nhanh",
        "🔍 Tìm Kiếm & Lọc",
        "👁️ Chi Tiết & Gợi Ý",
        "🧠 Phân Nhóm",
        "💡 Mẹo Nhanh"
    ])
    
    with tab_quick:
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <h2 style="
                font-size: 36px;
                font-weight: 700;
                color: #1d1d1f;
                margin-bottom: 20px;
            ">🎯 Chỉ 3 Bước Đơn Giản!</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px;
                border-radius: 16px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
                min-height: 280px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div style="
                    font-size: 64px;
                    margin-bottom: 16px;
                ">🔍</div>
                <div>
                    <h3 style="
                        font-size: 24px;
                        font-weight: 700;
                        margin-bottom: 12px;
                        color: white;
                    ">Bước 1</h3>
                    <p style="
                        font-size: 16px;
                        line-height: 1.6;
                        opacity: 0.95;
                    ">Vào tab <strong>Tìm Kiếm</strong> và nhập từ khóa</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 30px;
                border-radius: 16px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 24px rgba(245, 87, 108, 0.3);
                min-height: 280px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div style="
                    font-size: 64px;
                    margin-bottom: 16px;
                ">🔧</div>
                <div>
                    <h3 style="
                        font-size: 24px;
                        font-weight: 700;
                        margin-bottom: 12px;
                        color: white;
                    ">Bước 2</h3>
                    <p style="
                        font-size: 16px;
                        line-height: 1.6;
                        opacity: 0.95;
                    ">Thu hẹp kết quả bằng <strong>Bộ Lọc</strong>: hãng, giá, khu vực..</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 30px;
                border-radius: 16px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 24px rgba(79, 172, 254, 0.3);
                min-height: 280px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div style="
                    font-size: 64px;
                    margin-bottom: 16px;
                ">👁️</div>
                <div>
                    <h3 style="
                        font-size: 24px;
                        font-weight: 700;
                        margin-bottom: 12px;
                        color: white;
                    ">Bước 3</h3>
                    <p style="
                        font-size: 16px;
                        line-height: 1.6;
                        opacity: 0.95;
                    ">Xem <strong>Chi Tiết</strong> và nhận gợi ý xe tương tự!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 50px 0;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Bắt Đầu Tìm Kiếm Ngay!", use_container_width=True, type="primary"):
                st.session_state.page = "search"
                st.rerun()
    
    with tab_search:
        st.markdown("""
        <div style="
            background: #f5f5f7;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
        ">
            <h2 style="
                font-size: 32px;
                font-weight: 700;
                color: #1d1d1f;
                margin-bottom: 24px;
                text-align: center;
            ">🔍 Tìm Kiếm Thông Minh</h2>
        </div>
        """, unsafe_allow_html=True)
        
        features = [
            {
                "icon": "🔤",
                "title": "Tìm Kiếm Văn Bản",
                "desc": "Nhập từ khóa như tên hãng, model, hoặc loại xe. Hệ thống tự động tìm kết quả phù hợp nhất!",
                "color": "#667eea"
            },
            {
                "icon": "🔧",
                "title": "Bộ Lọc Đa Dạng",
                "desc": "Chọn hãng, model, loại xe, khu vực, dung tích động cơ để lọc chính xác.",
                "color": "#f5576c"
            },
            {
                "icon": "💰",
                "title": "Khoảng Giá Linh Hoạt",
                "desc": "Điều chỉnh thanh trượt hoặc nhập giá từ/đến (triệu VNĐ) để tìm xe trong tầm giá.",
                "color": "#2ecc71"
            },
            {
                "icon": "📥",
                "title": "Xuất Dữ Liệu",
                "desc": "Tải kết quả tìm kiếm dạng CSV để phân tích thêm trên Excel!",
                "color": "#3498db"
            },
            {
                "icon": "📊",
                "title": "Thống Kê Nhanh",
                "desc": "Xem giá TB, min/max, số lượng xe trong kết quả tìm kiếm.",
                "color": "#9b59b6"
            },
            {
                "icon": "🔄",
                "title": "Cập Nhật Tức Thì",
                "desc": "Kết quả tự động cập nhật khi thay đổi bộ lọc hoặc từ khóa.",
                "color": "#e67e22"
            }
        ]
        
        col1, col2, col3 = st.columns(3)
        for idx, feature in enumerate(features):
            col = [col1, col2, col3][idx % 3]
            with col:
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 24px;
                    border-radius: 16px;
                    margin-bottom: 20px;
                    border-left: 4px solid {feature['color']};
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 40px; margin-bottom: 12px;">{feature['icon']}</div>
                    <h4 style="
                        font-size: 18px;
                        font-weight: 700;
                        color: {feature['color']};
                        margin-bottom: 8px;
                    ">{feature['title']}</h4>
                    <p style="
                        font-size: 14px;
                        color: #6e6e73;
                        line-height: 1.6;
                        margin: 0;
                    ">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab_detail:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 40px;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
        ">
            <h2 style="
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 16px;
                color: white;
            ">👁️ Chi Tiết & Xe Tương Tự</h2>
            <p style="
                font-size: 18px;
                opacity: 0.95;
            ">Khám phá thông tin đầy đủ và nhận gợi ý xe phù hợp!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="
                background: #f5f5f7;
                padding: 30px;
                border-radius: 16px;
                height: 100%;
            ">
                <h3 style="
                    font-size: 24px;
                    font-weight: 700;
                    color: #1d1d1f;
                    margin-bottom: 20px;
                ">📋 Thông Tin Chi Tiết</h3>
                <ul style="
                    font-size: 16px;
                    color: #6e6e73;
                    line-height: 2;
                    list-style: none;
                    padding: 0;
                ">
                    <li>✅ Tên xe, hãng, model đầy đủ</li>
                    <li>💰 Giá bán, km đã đi, tuổi xe</li>
                    <li>🏍️ Dung tích, loại xe, khu vực</li>
                    <li>🎨 <strong>Badge màu</strong> phân khúc (cluster)</li>
                    <li>📊 So sánh với giá TB cùng cụm</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                background: #f5f5f7;
                padding: 30px;
                border-radius: 16px;
                height: 100%;
            ">
                <h3 style="
                    font-size: 24px;
                    font-weight: 700;
                    color: #1d1d1f;
                    margin-bottom: 20px;
                ">🎯 Gợi Ý Thông Minh</h3>
                <ul style="
                    font-size: 16px;
                    color: #6e6e73;
                    line-height: 2;
                    list-style: none;
                    padding: 0;
                ">
                    <li>🤖 Sử dụng AI để tìm xe tương tự</li>
                    <li>📊 Hiển thị <strong>% độ tương đồng</strong></li>
                    <li>🔍 Xem nhanh giá, km, tuổi xe gợi ý</li>
                    <li>👆 Click vào card để xem chi tiết</li>
                    <li>🔄 Liên tục khám phá xe mới!</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab_cluster:
        st.markdown("""
        <div style="
            text-align: center;
            margin-bottom: 40px;
        ">
            <h2 style="
                font-size: 36px;
                font-weight: 700;
                color: #1d1d1f;
                margin-bottom: 16px;
            ">🧠 Hiểu Về Phân Nhóm Thông Minh</h2>
            <p style="
                font-size: 18px;
                color: #6e6e73;
                max-width: 800px;
                margin: 0 auto;
            ">Hệ thống sử dụng thuật toán K-Means để phân loại xe máy thành 5 nhóm dựa trên giá, km, tuổi xe và đặc điểm khác</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: #f5f5f7;
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 30px;
        ">
            <h3 style="
                font-size: 24px;
                font-weight: 700;
                color: #1d1d1f;
                margin-bottom: 20px;
                text-align: center;
            ">💡 Lợi Ích Của Phân Nhóm</h3>
            <div style="
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-top: 24px;
            ">
                <div style="text-align: center;">
                    <div style="font-size: 48px; margin-bottom: 12px;">🎯</div>
                    <p style="font-size: 16px; color: #1d1d1f; font-weight: 600;">Tìm xe cùng phân khúc</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 48px; margin-bottom: 12px;">💰</div>
                    <p style="font-size: 16px; color: #1d1d1f; font-weight: 600;">So sánh giá trong nhóm</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 48px; margin-bottom: 12px;">📊</div>
                    <p style="font-size: 16px; color: #1d1d1f; font-weight: 600;">Phân tích thị trường</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎨 Các Phân Khúc Xe Máy")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                cluster_name = cluster_labels.get(idx, f'Nhóm {idx}')
                cluster_color = cluster_colors.get(idx, '#667eea')
                cluster_count = len(df[df['cluster'] == idx])
                
                st.markdown(f"""
<div style="
    background: {cluster_color};
    color: white;
    padding: 24px 16px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    transition: transform 0.3s;
">
    <div style="font-size: 36px; font-weight: 800; margin-bottom: 8px;">{cluster_count:,}</div>
    <div style="font-size: 13px; font-weight: 600; opacity: 0.95; line-height: 1.4;">{cluster_name}</div>
</div>
""", unsafe_allow_html=True)
    
    with tab_tips:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 40px;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 40px;
        ">
            <h2 style="
                font-size: 36px;
                font-weight: 700;
                margin-bottom: 16px;
                color: white;
            ">💡 Mẹo & Thủ Thuật Pro</h2>
            <p style="
                font-size: 18px;
                opacity: 0.95;
            ">Nâng cao trải nghiệm tìm kiếm của bạn!</p>
        </div>
        """, unsafe_allow_html=True)
        
        tips = [
            {
                "icon": "🔍",
                "title": "Tìm Kiếm Thông Minh",
                "tip": "Nhập từ khóa ngắn gọn (ví dụ: 'SH 2020' thay vì 'Honda SH Mode 2020') để có kết quả nhanh hơn",
                "color": "#667eea"
            },
            {
                "icon": "🎯",
                "title": "Kết Hợp Bộ Lọc",
                "tip": "Dùng nhiều bộ lọc cùng lúc (Hãng + Khoảng giá + Khu vực) để tìm chính xác nhất",
                "color": "#f5576c"
            },
            {
                "icon": "📥",
                "title": "Xuất & Phân Tích",
                "tip": "Tải CSV để phân tích sâu hơn trên Excel/Google Sheets hoặc chia sẻ với bạn bè",
                "color": "#2ecc71"
            },
            {
                "icon": "📊",
                "title": "Thống Kê Tức Thì",
                "tip": "Dùng nút thống kê để xem giá TB, min/max ngay lập tức, tiết kiệm thời gian tính toán",
                "color": "#3498db"
            },
            {
                "icon": "🔄",
                "title": "So Sánh Nhiều Xe",
                "tip": "Mở nhiều tab trình duyệt để so sánh chi tiết các xe khác nhau cùng lúc",
                "color": "#9b59b6"
            },
            {
                "icon": "📈",
                "title": "Phân Tích Thị Trường",
                "tip": "Vào tab Phân Tích để xem tổng quan thị trường, xu hướng giá và phân bố theo khu vực",
                "color": "#e67e22"
            },
            {
                "icon": "⚙️",
                "title": "Lọc Dung Tích",
                "tip": "Sử dụng bộ lọc dung tích động cơ để tìm xe phù hợp với nhu cầu di chuyển của bạn",
                "color": "#1abc9c"
            },
            {
                "icon": "🔖",
                "title": "Bookmark Yêu Thích",
                "tip": "Lưu link trang chi tiết những xe yêu thích để xem lại sau hoặc theo dõi giá",
                "color": "#e74c3c"
            },
            {
                "icon": "🎨",
                "title": "Phân Khúc Màu Sắc",
                "tip": "Chú ý badge màu của từng nhóm để nhanh chóng nhận diện phân khúc xe",
                "color": "#f39c12"
            }
        ]
        
        for i in range(0, len(tips), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(tips):
                    tip = tips[i + j]
                    with col:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            padding: 24px;
                            border-radius: 16px;
                            margin-bottom: 20px;
                            border-top: 4px solid {tip['color']};
                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                            height: 100%;
                        ">
                            <div style="font-size: 48px; text-align: center; margin-bottom: 16px;">{tip['icon']}</div>
                            <h4 style="
                                font-size: 18px;
                                font-weight: 700;
                                color: {tip['color']};
                                margin-bottom: 12px;
                                text-align: center;
                            ">{tip['title']}</h4>
                            <p style="
                                font-size: 14px;
                                color: #6e6e73;
                                line-height: 1.7;
                                margin: 0;
                                text-align: center;
                            ">{tip['tip']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.success("💡 **Mẹo Pro Cuối Cùng**: Sử dụng tính năng gợi ý xe tương tự để khám phá những lựa chọn bạn chưa từng nghĩ tới!")

def show_admin_page():
    """Trang quản trị viên"""
    st.header("🔑 Trang Quản Trị Viên")
    
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.warning("🔒 Vui lòng đăng nhập để truy cập trang quản trị")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password = st.text_input("🔑 Mật khẩu", type="password", key="admin_password")
            
            if st.button("✅ Đăng nhập", use_container_width=True):
                if password == "admin123":
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Mật khẩu không chính xác!")
        
        st.info("💡 **Gợi ý:** Mật khẩu mặc định là 'admin123'")
        return
    
    if st.button("🚪 Đăng xuất", key="logout_btn"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    st.markdown("---")
    
    # ✅ FIX: Changed from 5 tabs to 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Thống kê tổng quan",
        "📈 Phân Tích Chuyên Sâu",
        "💾 Xuất dữ liệu",
        "🛠️ Quản lý hệ thống"
    ])
    
    with tab1:
        st.subheader("📊 Thống Kê Tổng Quan")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Tổng số xe", f"{len(df):,}")
        
        with col2:
            avg_price = df['price'].mean()
            st.metric("💰 Giá trung bình", f"{avg_price:.1f}M")
        
        with col3:
            avg_km = df['km_driven'].mean()
            st.metric("📍 Km trung bình", f"{avg_km:,.0f} km")
        
        with col4:
            avg_age = df['age'].mean()
            st.metric("📅 Tuổi trung bình", f"{avg_age:.1f} năm")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 📈 Phân bố theo cụm")
            cluster_counts = df['cluster'].value_counts().sort_index()
            cluster_data = pd.DataFrame({
                'Cụm': [cluster_labels.get(i, f'Nhóm {i}') for i in cluster_counts.index],
                'Số lượng': cluster_counts.values,
                'Tỉ lệ (%)': (cluster_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(cluster_data, use_container_width=True, hide_index=True)
        
        with col_b:
            st.markdown("### 🏭 Top 5 thương hiệu")
            brand_counts = df['brand'].value_counts().head(5)
            brand_data = pd.DataFrame({
                'Thương hiệu': brand_counts.index,
                'Số lượng': brand_counts.values,
                'Tỉ lệ (%)': (brand_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(brand_data, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("""
        <h3 style="
            font-size: 24px;
            font-weight: 700;
            color: #1d1d1f;
            margin-bottom: 16px;
        ">📊 Phân Tích Thị Trường Chuyên Sâu</h3>
        """, unsafe_allow_html=True)
        show_analysis_page(show_header=False)
    
    with tab3:
        st.subheader("💾 Xuất Dữ Liệu")
        
        st.markdown("### 🎯 Chọn bộ lọc để xuất")
        
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            export_brands = st.multiselect(
                "Thương hiệu",
                options=['Tất cả'] + sorted(df['brand'].unique().tolist()),
                default=['Tất cả'],
                key="admin_export_brands"
            )
        
        with col_filter2:
            export_clusters = st.multiselect(
                "Cụm",
                options=['Tất cả'] + [cluster_labels.get(i, f'Nhóm {i}') for i in sorted(df['cluster'].unique())],
                default=['Tất cả'],
                key="admin_export_clusters"
            )
        
        export_df = df.copy()
        
        if export_brands and 'Tất cả' not in export_brands:
            export_df = export_df[export_df['brand'].isin(export_brands)]
        
        if export_clusters and 'Tất cả' not in export_clusters:
            cluster_ids = [k for k, v in cluster_labels.items() if v in export_clusters]
            export_df = export_df[export_df['cluster'].isin(cluster_ids)]
        
        st.info(f"📊 Số lượng xe sau khi lọc: **{len(export_df):,}**")
        
        st.markdown("---")
        st.markdown("### 📄 Chọn cột để xuất")
        
        all_export_cols = ['brand', 'model', 'price', 'km_driven', 'age', 'location', 'cluster']
        
        if 'vehicle_type_display' in export_df.columns:
            all_export_cols.append('vehicle_type_display')
        if 'engine_capacity_num' in export_df.columns:
            all_export_cols.append('engine_capacity_num')
        if 'origin_num' in export_df.columns:
            all_export_cols.append('origin_num')
        if 'description' in export_df.columns:
            all_export_cols.append('description')
        
        selected_cols = st.multiselect(
            "Cột dữ liệu",
            options=all_export_cols,
            default=['brand', 'model', 'price', 'km_driven', 'age', 'location', 'cluster'],
            key="admin_selected_cols"
        )
        
        if selected_cols:
            final_export_df = export_df[selected_cols].copy()
            
            if 'cluster' in selected_cols:
                final_export_df['cluster_name'] = final_export_df['cluster'].map(cluster_labels)
            
            col_rename = {
                'brand': 'Hãng',
                'model': 'Model',
                'price': 'Giá (triệu)',
                'km_driven': 'Km đã đi',
                'age': 'Tuổi xe',
                'location': 'Khu vực',
                'cluster': 'Mã nhóm',
                'cluster_name': 'Tên nhóm',
                'vehicle_type_display': 'Loại xe',
                'engine_capacity_num': 'Dung tích',
                'origin_num': 'Xuất xứ',
                'description': 'Mô tả'
            }
            
            final_export_df = final_export_df.rename(
                columns={k: v for k, v in col_rename.items() if k in final_export_df.columns}
            )
            
            st.markdown("### 👀 Xem trước")
            st.dataframe(final_export_df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                csv = final_export_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📅 Tải xuống CSV",
                    data=csv,
                    file_name=f"admin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                excel_buffer = pd.io.common.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    final_export_df.to_excel(writer, index=False, sheet_name='Data')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    "📊 Tải xuống Excel",
                    data=excel_data,
                    file_name=f"admin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_exp3:
                json_data = final_export_df.to_json(orient='records', force_ascii=False).encode('utf-8')
                st.download_button(
                    "📝 Tải xuống JSON",
                    data=json_data,
                    file_name=f"admin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    with tab4:  # ✅ Changed from tab3 to tab4
        st.subheader("🛠️ Quản Lý Hệ Thống")
        
        st.markdown("### 💾 Thông tin hệ thống")
        
        col_sys1, col_sys2 = st.columns(2)
        
        with col_sys1:
            st.markdown(f"""
            - **Tổng số dòng:** {len(df):,}
            - **Tổng số cột:** {len(df.columns)}
            - **Kích thước bộ nhớ:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
            - **Các cột:** {', '.join(df.columns[:10].tolist())}...
            """)
        
        with col_sys2:
            hybrid_status = '✅ Đã load' if hybrid_model else '❌ Chưa load'
            cluster_status = '✅ Đã load' if cluster_model else '❌ Chưa load'
            features_status = '✅ Sẵn sàng' if hybrid_model and hybrid_model.combined_features is not None else '❌ Chưa build'
            
            st.markdown(f"""
            - **Hybrid Model:** {hybrid_status}
            - **Clustering Model:** {cluster_status}
            - **Features built:** {features_status}
            - **Số nhóm:** {len(cluster_labels)}
            """)
        
        st.markdown("---")
        st.markdown("### 🗑️ Cache Management")
        
        col_cache1, col_cache2 = st.columns(2)
        
        with col_cache1:
            if st.button("🔄 Xóa cache dữ liệu", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Đã xóa cache dữ liệu!")
        
        with col_cache2:
            if st.button("🔄 Xóa cache model", use_container_width=True):
                st.cache_resource.clear()
                st.success("✅ Đã xóa cache model! Vui lòng tải lại trang.")

def show_about_page():
    """Trang giới thiệu - Apple/Google Style Design"""
    
    st.markdown("""
    <div style="
        text-align: center;
        padding: 100px 40px 80px 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px;
        margin-bottom: 80px;
        color: white;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="
            font-size: 64px;
            font-weight: 700;
            margin: 0 0 24px 0;
            letter-spacing: -0.03em;
            color: white;
            line-height: 1.1;
        ">
            Hệ Thống Xe Máy Cũ
        </h1>
        <p style="
            font-size: 28px;
            font-weight: 400;
            margin: 0 0 40px 0;
            opacity: 0.95;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.5;
            color: white;
        ">
            Tìm kiếm thông minh · Phân loại tự động · Gợi ý chính xác bằng AI
        </p>
        <div style="margin-top: 50px;">
            <span style="
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(20px);
                padding: 14px 36px;
                border-radius: 30px;
                font-size: 17px;
                font-weight: 600;
                display: inline-block;
                border: 1.5px solid rgba(255, 255, 255, 0.4);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
            ">
                🚀 Version 2.0.1 Optimized
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        text-align: center;
        margin-bottom: 60px;
    ">
        <h2 style="
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 20px;
            color: #1d1d1f;
            letter-spacing: -0.02em;
        ">
            Tại sao chọn chúng tôi?
        </h2>
        <p style="
            font-size: 21px;
            color: #86868b;
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.5;
        ">
            Công nghệ AI tiên tiến kết hợp với trải nghiệm người dùng tối ưu
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            padding: 48px 36px;
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
            text-align: center;
            height: 100%;
            transition: all 0.3s ease;
            border: 1px solid rgba(0, 0, 0, 0.05);
        ">
            <div style="
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 42px;
                margin-bottom: 28px;
                box-shadow: 0 12px 28px rgba(102, 126, 234, 0.25);
            ">🔍</div>
            <h3 style="
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 16px;
                color: #1d1d1f;
                letter-spacing: -0.01em;
            ">
                Tìm Kiếm Thông Minh
            </h3>
            <p style="
                font-size: 16px;
                color: #86868b;
                line-height: 1.6;
                margin: 0;
            ">
                Công nghệ <strong style="color: #667eea;">TF-IDF Vector</strong> với n-gram giúp tìm kiếm chính xác theo ngữ nghĩa
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            padding: 48px 36px;
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
            text-align: center;
            height: 100%;
            transition: all 0.3s ease;
            border: 1px solid rgba(0, 0, 0, 0.05);
        ">
            <div style="
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border-radius: 20px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 42px;
                margin-bottom: 28px;
                box-shadow: 0 12px 28px rgba(240, 147, 251, 0.25);
            ">🤖</div>
            <h3 style="
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 16px;
                color: #1d1d1f;
                letter-spacing: -0.01em;
            ">
                Phân Loại AI
            </h3>
            <p style="
                font-size: 16px;
                color: #86868b;
                line-height: 1.6;
                margin: 0;
            ">
                <strong style="color: #f5576c;">K-Means Clustering</strong> tự động phân loại xe thành 5 phân khúc dựa trên đặc điểm
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: white;
            padding: 48px 36px;
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
            text-align: center;
            height: 100%;
            transition: all 0.3s ease;
            border: 1px solid rgba(0, 0, 0, 0.05);
        ">
            <div style="
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                border-radius: 20px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 42px;
                margin-bottom: 28px;
                box-shadow: 0 12px 28px rgba(79, 172, 254, 0.25);
            ">🎯</div>
            <h3 style="
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 16px;
                color: #1d1d1f;
                letter-spacing: -0.01em;
            ">
                Gợi Ý Chính Xác
            </h3>
            <p style="
                font-size: 16px;
                color: #86868b;
                line-height: 1.6;
                margin: 0;
            ">
                <strong style="color: #00f2fe;">Cosine Similarity</strong> trên đặc trưng đa chiều cho gợi ý xe tương tự với độ chính xác cao
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        padding: 70px 50px;
        border-radius: 28px;
        margin: 1px 0;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.08);
    ">
        <h2 style="
            font-size: 44px;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
            color: #1d1d1f;
            letter-spacing: -0.02em;
        ">
            🛠️ Công Nghệ Tiên Tiến
        </h2>
        <p style="
            text-align: center;
            font-size: 18px;
            color: #86868b;
            margin-bottom: 50px;
        ">
            Được xây dựng trên nền tảng công nghệ hàng đầu
        </p>
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 24px;
            margin-top: 40px;
        ">
            <div style="
                background: white;
                padding: 32px 28px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
                border: 1px solid rgba(0, 0, 0, 0.04);
            ">
                <div style="
                    font-size: 48px;
                    margin-bottom: 16px;
                ">⚡</div>
                <strong style="
                    color: #667eea;
                    font-size: 22px;
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 700;
                ">Streamlit</strong>
                <p style="
                    color: #86868b;
                    font-size: 15px;
                    margin: 0;
                    font-weight: 500;
                ">Modern Web Framework</p>
            </div>
            <div style="
                background: white;
                padding: 32px 28px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
                border: 1px solid rgba(0, 0, 0, 0.04);
            ">
                <div style="
                    font-size: 48px;
                    margin-bottom: 16px;
                ">🧠</div>
                <strong style="
                    color: #f5576c;
                    font-size: 22px;
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 700;
                ">Scikit-learn</strong>
                <p style="
                    color: #86868b;
                    font-size: 15px;
                    margin: 0;
                    font-weight: 500;
                ">Machine Learning Engine</p>
            </div>
            <div style="
                background: white;
                padding: 32px 28px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
                border: 1px solid rgba(0, 0, 0, 0.04);
            ">
                <div style="
                    font-size: 48px;
                    margin-bottom: 16px;
                ">📊</div>
                <strong style="
                    color: #4facfe;
                    font-size: 22px;
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 700;
                ">Pandas</strong>
                <p style="
                    color: #86868b;
                    font-size: 15px;
                    margin: 0;
                    font-weight: 500;
                ">Data Processing</p>
            </div>
            <div style="
                background: white;
                padding: 32px 28px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
                border: 1px solid rgba(0, 0, 0, 0.04);
            ">
                <div style="
                    font-size: 48px;
                    margin-bottom: 16px;
                ">🔤</div>
                <strong style="
                    color: #fa709a;
                    font-size: 22px;
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 700;
                ">TF-IDF</strong>
                <p style="
                    color: #86868b;
                    font-size: 15px;
                    margin: 0;
                    font-weight: 500;
                ">Text Intelligence</p>
            </div>
        </div>
        <div style="
            margin-top: 50px;
            padding: 28px;
            background: white;
            border-radius: 16px;
            text-align: center;
            border: 2px solid rgba(102, 126, 234, 0.2);
        ">
            <p style="
                margin: 0;
                font-size: 16px;
                color: #667eea;
                font-weight: 600;
            ">
                💡 Kết hợp <strong>Machine Learning</strong> + <strong>Natural Language Processing</strong> 
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        text-align: center;
        margin: 20px 0 50px 0;
    ">
        <h2 style="
            font-size: 44px;
            font-weight: 700;
            margin-bottom: 20px;
            color: #1d1d1f;
            letter-spacing: -0.02em;
        ">
            🎨 Phân Khúc Thông Minh
        </h2>
        <p style="
            font-size: 20px;
            color: #86868b;
        ">
            Xe được tự động phân loại thành 5 nhóm bằng thuật toán AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    for i in range(0, len(cluster_labels), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            cluster_id = i + j
            if cluster_id < len(cluster_labels):
                cluster_name = cluster_labels[cluster_id]
                cluster_color = cluster_colors[cluster_id]
                cluster_data = df[df['cluster'] == cluster_id]
                
                with col:
                    st.markdown(f"""
                    <div style="
                        background: white;
                        padding: 36px;
                        border-radius: 24px;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
                        border: 1px solid rgba(0, 0, 0, 0.05);
                        margin-bottom: 24px;
                        transition: all 0.3s ease;
                    ">
                        <div style="
                            width: 72px;
                            height: 72px;
                            background: linear-gradient(135deg, {cluster_color} 0%, {cluster_color}dd 100%);
                            border-radius: 18px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin-bottom: 24px;
                            font-size: 36px;
                            box-shadow: 0 8px 24px {cluster_color}40;
                        ">
                            🏍️
                        </div>
                        <h3 style="
                            font-size: 20px;
                            font-weight: 700;
                            margin-bottom: 20px;
                            color: #1d1d1f;
                            letter-spacing: -0.01em;
                        ">
                            {cluster_name}
                        </h3>
                        <div style="
                            display: flex;
                            justify-content: space-between;
                            margin-bottom: 12px;
                            padding-bottom: 12px;
                            border-bottom: 1.5px solid #f5f5f7;
                        ">
                            <span style="color: #86868b; font-size: 15px; font-weight: 500;">Số lượng</span>
                            <strong style="color: #1d1d1f; font-size: 15px; font-weight: 700;">{len(cluster_data):,} xe</strong>
                        </div>
                        <div style="
                            display: flex;
                            justify-content: space-between;
                            margin-bottom: 16px;
                        ">
                            <span style="color: #86868b; font-size: 15px; font-weight: 500;">Tỷ lệ</span>
                            <strong style="color: {cluster_color}; font-size: 15px; font-weight: 700;">{len(cluster_data)/len(df)*100:.1f}%</strong>
                        </div>
                        <div style="
                            background: linear-gradient(135deg, {cluster_color}10 0%, {cluster_color}20 100%);
                            padding: 20px;
                            border-radius: 16px;
                            text-align: center;
                            margin-top: 20px;
                            border: 1.5px solid {cluster_color}30;
                        ">
                            <div style="color: #86868b; font-size: 13px; margin-bottom: 8px; font-weight: 600;">Giá trung bình</div>
                            <div style="color: {cluster_color}; font-size: 28px; font-weight: 800; letter-spacing: -0.02em;">
                                {cluster_data['price'].mean():.1f}M
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        text-align: center;
        margin: 10px 0 50px 0;
    ">
        <h2 style="
            font-size: 44px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #1d1d1f;
            letter-spacing: -0.02em;
        ">
            👥 Đội Ngũ Phát Triển
        </h2>
        <p style="
            font-size: 20px;
            color: #86868b;
            max-width: 700px;
            margin: 0 auto;
        ">
            Chuyên gia Machine Learning & Data Science
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col_space, col2 = st.columns([1, 0.2, 1])
    
    with col1:
        st.markdown("""
        <div style="
            background: #fafafa;
            padding: 32px;
            border-radius: 20px;
            text-align: center;
            height: 100%;
        ">
        """, unsafe_allow_html=True)
        try:
            import base64
            with open('Hoang_Phuc.jpg', 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 16px;">
                <img src="data:image/jpeg;base64,{img_data}" 
                     style="
                    width: 160px;
                    height: 160px;
                    border-radius: 50%;
                    object-fit: cover;
                    border: 4px solid #667eea;
                    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
                ">
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 16px;">
                <div style="
                    width: 160px;
                    height: 160px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 80px;
                    box-shadow: 0 12px 36px rgba(102, 126, 234, 0.3);
                ">👨‍💻</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="
            font-size: 28px;
            font-weight: 700;
            margin: 24px 0 12px 0;
            color: #1d1d1f;
            letter-spacing: -0.01em;
        ">Hoàng Phúc</h3>
        <p style="
            font-size: 18px;
            color: #667eea;
            font-weight: 600;
            margin-bottom: 24px;
        ">UX/UI Designer & ML Engineer</p>
        <div style="
            text-align: left;
            background: #f5f5f7;
            padding: 24px;
            border-radius: 16px;
            margin-top: 20px;
        ">
            <p style="
                font-size: 15px;
                color: #1d1d1f;
                line-height: 1.7;
                margin: 0 0 12px 0;
            "><strong style="color: #667eea;">🎨 Chuyên môn:</strong></p>
            <ul style="
                margin: 0;
                padding-left: 20px;
                font-size: 15px;
                color: #6e6e73;
                line-height: 1.8;
            ">
                <li>Thiết kế UX/UI cho ứng dụng</li>
                <li>Xây dựng mô hình phân loại K-Means</li>
                <li>Khám phá & phân tích dữ liệu</li>
                <li>Tối ưu hóa trải nghiệm người dùng</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_space:
        st.write("")
    
    with col2:
        st.markdown("""
        <div style="
            background: #fafafa;
            padding: 32px;
            border-radius: 20px;
            text-align: center;
            height: 100%;
        ">
        """, unsafe_allow_html=True)
        try:
            import base64
            with open('Bich_Thuy.jpg', 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 16px;">
                <img src="data:image/jpeg;base64,{img_data}" 
                     style="
                    width: 160px;
                    height: 160px;
                    border-radius: 50%;
                    object-fit: cover;
                    border: 4px solid #f5576c;
                    box-shadow: 0 8px 24px rgba(245, 87, 108, 0.3);
                ">
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 16px;">
                <div style="
                    width: 160px;
                    height: 160px;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    border-radius: 50%;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 80px;
                    box-shadow: 0 12px 36px rgba(240, 147, 251, 0.3);
                ">👩‍💻</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="
            font-size: 28px;
            font-weight: 700;
            margin: 24px 0 12px 0;
            color: #1d1d1f;
            letter-spacing: -0.01em;
        ">Bích Thủy</h3>
        <p style="
            font-size: 18px;
            color: #f5576c;
            font-weight: 600;
            margin-bottom: 24px;
        ">Data Scientist & ML Engineer</p>
        <div style="
            text-align: left;
            background: #f5f5f7;
            padding: 24px;
            border-radius: 16px;
            margin-top: 20px;
        ">
            <p style="
                font-size: 15px;
                color: #1d1d1f;
                line-height: 1.7;
                margin: 0 0 12px 0;
            "><strong style="color: #f5576c;">🔬 Chuyên môn:</strong></p>
            <ul style="
                margin: 0;
                padding-left: 20px;
                font-size: 15px;
                color: #6e6e73;
                line-height: 1.8;
            ">
                <li>Xây dựng mô hình gợi ý</li>
                <li>Khám phá & xử lý dữ liệu lớn</li>
                <li>Phân tích ngữ nghĩa TF-IDF</li>
                <li>Tối ưu hóa thuật toán ML</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px 20px;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin: 60px 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    ">
        <h3 style="
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 16px;
            color: white;
        ">Liên Hệ Với Chúng Tôi</h3>
        <p style="
            font-size: 18px;
            opacity: 0.95;
            font-weight: 500;
        ">📧 Email: <strong style="font-weight: 700;">phucthuy@buonbanxemay.vn</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        text-align: center;
        padding: 50px 20px;
        color: #86868b;
    ">
        <p style="font-size: 15px; margin-bottom: 12px; font-weight: 500;">
            © 2025 Hệ Thống Xe Máy Cũ. All rights reserved.
        </p>
        <p style="font-size: 14px; opacity: 0.8; font-weight: 400;">
            Powered by PhucThuy Technologies
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# 🚀 MAIN APPLICATION
# ==============================

show_banner()

if 'page' not in st.session_state:
    st.session_state.page = "about"
if 'selected_bike_idx' not in st.session_state:
    st.session_state.selected_bike_idx = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'search_page_num' not in st.session_state:
    st.session_state.search_page_num = 0

with st.sidebar:
    st.markdown("## Điều Hướng")
    
    nav_map = {
        "🏠 Trang Chủ": "home",
        "🔍 Tìm Kiếm": "search",
        "📝 Đăng Bán": "sell",  # ✅ THÊM MỚI
        "🔑 Quản Trị": "admin",
        "📘 Hướng Dẫn": "help",
        "📖 Giới Thiệu": "about"
    }
        
    labels = list(nav_map.keys())
    current_page = st.session_state.page
    
    if current_page == "detail":
        current_page = "search"
    
    current_label = None
    for lab, p in nav_map.items():
        if current_page == p:
            current_label = lab
            break
    
    default_index = labels.index(current_label) if current_label in labels else 0
    
    nav_choice = st.radio(
        label="Chọn trang",
        options=labels,
        index=default_index,
        key="nav_radio",
        label_visibility="collapsed"
    )
    
    selected_page = nav_map.get(nav_choice)
    if selected_page and selected_page != st.session_state.page:
        if not (st.session_state.page == "detail" and selected_page == "search"):
            st.session_state.page = selected_page
            st.rerun()
    
st.markdown("""
    <div style='
        margin-top: 30px;
        margin-bottom: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    '>
        <h4 style='margin: 0 0 10px 0; color: white;'>👥 Tác Giả</h4>
        <p style='margin: 5px 0; font-size: 14px;'>
            <strong>Hoàng Phúc & Bích Thủy</strong>
        </p>
        <hr style='border: 1px solid rgba(255,255,255,0.3); margin: 10px 0;'>
        <p style='margin: 5px 0; font-size: 13px;'>
            📅 <strong>Ngày phát hành:</strong><br>28/11/2025
        </p>
    </div>
""", unsafe_allow_html=True)



# ==============================
# 🔀 PAGE ROUTING
# ==============================

if st.session_state.page == "about":
    show_about_page()
elif st.session_state.page == "help":
    show_help_page()
elif st.session_state.page == "search":
    show_search_page()
elif st.session_state.page == "sell":  # ✅ THÊM MỚI
    show_sell_page()
elif st.session_state.page == "admin":
    show_admin_page()
elif st.session_state.page == "detail":
    show_detail_page()
else:
    show_home_page()

st.markdown("---")
st.markdown(f"*Hệ thống gợi ý xe máy - Tổng số xe: {len(df):,}*")

