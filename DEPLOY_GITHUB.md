# 🚀 Hướng Dẫn Deploy Lên GitHub

## Bước 1: Khởi tạo Git Repository (nếu chưa có)

```bash
cd Done
git init
```

## Bước 2: Add tất cả files

```bash
git add .
```

## Bước 3: Commit

```bash
git commit -m "Initial commit: Motorcycle Recommendation System with ML"
```

## Bước 4: Tạo Repository trên GitHub

1. Truy cập: https://github.com/new
2. Điền thông tin:
   - Repository name: `motorcycle-recommendation-system`
   - Description: `🏍️ AI-powered motorcycle recommendation system with K-Means clustering & hybrid recommender`
   - Public/Private: Chọn Public
3. **KHÔNG** check "Add README" (vì đã có sẵn)
4. Click "Create repository"

## Bước 5: Connect & Push

```bash
git remote add origin https://github.com/YOUR-USERNAME/motorcycle-recommendation-system.git
git branch -M main
git push -u origin main
```

## Bước 6: Cập nhật README với link của bạn

Sửa phần này trong README.md:
```markdown
git clone https://github.com/YOUR-USERNAME/motorcycle-recommendation-system.git
```

Thay `YOUR-USERNAME` bằng username GitHub của bạn.

## 📦 Danh Sách Files Đã Được Copy

✅ **Core Application**
- final_app.py (184 KB) - Main Streamlit app

✅ **ML Models**
- clustering_model.joblib (28 KB) - K-Means K=5
- clustering_scaler.joblib (4 KB) - ColumnTransformer
- clustering_info.joblib (1 KB) - Cluster metadata
- hybrid_model.joblib (12 MB) - Hybrid recommender

✅ **Datasets**
- df_clustering.parquet (944 KB) - Main dataset (6,695 bikes)
- motorcycles_clustered_v2_final.csv (3.2 MB) - CSV backup
- user_listings.parquet (11 KB) - User-generated listings
- user_listings_backup.parquet (11 KB) - Backup

✅ **Source Code**
- src/ folder - Modular components
  - components/ (cards, filters)
  - config/ (settings)
  - models/ (recommender)
  - styles/ (apple_design)
  - utils/ (data_loader, helpers)

✅ **Configuration**
- requirements.txt - Python dependencies
- setup.bat - Setup script (Windows)
- start_app.bat - Start script (Windows)
- .gitignore - Git ignore rules

✅ **Documentation**
- README.md - Full documentation

## 🎯 Commands Tóm Tắt (Copy & Paste)

```bash
# Di chuyển vào folder Done
cd Done

# Khởi tạo Git (nếu chưa có)
git init

# Add tất cả files
git add .

# Commit
git commit -m "Initial commit: Motorcycle Recommendation System"

# Connect với GitHub repo (thay YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/motorcycle-recommendation-system.git

# Push lên GitHub
git branch -M main
git push -u origin main
```

## 🔄 Cập Nhật Sau Này

Khi có thay đổi:

```bash
git add .
git commit -m "Update: Mô tả thay đổi"
git push
```

## 🌐 Deploy Lên Streamlit Cloud (Bonus)

1. Truy cập: https://share.streamlit.io/
2. Sign in với GitHub
3. Click "New app"
4. Chọn repository: `motorcycle-recommendation-system`
5. Branch: `main`
6. Main file: `final_app.py`
7. Click "Deploy"

**Lưu ý**: Streamlit Cloud có giới hạn 1GB, nên có thể cần giảm kích thước hybrid_model.joblib nếu muốn deploy.

## ✨ Done!

Repository của bạn sẽ có cấu trúc:
```
motorcycle-recommendation-system/
├── 📄 README.md (Documentation đầy đủ)
├── 📄 requirements.txt
├── 🐍 final_app.py (Main app)
├── 🧠 *.joblib (ML models)
├── 📊 *.parquet, *.csv (Datasets)
├── 📁 src/ (Source modules)
├── 🔧 setup.bat, start_app.bat
└── 📄 .gitignore
```

## 🏆 Kết Quả

- ✅ Code được version control
- ✅ Backup an toàn trên cloud
- ✅ Chia sẻ với người khác
- ✅ Portfolio cho GitHub profile
- ✅ Có thể deploy public
