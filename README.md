# Customer Churn Prediction with LightGBM

So sánh Logistic Regression, LightGBM, RandomizedSearchCV và early stopping cho bài toán dự đoán khách hàng rời ngân hàng (`Exited`). Notebook có EDA, đánh giá mô hình, tối ưu ngưỡng theo F1 và giải thích bằng SHAP.

## Cấu trúc

```text
lightgbm-churn/
├── ML_lightGBM.ipynb   # Toàn bộ quy trình, đã xóa output để dễ xem diff
├── requirements.txt
├── .gitignore
└── data/
    └── README.md      # Hướng dẫn chuẩn bị dữ liệu
```

## Chạy trên máy

Dùng Python 3.12 hoặc 3.13. Chạy các lệnh tại thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Đặt file dữ liệu của bạn tại `data/Churn_Modelling.csv`, mở `ML_lightGBM.ipynb`, chọn kernel của `.venv` rồi **Restart Kernel and Run All Cells**. Nếu cần đăng ký kernel:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name lightgbm-churn --display-name "Python (lightgbm-churn)"
```

Cell đầu có cấu hình đường dẫn, số lượt tìm kiếm (`N_ITER=40`), số fold (`CV=5`) và bật/tắt SHAP (`RUN_SHAP=True`). Có thể đặt `N_ITER=2`, `CV=2`, `RUN_SHAP=False` để kiểm tra nhanh. Khi tìm kiếm, chỉ một tầng dùng toàn bộ CPU để tránh chạy song song lồng nhau.

Notebook tạo `artifacts/` chứa pipeline LightGBM fine-tuned, ngưỡng dự đoán và bảng so sánh CSV. Pipeline đã bao gồm preprocessing; luôn dùng ngưỡng đi kèm khi chuyển xác suất thành nhãn. Ví dụ lưu/đọc model và dự đoán nằm cuối notebook.

## Phương pháp và giới hạn

- Chia train/test theo tỷ lệ 80/20, phân tầng theo `Exited`, seed 42.
- One-hot encoding cho Geography/Gender; chuẩn hóa số cho Logistic Regression.
- Tìm tham số LightGBM bằng ROC-AUC; chọn ngưỡng bằng F1 trên validation tách từ tập train.
- Early stopping tìm số cây rồi huấn luyện lại trên toàn bộ tập train.
- Bảng xếp hạng test dùng để báo cáo thí nghiệm. File xuất dùng chiến lược fine-tuned + ngưỡng validation đã xác định trong notebook, không tự chọn model theo bảng test.

Giữ phương pháp thí nghiệm của notebook gốc: validation được tách từ tập train sau khi đã tìm tham số trên tập train đó, đồng thời được dùng cho early stopping và chọn ngưỡng. Điểm validation vì vậy chưa phải ước lượng độc lập. Khi cần đánh giá nghiêm ngặt hơn, dành validation riêng trước bước tuning hoặc dùng nested CV; giữ test riêng cho đánh giá cuối cùng.

Notebook gốc có output đã lưu, nhưng repo này xóa output để giảm dung lượng. Chưa chạy lại trên `Churn_Modelling.csv` vì dữ liệu không có trong workspace; không coi các số liệu cũ là kết quả đã tái lập. Các phiên bản thư viện được giới hạn theo major version, không phải lockfile tái lập chính xác môi trường cũ.

## Đưa lên GitHub

Repo đã được khởi tạo Git tại thư mục này. Tạo một repository trống trên GitHub, sau đó chạy:

```powershell
git add .
git commit -m "Add customer churn LightGBM notebook"
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

Dữ liệu CSV, model, môi trường ảo và thư mục `artifacts/` đã được bỏ qua bằng `.gitignore`. Trước những commit sau, chọn **Clear Outputs of All Cells** rồi lưu notebook để giữ repo nhỏ.
