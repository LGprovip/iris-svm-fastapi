# Iris SVM FastAPI 🌸

Hệ thống phân loại hoa Iris sử dụng thuật toán học máy
SVM (Support Vector Machine), xây dựng API bằng FastAPI
và hỗ trợ triển khai ứng dụng trên Render Cloud.

## 📌 1. Giới thiệu

Dự án xây dựng mô hình học máy nhằm phân loại hoa Iris
dựa trên các đặc trưng hình thái của hoa.

Người dùng cung cấp 4 thông số đầu vào:
- Sepal Length: Chiều dài đài hoa.
- Sepal Width: Chiều rộng đài hoa.
- Petal Length: Chiều dài cánh hoa.
- Petal Width: Chiều rộng cánh hoa.

Từ các thông số này, mô hình SVM đưa ra dự đoán về
loài hoa thuộc một trong ba nhóm:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

### Công nghệ sử dụng

- Python: Ngôn ngữ lập trình chính.
- Pandas: Hỗ trợ xử lý dữ liệu.
- Scikit-learn: Xây dựng và huấn luyện mô hình SVM.
- FastAPI: Xây dựng REST API.
- Uvicorn: Chạy ứng dụng FastAPI.
- Render: Nền tảng triển khai ứng dụng trực tuyến.
- Cloudflare Tunnel: Hỗ trợ tạo đường dẫn truy cập
  từ bên ngoài đến ứng dụng chạy cục bộ, nếu được sử dụng.

## 📂 2. Cấu trúc dự án

```text
.
├── sample_data/
│   └──                         # Thư mục dữ liệu mẫu
├── Iris.csv                    # Bộ dữ liệu hoa Iris
├── Procfile                    # Lệnh khởi chạy ứng dụng
├── app.py                      # Ứng dụng FastAPI và các API
├── cloudflared                 # Tệp thực thi Cloudflare Tunnel
├── cloudflared.log             # Nhật ký Cloudflare Tunnel
├── render.yaml                 # Cấu hình triển khai trên Render
├── requirements.txt            # Danh sách thư viện Python
├── svm_model.pkl               # Mô hình SVM đã huấn luyện
├── train.py                    # Mã nguồn huấn luyện mô hình
└── README.md                   # Tài liệu giới thiệu dự án
```
Sinh viên thực hiện: Hoàng Gia

Lớp: 24CKDL

Trường: Trường Đại học Sư phạm – ĐH Đà Nẵng, Khoa Toán – Tin