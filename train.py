
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu Iris.csv
df = pd.read_csv("Iris.csv")

# Chọn 4 biến đặc trưng
X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

# Chuyển tên loài thành mã số
species = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

y = df["Species"].map(species)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Tạo mô hình SVM
model = SVC(kernel="linear")

# Huấn luyện
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Lưu mô hình
joblib.dump(model, "svm_model.pkl")

print("Model saved!")
