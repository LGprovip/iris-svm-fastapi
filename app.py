from fastapi import FastAPI
from fastapi.responses import HTMLResponseimport joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Iris Species Classifier",
    description="Ứng dụng dự đoán loài hoa Iris với giao diện tương tác hiện đại",
    version="2.0.0"
)

# Load mô hình Machine Learning (đảm bảo model hỗ trợ predict_proba, ví dụ SVM với probability=True)
try:
    model = joblib.load("svm_model.pkl")
except Exception as e:
    model = None

# Cấu hình dữ liệu đầu vào với Validation
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=15, description="Chiều dài đài hoa (cm)", example=5.1)
    sepal_width: float = Field(..., gt=0, lt=10, description="Chiều rộng đài hoa (cm)", example=3.5)
    petal_length: float = Field(..., gt=0, lt=15, description="Chiều dài cánh hoa (cm)", example=1.4)
    petal_width: float = Field(..., gt=0, lt=10, description="Chiều rộng cánh hoa (cm)", example=0.2)

# Từ điển thông tin loài hoa
SPECIES_INFO = {
    0: {
        "name": "Setosa",
        "badge": "🌸 Setosa",
        "description": "Kích thước đài hoa lớn, cánh hoa nhỏ. Rất dễ phân biệt.",
        "color": "#4facfe"
    },
    1: {
        "name": "Versicolor",
        "badge": "🌿 Versicolor",
        "description": "Kích thước trung bình, màu sắc đa dạng.",
        "color": "#43e97b"
    },
    2: {
        "name": "Virginica",
        "badge": "🌺 Virginica",
        "description": "Cánh hoa dài và rộng nhất trong các loài Iris.",
        "color": "#fa709a"
    }
}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Classifier AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
                --glass-bg: rgba(255, 255, 255, 0.05);
                --glass-border: rgba(255, 255, 255, 0.12);
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }

            body {
                min-height: 100vh;
                background: var(--bg-gradient);
                color: var(--text-main);
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
                overflow-x: hidden;
            }

            .container {
                width: 100%;
                max-width: 900px;
                background: var(--glass-bg);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid var(--glass-border);
                border-radius: 28px;
                padding: 40px;
                box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
                position: relative;
            }

            .header { text-align: center; margin-bottom: 35px; }
            .header h1 {
                font-size: 38px;
                font-weight: 700;
                background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            }
            .header p { color: var(--text-muted); font-size: 15px; }

            .form-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }

            .input-card {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--glass-border);
                border-radius: 16px;
                padding: 16px;
                transition: all 0.3s ease;
            }
            .input-card:focus-within {
                border-color: var(--primary);
                box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
            }

            .input-card label {
                display: block;
                font-size: 13px;
                font-weight: 600;
                color: var(--text-muted);
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .input-card input {
                width: 100%;
                background: transparent;
                border: none;
                color: #fff;
                font-size: 22px;
                font-weight: 600;
                outline: none;
            }

            .btn-predict {
                width: 100%;
                padding: 18px;
                border: none;
                border-radius: 16px;
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                color: white;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
            }
            .btn-predict:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 30px rgba(99, 102, 241, 0.6);
            }

            .result-container {
                display: none;
                margin-top: 35px;
                padding: 25px;
                border-radius: 20px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--glass-border);
                animation: fadeIn 0.5s ease;
            }

            .result-main {
                text-align: center;
                margin-bottom: 25px;
            }
            .result-badge {
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 5px;
            }
            .result-desc { color: var(--text-muted); font-size: 14px; }

            .proba-bars { display: flex; flex-direction: column; gap: 12px; }
            .proba-item { display: flex; flex-direction: column; gap: 5px; }
            .proba-label { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; }
            .bar-bg {
                width: 100%;
                height: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                overflow: hidden;
            }
            .bar-fill {
                height: 100%;
                border-radius: 10px;
                width: 0%;
                transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .spinner {
                display: none;
                width: 22px;
                height: 22px;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 0.8s linear infinite;
            }

            @keyframes spin { to { transform: rotate(360deg); } }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            .footer-links {
                margin-top: 30px;
                display: flex;
                justify-content: center;
                gap: 20px;
            }
            .footer-links a {
                color: var(--text-muted);
                text-decoration: none;
                font-size: 13px;
                transition: color 0.2s;
            }
            .footer-links a:hover { color: #fff; }
        </style>
    </head>
    <body>

        <div class="container">
            <div class="header">
                <h1>Iris AI Classifier</h1>
                <p>Nhập thông số hình thái học để nhận diện loài hoa Iris tức thì</p>
            </div>

            <div class="form-grid">
                <div class="input-card">
                    <label>Sepal Length (cm)</label>
                    <input type="number" id="sepal_length" step="0.1" value="5.1">
                </div>
                <div class="input-card">
                    <label>Sepal Width (cm)</label>
                    <input type="number" id="sepal_width" step="0.1" value="3.5">
                </div>
                <div class="input-card">
                    <label>Petal Length (cm)</label>
                    <input type="number" id="petal_length" step="0.1" value="1.4">
                </div>
                <div class="input-card">
                    <label>Petal Width (cm)</label>
                    <input type="number" id="petal_width" step="0.1" value="0.2">
                </div>
            </div>

            <button class="btn-predict" onclick="predict()">
                <div class="spinner" id="spinner"></div>
                <span id="btn-text">Dự Đoán Tức Thì ✨</span>
            </button>

            <div class="result-container" id="resultCard">
                <div class="result-main">
                    <div class="result-badge" id="resBadge">--</div>
                    <div class="result-desc" id="resDesc">--</div>
                </div>

                <div class="proba-bars" id="probaBars"></div>
            </div>

            <div class="footer-links">
                <a href="/docs" target="_blank">Swagger API</a>
                <a href="/redoc" target="_blank">ReDoc</a>
                <a href="/health" target="_blank">Health Check</a>
            </div>
        </div>

        <script>
            async function predict() {
                const spinner = document.getElementById("spinner");
                const btnText = document.getElementById("btn-text");
                const resultCard = document.getElementById("resultCard");

                const payload = {
                    sepal_length: parseFloat(document.getElementById("sepal_length").value),
                    sepal_width: parseFloat(document.getElementById("sepal_width").value),
                    petal_length: parseFloat(document.getElementById("petal_length").value),
                    petal_width: parseFloat(document.getElementById("petal_width").value)
                };

                // Show spinner
                spinner.style.display = "block";
                btnText.innerText = "Đang phân tích...";

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();

                    if (!response.ok) throw new Error(data.detail || "Có lỗi xảy ra");

                    // Render UI Result
                    document.getElementById("resBadge").innerText = data.info.badge;
                    document.getElementById("resBadge").style.color = data.info.color;
                    document.getElementById("resDesc").innerText = data.info.description;

                    // Render Probability Bars
                    const probaContainer = document.getElementById("probaBars");
                    probaContainer.innerHTML = "";

                    if (data.probabilities) {
                        Object.keys(data.probabilities).forEach(speciesName => {
                            const proba = data.probabilities[speciesName];
                            const percent = (proba * 100).toFixed(1);

                            const item = document.createElement("div");
                            item.className = "proba-item";
                            item.innerHTML = `
                                <div class="proba-label">
                                    <span>${speciesName}</span>
                                    <span>${percent}%</span>
                                </div>
                                <div class="bar-bg">
                                    <div class="bar-fill" style="width: 0%; background: ${speciesName === data.prediction ? data.info.color : '#6366f1'}"></div>
                                </div>
                            `;
                            probaContainer.appendChild(item);

                            // Trigger animation
                            setTimeout(() => {
                                item.querySelector(".bar-fill").style.width = percent + "%";
                            }, 50);
                        });
                    }

                    resultCard.style.display = "block";

                } catch (err) {
                    alert(err.message);
                } finally {
                    spinner.style.display = "none";
                    btnText.innerText = "Dự Đoán Tức Thì ✨";
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health_check():
    return {
        "status": "online" if model is not None else "model_missing",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_iris(data: IrisInput):
    if model is None:
        return {"error": "Mô hình chưa được nạp. Hãy kiểm tra file svm_model.pkl"}

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    pred_class = int(model.predict(features)[0])
    info = SPECIES_INFO.get(pred_class, {"name": "Unknown", "badge": "❓ Unknown", "description": "", "color": "#fff"})

    # Tính toán xác suất (Nếu model hỗ trợ predict_proba)
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        for idx, prob in enumerate(probs):
            name = SPECIES_INFO[idx]["name"]
            probabilities[name] = float(prob)

    return {
        "class_id": pred_class,
        "prediction": info["name"],
        "info": info,
        "probabilities": probabilities
    }
from pydantic import BaseModel
import joblib

# Load mô hình SVM
model = joblib.load("svm_model.pkl")

# Khởi tạo FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0"
)


# =========================
# DỮ LIỆU ĐẦU VÀO
# =========================

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# =========================
# TÊN CÁC LỚP
# =========================

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


# =========================
# TRANG WEB CHÍNH
# =========================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>
    <html lang="vi">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>Iris Classification</title>

        <style>

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: Arial, sans-serif;

                min-height: 100vh;

                background:
                    linear-gradient(
                        135deg,
                        #667eea,
                        #764ba2
                    );

                display: flex;
                justify-content: center;
                align-items: center;

                padding: 30px;
            }

            .container {
                width: 100%;
                max-width: 850px;

                background: rgba(255,255,255,0.96);

                border-radius: 25px;

                padding: 40px;

                box-shadow:
                    0 20px 50px rgba(0,0,0,0.25);
            }

            .header {
                text-align: center;
                margin-bottom: 35px;
            }

            .flower {
                font-size: 55px;
                margin-bottom: 10px;
            }

            h1 {
                color: #333;
                font-size: 34px;
                margin-bottom: 10px;
            }

            .subtitle {
                color: #777;
                font-size: 16px;
            }

            .model {
                display: inline-block;

                margin-top: 15px;

                padding: 7px 15px;

                border-radius: 20px;

                background: #eee;

                color: #555;

                font-size: 14px;
            }

            .form-grid {
                display: grid;

                grid-template-columns:
                    repeat(2, 1fr);

                gap: 20px;

                margin-bottom: 25px;
            }

            .input-group label {
                display: block;

                font-weight: bold;

                color: #444;

                margin-bottom: 8px;
            }

            .input-group input {
                width: 100%;

                padding: 14px;

                border: 2px solid #ddd;

                border-radius: 12px;

                font-size: 16px;

                outline: none;

                transition: 0.3s;
            }

            .input-group input:focus {
                border-color: #667eea;

                box-shadow:
                    0 0 0 3px
                    rgba(102,126,234,0.15);
            }

            .unit {
                color: #999;

                font-size: 13px;

                margin-top: 5px;
            }

            button {
                width: 100%;

                padding: 16px;

                border: none;

                border-radius: 14px;

                background:
                    linear-gradient(
                        135deg,
                        #667eea,
                        #764ba2
                    );

                color: white;

                font-size: 18px;

                font-weight: bold;

                cursor: pointer;

                transition: 0.3s;
            }

            button:hover {
                transform: translateY(-2px);

                box-shadow:
                    0 10px 25px
                    rgba(102,126,234,0.35);
            }

            button:active {
                transform: translateY(0);
            }

            .result {
                display: none;

                margin-top: 30px;

                padding: 25px;

                border-radius: 18px;

                text-align: center;

                background: #f5f7ff;

                border: 2px solid #e1e5ff;
            }

            .result-title {
                color: #777;

                font-size: 15px;

                margin-bottom: 8px;
            }

            .result-name {
                font-size: 32px;

                font-weight: bold;

                color: #667eea;

                margin-bottom: 8px;
            }

            .class-id {
                color: #666;

                font-size: 14px;
            }

            .error {
                display: none;

                margin-top: 20px;

                padding: 15px;

                background: #ffe5e5;

                color: #c0392b;

                border-radius: 12px;

                text-align: center;
            }

            .footer {
                text-align: center;

                margin-top: 30px;

                color: #999;

                font-size: 13px;
            }

            .links {
                margin-top: 15px;
            }

            .links a {
                color: #667eea;

                text-decoration: none;

                margin: 0 8px;

                font-weight: bold;
            }

            .links a:hover {
                text-decoration: underline;
            }

            @media (max-width: 600px) {

                .container {
                    padding: 25px;
                }

                .form-grid {
                    grid-template-columns: 1fr;
                }

                h1 {
                    font-size: 28px;
                }

            }

        </style>

    </head>


    <body>

        <div class="container">

            <div class="header">

                <div class="flower">
                    🌸
                </div>

                <h1>
                    Iris Classification
                </h1>

                <p class="subtitle">
                    Dự đoán loài hoa Iris bằng mô hình SVM
                </p>

                <span class="model">
                    Machine Learning • SVM • FastAPI
                </span>

            </div>


            <div class="form-grid">

                <div class="input-group">

                    <label>
                        Sepal Length
                    </label>

                    <input
                        type="number"
                        id="sepal_length"
                        step="0.1"
                        value="5.1"
                    >

                    <div class="unit">
                        Chiều dài đài hoa (cm)
                    </div>

                </div>


                <div class="input-group">

                    <label>
                        Sepal Width
                    </label>

                    <input
                        type="number"
                        id="sepal_width"
                        step="0.1"
                        value="3.5"
                    >

                    <div class="unit">
                        Chiều rộng đài hoa (cm)
                    </div>

                </div>


                <div class="input-group">

                    <label>
                        Petal Length
                    </label>

                    <input
                        type="number"
                        id="petal_length"
                        step="0.1"
                        value="1.4"
                    >

                    <div class="unit">
                        Chiều dài cánh hoa (cm)
                    </div>

                </div>


                <div class="input-group">

                    <label>
                        Petal Width
                    </label>

                    <input
                        type="number"
                        id="petal_width"
                        step="0.1"
                        value="0.2"
                    >

                    <div class="unit">
                        Chiều rộng cánh hoa (cm)
                    </div>

                </div>

            </div>


            <button onclick="predict()">
                🔍 Dự đoán loài hoa
            </button>


            <div class="result" id="result">

                <div class="result-title">
                    Kết quả dự đoán
                </div>

                <div
                    class="result-name"
                    id="prediction"
                >
                </div>

                <div
                    class="class-id"
                    id="class_id"
                >
                </div>

            </div>


            <div class="error" id="error">
            </div>


            <div class="footer">

                Mô hình SVM được huấn luyện
                trên bộ dữ liệu Iris

                <div class="links">

                    <a href="/docs" target="_blank">
                        API Docs
                    </a>

                    <a href="/health" target="_blank">
                        Health Check
                    </a>

                </div>

            </div>

        </div>


        <script>

            async function predict() {

                const result =
                    document.getElementById("result");

                const error =
                    document.getElementById("error");

                result.style.display = "none";

                error.style.display = "none";


                const data = {

                    sepal_length:
                        parseFloat(
                            document.getElementById(
                                "sepal_length"
                            ).value
                        ),

                    sepal_width:
                        parseFloat(
                            document.getElementById(
                                "sepal_width"
                            ).value
                        ),

                    petal_length:
                        parseFloat(
                            document.getElementById(
                                "petal_length"
                            ).value
                        ),

                    petal_width:
                        parseFloat(
                            document.getElementById(
                                "petal_width"
                            ).value
                        )

                };


                if (
                    Object.values(data)
                        .some(value => isNaN(value))
                ) {

                    error.innerText =
                        "Vui lòng nhập đầy đủ 4 thông số.";

                    error.style.display = "block";

                    return;

                }


                try {

                    const response =
                        await fetch(
                            "/predict",
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify(data)
                            }
                        );


                    if (!response.ok) {
                        throw new Error(
                            "Không thể kết nối API."
                        );
                    }


                    const resultData =
                        await response.json();


                    document.getElementById(
                        "prediction"
                    ).innerText =
                        "🌸 " +
                        resultData.prediction;


                    document.getElementById(
                        "class_id"
                    ).innerText =
                        "Class ID: " +
                        resultData.class_id;


                    result.style.display =
                        "block";

                }

                catch (err) {

                    error.innerText =
                        "Có lỗi xảy ra: " +
                        err.message;

                    error.style.display =
                        "block";

                }

            }

        </script>

    </body>

    </html>
    """


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================
# DỰ ĐOÁN
# =========================

@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(
        model.predict(features)[0]
    )

    return {
        "class_id": prediction,
        "prediction": species[prediction]
    }
