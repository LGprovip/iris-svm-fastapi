from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
