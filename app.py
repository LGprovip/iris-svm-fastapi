from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib


# ============================================================
# LOAD MÔ HÌNH
# ============================================================

model = joblib.load("svm_model.pkl")


# ============================================================
# KHỞI TẠO FASTAPI
# ============================================================

app = FastAPI(
    title="Iris Classification API",
    description="Iris Flower Classification using SVM",
    version="1.0.0"
)


# ============================================================
# DỮ LIỆU ĐẦU VÀO
# ============================================================

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ============================================================
# TÊN CÁC LỚP
# ============================================================

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


# ============================================================
# TRANG WEB
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html lang="vi">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Iris AI • SVM Classification</title>


    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }


        body {

            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                Arial,
                sans-serif;

            min-height: 100vh;

            color: #172033;

            background:
                radial-gradient(
                    circle at 10% 20%,
                    rgba(124, 92, 255, 0.45),
                    transparent 35%
                ),

                radial-gradient(
                    circle at 90% 80%,
                    rgba(0, 210, 255, 0.35),
                    transparent 35%
                ),

                linear-gradient(
                    135deg,
                    #111827,
                    #312e81,
                    #4c1d95
                );

            padding: 40px 20px;

        }


        /* =========================
           CONTAINER
        ========================= */

        .page {

            width: 100%;

            max-width: 1050px;

            margin: auto;

        }


        /* =========================
           HEADER
        ========================= */

        .header {

            text-align: center;

            color: white;

            margin-bottom: 30px;

        }


        .logo {

            width: 78px;

            height: 78px;

            margin: auto;

            display: flex;

            align-items: center;

            justify-content: center;

            font-size: 40px;

            border-radius: 24px;

            background:
                rgba(255,255,255,0.15);

            border:
                1px solid rgba(255,255,255,0.25);

            backdrop-filter:
                blur(15px);

            box-shadow:
                0 15px 40px
                rgba(0,0,0,0.25);

            margin-bottom: 18px;

        }


        .header h1 {

            font-size: 42px;

            font-weight: 800;

            letter-spacing: -1px;

            margin-bottom: 10px;

        }


        .header p {

            font-size: 17px;

            color:
                rgba(255,255,255,0.78);

        }


        .badge {

            display: inline-flex;

            align-items: center;

            gap: 7px;

            margin-top: 18px;

            padding: 8px 15px;

            border-radius: 30px;

            background:
                rgba(255,255,255,0.12);

            border:
                1px solid rgba(255,255,255,0.2);

            font-size: 13px;

            color: white;

        }


        .status-dot {

            width: 8px;

            height: 8px;

            border-radius: 50%;

            background: #34d399;

            box-shadow:
                0 0 10px #34d399;

        }


        /* =========================
           MAIN CARD
        ========================= */

        .card {

            background:
                rgba(255,255,255,0.94);

            border:
                1px solid
                rgba(255,255,255,0.5);

            border-radius: 30px;

            padding: 40px;

            box-shadow:
                0 30px 80px
                rgba(0,0,0,0.3);

            backdrop-filter:
                blur(20px);

        }


        .card-title {

            display: flex;

            justify-content: space-between;

            align-items: center;

            margin-bottom: 28px;

        }


        .card-title h2 {

            font-size: 23px;

            color: #171c2c;

        }


        .card-title span {

            color: #7c3aed;

            font-size: 14px;

            font-weight: 600;

        }


        /* =========================
           INPUT GRID
        ========================= */

        .input-grid {

            display: grid;

            grid-template-columns:
                repeat(2, 1fr);

            gap: 20px;

        }


        .input-card {

            padding: 20px;

            border-radius: 20px;

            background: #f8f9ff;

            border:
                1px solid #e7e9f5;

            transition: 0.25s;

        }


        .input-card:hover {

            transform: translateY(-2px);

            border-color: #b9b0ff;

            box-shadow:
                0 10px 25px
                rgba(76,29,149,0.08);

        }


        .input-top {

            display: flex;

            justify-content: space-between;

            align-items: center;

            margin-bottom: 12px;

        }


        .input-label {

            font-size: 14px;

            font-weight: 700;

            color: #374151;

        }


        .input-number {

            width: 30px;

            height: 30px;

            display: flex;

            justify-content: center;

            align-items: center;

            border-radius: 10px;

            background: #ede9fe;

            color: #6d28d9;

            font-size: 13px;

            font-weight: 800;

        }


        input {

            width: 100%;

            border: 2px solid #e5e7eb;

            border-radius: 13px;

            padding: 14px 15px;

            font-size: 17px;

            font-weight: 600;

            color: #111827;

            outline: none;

            background: white;

            transition: 0.2s;

        }


        input:focus {

            border-color: #7c3aed;

            box-shadow:
                0 0 0 4px
                rgba(124,58,237,0.10);

        }


        .description {

            margin-top: 8px;

            color: #8b91a1;

            font-size: 12px;

        }


        /* =========================
           EXAMPLES
        ========================= */

        .examples {

            margin-top: 28px;

        }


        .examples-title {

            font-size: 13px;

            color: #6b7280;

            font-weight: 600;

            margin-bottom: 12px;

        }


        .example-buttons {

            display: flex;

            flex-wrap: wrap;

            gap: 10px;

        }


        .example-btn {

            width: auto;

            padding: 9px 14px;

            font-size: 13px;

            font-weight: 600;

            border-radius: 10px;

            background: #f3f4f6;

            color: #4b5563;

            border: 1px solid #e5e7eb;

            cursor: pointer;

            transition: 0.2s;

        }


        .example-btn:hover {

            background: #ede9fe;

            color: #6d28d9;

            border-color: #c4b5fd;

            transform: translateY(-1px);

        }


        /* =========================
           PREDICT BUTTON
        ========================= */

        .predict-btn {

            margin-top: 28px;

            width: 100%;

            border: none;

            border-radius: 16px;

            padding: 17px;

            font-size: 17px;

            font-weight: 800;

            color: white;

            cursor: pointer;

            background:
                linear-gradient(
                    135deg,
                    #7c3aed,
                    #4f46e5
                );

            box-shadow:
                0 12px 25px
                rgba(79,70,229,0.3);

            transition: 0.25s;

        }


        .predict-btn:hover {

            transform: translateY(-2px);

            box-shadow:
                0 16px 30px
                rgba(79,70,229,0.4);

        }


        .predict-btn:active {

            transform: scale(0.99);

        }


        .predict-btn.loading {

            opacity: 0.75;

            cursor: wait;

        }


        /* =========================
           RESULT
        ========================= */

        .result {

            display: none;

            margin-top: 28px;

            padding: 28px;

            border-radius: 22px;

            text-align: center;

            background:
                linear-gradient(
                    135deg,
                    #f5f3ff,
                    #eef2ff
                );

            border:
                1px solid #ddd6fe;

            animation:
                resultIn 0.35s ease;

        }


        @keyframes resultIn {

            from {

                opacity: 0;

                transform:
                    translateY(10px);

            }

            to {

                opacity: 1;

                transform:
                    translateY(0);

            }

        }


        .result-icon {

            font-size: 45px;

            margin-bottom: 8px;

        }


        .result-small {

            color: #7c8494;

            font-size: 13px;

            font-weight: 600;

            text-transform: uppercase;

            letter-spacing: 1px;

        }


        .result-name {

            font-size: 34px;

            font-weight: 900;

            margin: 6px 0;

            color: #5b21b6;

        }


        .result-id {

            color: #6b7280;

            font-size: 13px;

        }


        /* =========================
           ERROR
        ========================= */

        .error {

            display: none;

            margin-top: 20px;

            padding: 14px;

            border-radius: 13px;

            background: #fef2f2;

            border: 1px solid #fecaca;

            color: #b91c1c;

            text-align: center;

            font-size: 14px;

        }


        /* =========================
           FOOTER
        ========================= */

        .footer {

            text-align: center;

            color:
                rgba(255,255,255,0.65);

            margin-top: 25px;

            font-size: 13px;

        }


        .footer-links {

            margin-top: 10px;

        }


        .footer a {

            color: white;

            text-decoration: none;

            margin: 0 8px;

            font-weight: 600;

        }


        .footer a:hover {

            text-decoration: underline;

        }


        /* =========================
           RESPONSIVE
        ========================= */

        @media (max-width: 700px) {

            body {

                padding: 25px 15px;

            }

            .header h1 {

                font-size: 32px;

            }

            .card {

                padding: 25px;

                border-radius: 24px;

            }

            .input-grid {

                grid-template-columns: 1fr;

            }

            .card-title {

                display: block;

            }

            .card-title span {

                display: block;

                margin-top: 5px;

            }

        }

    </style>

</head>


<body>


<div class="page">


    <!-- HEADER -->

    <div class="header">

        <div class="logo">
            🌸
        </div>

        <h1>
            Iris Classification
        </h1>

        <p>
            Dự đoán loài hoa Iris bằng trí tuệ nhân tạo
        </p>

        <div class="badge">

            <span class="status-dot"></span>

            SVM Model • API Online

        </div>

    </div>



    <!-- MAIN CARD -->

    <div class="card">


        <div class="card-title">

            <h2>
                🌿 Nhập thông số hoa
            </h2>

            <span>
                Đơn vị: centimet (cm)
            </span>

        </div>



        <!-- INPUTS -->

        <div class="input-grid">


            <!-- SEPAL LENGTH -->

            <div class="input-card">

                <div class="input-top">

                    <span class="input-label">
                        Sepal Length
                    </span>

                    <span class="input-number">
                        01
                    </span>

                </div>

                <input
                    type="number"
                    id="sepal_length"
                    step="0.1"
                    min="0"
                    value="5.1"
                >

                <div class="description">
                    Chiều dài đài hoa
                </div>

            </div>



            <!-- SEPAL WIDTH -->

            <div class="input-card">

                <div class="input-top">

                    <span class="input-label">
                        Sepal Width
                    </span>

                    <span class="input-number">
                        02
                    </span>

                </div>

                <input
                    type="number"
                    id="sepal_width"
                    step="0.1"
                    min="0"
                    value="3.5"
                >

                <div class="description">
                    Chiều rộng đài hoa
                </div>

            </div>



            <!-- PETAL LENGTH -->

            <div class="input-card">

                <div class="input-top">

                    <span class="input-label">
                        Petal Length
                    </span>

                    <span class="input-number">
                        03
                    </span>

                </div>

                <input
                    type="number"
                    id="petal_length"
                    step="0.1"
                    min="0"
                    value="1.4"
                >

                <div class="description">
                    Chiều dài cánh hoa
                </div>

            </div>



            <!-- PETAL WIDTH -->

            <div class="input-card">

                <div class="input-top">

                    <span class="input-label">
                        Petal Width
                    </span>

                    <span class="input-number">
                        04
                    </span>

                </div>

                <input
                    type="number"
                    id="petal_width"
                    step="0.1"
                    min="0"
                    value="0.2"
                >

                <div class="description">
                    Chiều rộng cánh hoa
                </div>

            </div>


        </div>



        <!-- EXAMPLES -->

        <div class="examples">

            <div class="examples-title">
                ⚡ Thử nhanh một mẫu:
            </div>


            <div class="example-buttons">

                <button
                    class="example-btn"
                    onclick="setExample(
                        5.1, 3.5, 1.4, 0.2
                    )"
                >
                    🌸 Setosa
                </button>


                <button
                    class="example-btn"
                    onclick="setExample(
                        6.0, 2.9, 4.5, 1.5
                    )"
                >
                    🌺 Versicolor
                </button>


                <button
                    class="example-btn"
                    onclick="setExample(
                        6.5, 3.0, 5.2, 2.0
                    )"
                >
                    🌷 Virginica
                </button>

            </div>

        </div>



        <!-- BUTTON -->

        <button
            class="predict-btn"
            id="predictButton"
            onclick="predict()"
        >

            🔮 Dự đoán loài hoa

        </button>



        <!-- RESULT -->

        <div
            class="result"
            id="result"
        >

            <div
                class="result-icon"
                id="resultIcon"
            >
                🌸
            </div>

            <div class="result-small">
                Kết quả dự đoán
            </div>

            <div
                class="result-name"
                id="prediction"
            >
                Setosa
            </div>

            <div
                class="result-id"
                id="class_id"
            >
                Class ID: 0
            </div>

        </div>



        <!-- ERROR -->

        <div
            class="error"
            id="error"
        >
        </div>


    </div>



    <!-- FOOTER -->

    <div class="footer">

        <div>
            Machine Learning • Support Vector Machine • FastAPI
        </div>

        <div class="footer-links">

            <a
                href="/docs"
                target="_blank"
            >
                📘 API Documentation
            </a>

            <a
                href="/health"
                target="_blank"
            >
                💚 Health
            </a>

        </div>

    </div>


</div>



<script>


    // ========================================================
    // CHỌN MẪU CÓ SẴN
    // ========================================================

    function setExample(
        sepalLength,
        sepalWidth,
        petalLength,
        petalWidth
    ) {

        document.getElementById(
            "sepal_length"
        ).value = sepalLength;


        document.getElementById(
            "sepal_width"
        ).value = sepalWidth;


        document.getElementById(
            "petal_length"
        ).value = petalLength;


        document.getElementById(
            "petal_width"
        ).value = petalWidth;


        document.getElementById(
            "result"
        ).style.display = "none";


        document.getElementById(
            "error"
        ).style.display = "none";

    }



    // ========================================================
    // DỰ ĐOÁN
    // ========================================================

    async function predict() {


        const button =
            document.getElementById(
                "predictButton"
            );


        const result =
            document.getElementById(
                "result"
            );


        const error =
            document.getElementById(
                "error"
            );


        // Ẩn kết quả cũ

        result.style.display = "none";

        error.style.display = "none";


        // Lấy dữ liệu

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


        // Kiểm tra dữ liệu

        if (
            Object.values(data)
            .some(value => isNaN(value))
        ) {

            error.innerText =
                "⚠️ Vui lòng nhập đầy đủ 4 thông số.";

            error.style.display =
                "block";

            return;

        }


        // Loading

        button.classList.add("loading");

        button.innerText =
            "⏳ Đang phân tích...";


        try {


            // Gọi API

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
                    "API không phản hồi."
                );

            }


            const resultData =
                await response.json();


            // Hiển thị kết quả

            const prediction =
                resultData.prediction;


            document.getElementById(
                "prediction"
            ).innerText =
                prediction;


            document.getElementById(
                "class_id"
            ).innerText =
                "Class ID: " +
                resultData.class_id;


            // Đổi icon theo loài

            const icons = {

                "Setosa": "🌸",

                "Versicolor": "🌺",

                "Virginica": "🌷"

            };


            document.getElementById(
                "resultIcon"
            ).innerText =
                icons[prediction] || "🌸";


            result.style.display =
                "block";


        }

        catch (err) {

            error.innerText =
                "❌ Có lỗi xảy ra: " +
                err.message;

            error.style.display =
                "block";

        }


        finally {

            button.classList.remove(
                "loading"
            );

            button.innerText =
                "🔮 Dự đoán loài hoa";

        }

    }

</script>


</body>

</html>
    """


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# API DỰ ĐOÁN
# ============================================================

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
