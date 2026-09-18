from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("svm_model.pkl")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Iris Classification API",
    description="Iris Flower Classification using SVM",
    version="1.0.0"
)


# ============================================================
# INPUT MODEL
# ============================================================

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ============================================================
# SPECIES
# ============================================================

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


# ============================================================
# IMAGE
# ============================================================

flower_images = {
    "Setosa":
        "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG",

    "Versicolor":
        "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg",

    "Virginica":
        "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg"
}


# ============================================================
# HOME
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

<title>Iris Flower Classification</title>


<style>

/* =========================================================
   RESET
========================================================= */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {

    font-family:
        "Segoe UI",
        Arial,
        sans-serif;

    min-height: 100vh;

    color: #172554;

    background:
        linear-gradient(
            135deg,
            #eef2ff 0%,
            #f8fafc 45%,
            #e0f2fe 100%
        );

}


/* =========================================================
   NAVBAR
========================================================= */

.navbar {

    height: 70px;

    background:
        linear-gradient(
            100deg,
            #172554,
            #312e81,
            #1e3a8a
        );

    display: flex;

    align-items: center;

    padding: 0 6%;

    color: white;

    box-shadow:
        0 5px 25px
        rgba(15,23,42,0.18);

}


.logo {

    display: flex;

    align-items: center;

    gap: 12px;

    font-size: 22px;

    font-weight: 800;

    margin-right: 60px;

}


.logo-flower {

    width: 42px;

    height: 42px;

    border-radius: 12px;

    display: flex;

    align-items: center;

    justify-content: center;

    background:
        rgba(255,255,255,0.15);

    font-size: 25px;

}


.nav-links {

    display: flex;

    align-items: center;

    gap: 12px;

}


.nav-link {

    color:
        rgba(255,255,255,0.85);

    text-decoration: none;

    padding: 10px 18px;

    border-radius: 25px;

    font-size: 14px;

    transition: 0.2s;

}


.nav-link:hover {

    background:
        rgba(255,255,255,0.12);

    color: white;

}


.nav-link.active {

    background:
        rgba(255,255,255,0.16);

    color: white;

}


.api-status {

    margin-left: auto;

    display: flex;

    align-items: center;

    gap: 8px;

    padding: 8px 15px;

    border-radius: 25px;

    background:
        rgba(52,211,153,0.15);

    color: #d1fae5;

    font-size: 13px;

    font-weight: 600;

}


.status-dot {

    width: 9px;

    height: 9px;

    background: #34d399;

    border-radius: 50%;

    box-shadow:
        0 0 10px #34d399;

}


/* =========================================================
   HERO
========================================================= */

.hero {

    position: relative;

    padding: 45px 20px 30px;

    text-align: center;

    overflow: hidden;

}


.hero::before {

    content: "";

    position: absolute;

    inset: 0;

    background-image:
        url("https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg");

    background-size: cover;

    background-position: center;

    opacity: 0.09;

    filter: blur(5px);

    z-index: -2;

}


.hero::after {

    content: "";

    position: absolute;

    inset: 0;

    background:
        linear-gradient(
            180deg,
            rgba(238,242,255,0.75),
            rgba(248,250,252,0.95)
        );

    z-index: -1;

}


.hero-inner {

    max-width: 1050px;

    margin: auto;

    display: flex;

    align-items: center;

    justify-content: center;

    gap: 25px;

}


.hero-flower {

    width: 150px;

    height: 150px;

    object-fit: cover;

    border-radius: 35px;

    box-shadow:
        0 15px 40px
        rgba(49,46,129,0.22);

}


.hero-text {

    text-align: left;

}


.hero h1 {

    font-size: 47px;

    font-weight: 850;

    color: #0f172a;

    letter-spacing: -1.5px;

}


.hero h1 span {

    color: #4f46e5;

}


.hero-subtitle {

    margin-top: 7px;

    color: #475569;

    font-size: 19px;

}


.badges {

    display: flex;

    gap: 10px;

    margin-top: 18px;

}


.badge {

    padding: 8px 14px;

    border-radius: 20px;

    background: white;

    border:
        1px solid #e2e8f0;

    color: #475569;

    font-size: 13px;

    box-shadow:
        0 5px 15px
        rgba(15,23,42,0.06);

}


/* =========================================================
   MAIN
========================================================= */

.main {

    width: 92%;

    max-width: 1250px;

    margin: 10px auto 30px;

}


.dashboard {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 20px;

}


/* =========================================================
   CARD
========================================================= */

.card {

    background:
        rgba(255,255,255,0.92);

    border:
        1px solid rgba(255,255,255,0.8);

    border-radius: 24px;

    padding: 27px;

    box-shadow:
        0 15px 45px
        rgba(30,41,59,0.10);

}


.card-header {

    display: flex;

    align-items: center;

    gap: 13px;

    margin-bottom: 22px;

}


.card-icon {

    width: 43px;

    height: 43px;

    border-radius: 13px;

    background:
        #eef2ff;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 22px;

}


.card-header h2 {

    font-size: 21px;

    color: #172554;

}


.card-header p {

    color: #64748b;

    font-size: 13px;

    margin-top: 3px;

}


/* =========================================================
   INPUTS
========================================================= */

.input-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 14px;

}


.input-box {

    background:
        linear-gradient(
            135deg,
            #f8faff,
            #f1f5ff
        );

    border:
        1px solid #dbe3ff;

    border-radius: 15px;

    padding: 15px;

}


.input-label {

    display: flex;

    align-items: center;

    gap: 7px;

    font-size: 14px;

    font-weight: 700;

    color: #1e3a8a;

    margin-bottom: 9px;

}


.input-box input {

    width: 100%;

    padding: 12px;

    border-radius: 10px;

    border:
        1px solid #d5ddf5;

    background: white;

    color: #172554;

    font-size: 16px;

    font-weight: 600;

    outline: none;

}


.input-box input:focus {

    border-color: #6366f1;

    box-shadow:
        0 0 0 3px
        rgba(99,102,241,0.10);

}


.input-desc {

    color: #64748b;

    font-size: 11px;

    margin-top: 6px;

}


/* =========================================================
   QUICK SAMPLE
========================================================= */

.quick-title {

    margin: 22px 0 11px;

    color: #475569;

    font-size: 13px;

    font-weight: 700;

}


.quick-grid {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 9px;

}


.quick {

    border:
        1px solid #dbe3ff;

    background: white;

    border-radius: 12px;

    padding: 8px;

    cursor: pointer;

    transition: 0.2s;

}


.quick:hover {

    transform:
        translateY(-2px);

    border-color:
        #818cf8;

    box-shadow:
        0 7px 18px
        rgba(79,70,229,0.12);

}


.quick img {

    width: 100%;

    height: 62px;

    object-fit: cover;

    border-radius: 8px;

}


.quick-name {

    font-size: 13px;

    font-weight: 700;

    margin-top: 6px;

    color: #1e293b;

}


.quick-value {

    font-size: 10px;

    color: #64748b;

    margin-top: 3px;

}


/* =========================================================
   BUTTON
========================================================= */

.predict-btn {

    margin-top: 20px;

    width: 100%;

    padding: 15px;

    border: none;

    border-radius: 13px;

    cursor: pointer;

    color: white;

    font-size: 16px;

    font-weight: 800;

    background:
        linear-gradient(
            100deg,
            #7c3aed,
            #4f46e5,
            #3b82f6
        );

    box-shadow:
        0 10px 25px
        rgba(79,70,229,0.28);

    transition: 0.2s;

}


.predict-btn:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 15px 30px
        rgba(79,70,229,0.35);

}


/* =========================================================
   RESULT
========================================================= */

.result-card {

    min-height: 100%;

}


.result-content {

    background:
        linear-gradient(
            135deg,
            #f0fdf9,
            #eff6ff
        );

    border:
        1px solid #d8f3ea;

    border-radius: 18px;

    padding: 16px;

    display: grid;

    grid-template-columns:
        1.2fr 1fr;

    gap: 18px;

    align-items: center;

}


.result-image {

    width: 100%;

    height: 270px;

    object-fit: cover;

    border-radius: 15px;

    box-shadow:
        0 10px 25px
        rgba(15,23,42,0.12);

}


.result-info {

    padding: 5px;

}


.result-label {

    display: inline-block;

    padding: 7px 12px;

    border-radius: 20px;

    background: #ecfdf5;

    color: #047857;

    font-size: 12px;

    font-weight: 700;

    margin-bottom: 13px;

}


.result-name {

    font-size: 35px;

    font-weight: 850;

    color: #0f766e;

    margin-bottom: 5px;

}


.class-id {

    color: #059669;

    font-size: 18px;

    font-weight: 700;

    margin-bottom: 16px;

}


.description-result {

    color: #475569;

    line-height: 1.65;

    font-size: 13px;

}


.result-species {

    margin-top: 18px;

    padding: 11px;

    background: white;

    border:
        1px solid #dbeafe;

    border-radius: 11px;

    color: #334155;

    font-size: 13px;

}


.result-species strong {

    color: #1e3a8a;

}


/* =========================================================
   FLOWER SELECT
========================================================= */

.flower-list {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 10px;

    margin-top: 15px;

}


.flower-item {

    border:
        1px solid #dbe3ff;

    background: white;

    border-radius: 12px;

    overflow: hidden;

    cursor: pointer;

    transition: 0.2s;

}


.flower-item:hover {

    transform:
        translateY(-2px);

    border-color:
        #818cf8;

}


.flower-item img {

    width: 100%;

    height: 80px;

    object-fit: cover;

}


.flower-item div {

    padding: 8px;

    text-align: center;

}


.flower-item strong {

    font-size: 13px;

    color: #1e293b;

}


.flower-item small {

    display: block;

    margin-top: 3px;

    color: #64748b;

    font-size: 10px;

}


/* =========================================================
   FOOTER INFORMATION
========================================================= */

.info-bar {

    margin-top: 18px;

    background:
        rgba(255,255,255,0.88);

    border-radius: 20px;

    padding: 18px 25px;

    display: grid;

    grid-template-columns:
        1fr 1fr 1fr 1fr;

    gap: 20px;

    box-shadow:
        0 10px 35px
        rgba(30,41,59,0.08);

}


.info-item {

    border-right:
        1px solid #e2e8f0;

}


.info-item:last-child {

    border: none;

}


.info-title {

    font-size: 12px;

    color: #64748b;

    margin-bottom: 4px;

}


.info-value {

    color: #1e3a8a;

    font-size: 14px;

    font-weight: 700;

}


.footer {

    text-align: center;

    color: #64748b;

    font-size: 12px;

    padding: 22px;

}


.footer a {

    color: #4f46e5;

    text-decoration: none;

    font-weight: 600;

}


/* =========================================================
   ERROR
========================================================= */

.error {

    display: none;

    margin-top: 12px;

    padding: 12px;

    border-radius: 10px;

    background: #fef2f2;

    color: #b91c1c;

    font-size: 13px;

}


/* =========================================================
   RESPONSIVE
========================================================= */

@media (max-width: 850px) {

    .dashboard {

        grid-template-columns: 1fr;

    }

    .info-bar {

        grid-template-columns:
            1fr 1fr;

    }

}


@media (max-width: 600px) {

    .navbar {

        padding: 0 15px;

    }

    .logo {

        margin-right: 10px;

    }

    .nav-links {

        display: none;

    }

    .hero-inner {

        flex-direction: column;

    }

    .hero-text {

        text-align: center;

    }

    .hero h1 {

        font-size: 34px;

    }

    .badges {

        justify-content: center;

        flex-wrap: wrap;

    }

    .input-grid {

        grid-template-columns: 1fr;

    }

    .quick-grid {

        grid-template-columns: 1fr;

    }

    .result-content {

        grid-template-columns: 1fr;

    }

    .result-image {

        height: 220px;

    }

    .flower-list {

        grid-template-columns: 1fr 1fr 1fr;

    }

    .info-bar {

        grid-template-columns: 1fr;

    }

    .info-item {

        border-right: none;

        border-bottom:
            1px solid #e2e8f0;

        padding-bottom: 10px;

    }

}

</style>

</head>


<body>


<!-- ======================================================
     NAVBAR
======================================================= -->

<nav class="navbar">

    <div class="logo">

        <div class="logo-flower">
            🌸
        </div>

        Iris Classification

    </div>


    <div class="nav-links">

        <a class="nav-link active"
           href="/">
            🏠 Home
        </a>

        <a class="nav-link"
           href="/docs"
           target="_blank">
            📄 API Docs
        </a>

        <a class="nav-link"
           href="/health"
           target="_blank">
            〽 Health
        </a>

    </div>


    <div class="api-status">

        <span class="status-dot"></span>

        API Online

    </div>

</nav>



<!-- ======================================================
     HERO
======================================================= -->

<section class="hero">

    <div class="hero-inner">

        <img
            class="hero-flower"
            src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg"
            alt="Iris flower"
        >


        <div class="hero-text">

            <h1>
                <span>Iris</span>
                Flower Classification
            </h1>

            <div class="hero-subtitle">

                Dự đoán loài hoa Iris bằng mô hình SVM

            </div>


            <div class="badges">

                <div class="badge">
                    🧠 Machine Learning
                </div>

                <div class="badge">
                    ⚡ SVM
                </div>

                <div class="badge">
                    🔗 FastAPI
                </div>

            </div>

        </div>

    </div>

</section>



<!-- ======================================================
     MAIN
======================================================= -->

<main class="main">


<div class="dashboard">


<!-- ======================================================
     INPUT CARD
======================================================= -->

<div class="card">

    <div class="card-header">

        <div class="card-icon">
            🌿
        </div>

        <div>

            <h2>
                Nhập thông số hoa
            </h2>

            <p>
                Nhập 4 thông số đặc trưng của hoa Iris
                (đơn vị: cm)
            </p>

        </div>

    </div>


    <div class="input-grid">


        <!-- SEPAL LENGTH -->

        <div class="input-box">

            <div class="input-label">
                🍃 Sepal Length
            </div>

            <input
                id="sepal_length"
                type="number"
                step="0.1"
                value="5.1"
            >

            <div class="input-desc">
                Chiều dài đài hoa (cm)
            </div>

        </div>


        <!-- SEPAL WIDTH -->

        <div class="input-box">

            <div class="input-label">
                💧 Sepal Width
            </div>

            <input
                id="sepal_width"
                type="number"
                step="0.1"
                value="3.5"
            >

            <div class="input-desc">
                Chiều rộng đài hoa (cm)
            </div>

        </div>


        <!-- PETAL LENGTH -->

        <div class="input-box">

            <div class="input-label">
                🌼 Petal Length
            </div>

            <input
                id="petal_length"
                type="number"
                step="0.1"
                value="1.4"
            >

            <div class="input-desc">
                Chiều dài cánh hoa (cm)
            </div>

        </div>


        <!-- PETAL WIDTH -->

        <div class="input-box">

            <div class="input-label">
                ✨ Petal Width
            </div>

            <input
                id="petal_width"
                type="number"
                step="0.1"
                value="0.2"
            >

            <div class="input-desc">
                Chiều rộng cánh hoa (cm)
            </div>

        </div>


    </div>


    <!-- QUICK SAMPLES -->

    <div class="quick-title">
        ⚡ Mẫu thử nhanh
    </div>


    <div class="quick-grid">


        <div
            class="quick"
            onclick="setSample(
                5.1,3.5,1.4,0.2
            )"
        >

            <img
                src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
            >

            <div class="quick-name">
                Setosa
            </div>

            <div class="quick-value">
                5.1 / 3.5 / 1.4 / 0.2
            </div>

        </div>


        <div
            class="quick"
            onclick="setSample(
                6.0,2.9,4.5,1.5
            )"
        >

            <img
                src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg"
            >

            <div class="quick-name">
                Versicolor
            </div>

            <div class="quick-value">
                6.0 / 2.9 / 4.5 / 1.5
            </div>

        </div>


        <div
            class="quick"
            onclick="setSample(
                6.5,3.0,5.2,2.0
            )"
        >

            <img
                src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg"
            >

            <div class="quick-name">
                Virginica
            </div>

            <div class="quick-value">
                6.5 / 3.0 / 5.2 / 2.0
            </div>

        </div>


    </div>


    <!-- BUTTON -->

    <button
        class="predict-btn"
        onclick="predict()"
        id="predictButton"
    >
        ✨ Dự đoán loài hoa →
    </button>


    <div
        class="error"
        id="error"
    ></div>


</div>



<!-- ======================================================
     RESULT CARD
======================================================= -->

<div class="card result-card">

    <div class="card-header">

        <div class="card-icon">
            🔮
        </div>

        <div>

            <h2>
                Kết quả dự đoán
            </h2>

            <p>
                Kết quả từ mô hình SVM
            </p>

        </div>

    </div>


    <div
        class="result-content"
        id="resultContent"
    >


        <img
            id="resultImage"
            class="result-image"
            src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
            alt="Iris"
        >


        <div class="result-info">

            <div class="result-label">
                ✓ Mô hình đã dự đoán
            </div>

            <div
                class="result-name"
                id="resultName"
            >
                Setosa
            </div>

            <div
                class="class-id"
                id="classId"
            >
                Class ID: 0
            </div>

            <div
                class="description-result"
                id="resultDescription"
            >

                Iris Setosa là một trong ba lớp
                của bộ dữ liệu Iris.

                Mô hình SVM sử dụng các đặc trưng
                về chiều dài và chiều rộng của đài hoa
                và cánh hoa để phân loại.

            </div>

            <div class="result-species">

                🌸 Loài hoa:

                <strong id="speciesText">
                    Iris Setosa
                </strong>

            </div>

        </div>

    </div>



    <!-- FLOWER LIST -->

    <div class="flower-list">


        <div
            class="flower-item"
            onclick="setSample(
                5.1,3.5,1.4,0.2
            )"
        >

            <img
                src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
            >

            <div>

                <strong>
                    Setosa
                </strong>

                <small>
                    Cánh hoa nhỏ
                </small>

            </div>

        </div>


        <div
            class="flower-item"
            onclick="setSample(
                6.0,2.9,4.5,1.5
            )"
        >

            <img
                src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg"
            >

            <div>

                <strong>
                    Versicolor
                </strong>

                <small>
                    Cánh hoa tím xanh
                </small>

            </div>

        </div>


        <div
            class="flower-item"
            onclick="setSample(
                6.5,3.0,5.2,2.0
            )"
        >

            <img
                src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg"
            >

            <div>

                <strong>
                    Virginica
                </strong>

                <small>
                    Cánh hoa lớn
                </small>

            </div>

        </div>


    </div>

</div>


</div>



<!-- ======================================================
     INFORMATION BAR
======================================================= -->

<div class="info-bar">


    <div class="info-item">

        <div class="info-title">
            🗄️ Dữ liệu
        </div>

        <div class="info-value">
            Iris Dataset · 150 mẫu
        </div>

    </div>


    <div class="info-item">

        <div class="info-title">
            ⚙️ Mô hình
        </div>

        <div class="info-value">
            Support Vector Machine
        </div>

    </div>


    <div class="info-item">

        <div class="info-title">
            ⚡ API Endpoints
        </div>

        <div class="info-value">
            /predict · /health · /docs
        </div>

    </div>


    <div class="info-item">

        <div class="info-title">
            🌱 Công nghệ
        </div>

        <div class="info-value">
            Python · FastAPI · SVM
        </div>

    </div>


</div>


</main>



<!-- ======================================================
     FOOTER
======================================================= -->

<footer class="footer">

    Iris Flower Classification

    ·

    Machine Learning Project

    <br>

    <a href="/docs" target="_blank">
        API Documentation
    </a>

    &nbsp; · &nbsp;

    <a href="/health" target="_blank">
        Health Check
    </a>

</footer>



<!-- ======================================================
     JAVASCRIPT
======================================================= -->

<script>


// ==========================================================
// FLOWER DATA
// ==========================================================

const flowerData = {

    "Setosa": {

        id: 0,

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG",

        description:
            "Iris Setosa là một trong ba lớp của bộ dữ liệu Iris. Loài này thường có kích thước cánh hoa nhỏ hơn so với hai lớp còn lại."

    },

    "Versicolor": {

        id: 1,

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg",

        description:
            "Iris Versicolor là lớp trung gian trong bộ dữ liệu Iris, có các đặc trưng kích thước nằm giữa Setosa và Virginica."

    },

    "Virginica": {

        id: 2,

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg",

        description:
            "Iris Virginica thường có kích thước cánh hoa lớn hơn và là một trong ba lớp được mô hình SVM phân loại."

    }

};


// ==========================================================
// SET SAMPLE
// ==========================================================

function setSample(
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
        "error"
    ).style.display = "none";

}


// ==========================================================
// PREDICT
// ==========================================================

async function predict() {


    const button =
        document.getElementById(
            "predictButton"
        );


    const error =
        document.getElementById(
            "error"
        );


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

    button.innerText =
        "⏳ Đang phân tích...";

    button.disabled = true;


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


        const result =
            await response.json();


        const name =
            result.prediction;


        const info =
            flowerData[name];


        // ==============================================
        // UPDATE RESULT
        // ==============================================

        document.getElementById(
            "resultImage"
        ).src =
            info.image;


        document.getElementById(
            "resultName"
        ).innerText =
            name;


        document.getElementById(
            "classId"
        ).innerText =
            "Class ID: " + info.id;


        document.getElementById(
            "speciesText"
        ).innerText =
            "Iris " + name;


        document.getElementById(
            "resultDescription"
        ).innerText =
            info.description;


        // Cuộn tới kết quả

        document.getElementById(
            "resultContent"
        ).scrollIntoView({

            behavior: "smooth",

            block: "center"

        });


    }

    catch (err) {

        error.innerText =
            "❌ " + err.message;

        error.style.display =
            "block";

    }


    finally {

        button.innerText =
            "✨ Dự đoán loài hoa →";

        button.disabled = false;

    }

}

</script>


</body>

</html>
    """


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICT API
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
