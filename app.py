from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("svm_model.pkl")

app = FastAPI(
    title="Iris AI Classification",
    description="Iris flower classification using Support Vector Machine",
    version="2.0.0"
)


# =========================
# INPUT MODEL
# =========================
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


# =========================
# HOME PAGE
# =========================
@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html lang="vi">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Iris AI Classification</title>

<style>

/* =====================================================
   GLOBAL
===================================================== */

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

:root{
    --bg:#070b17;
    --panel:rgba(17,24,39,.72);
    --panel2:rgba(255,255,255,.055);
    --border:rgba(255,255,255,.10);

    --text:#f8fafc;
    --muted:#94a3b8;

    --primary:#8b5cf6;
    --secondary:#06b6d4;
    --success:#22c55e;

    --shadow:
        0 25px 80px rgba(0,0,0,.45);
}

body{

    min-height:100vh;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color:var(--text);

    background:

        radial-gradient(
            circle at 15% 15%,
            rgba(139,92,246,.22),
            transparent 30%
        ),

        radial-gradient(
            circle at 85% 80%,
            rgba(6,182,212,.16),
            transparent 30%
        ),

        var(--bg);

    overflow-x:hidden;
}


/* =====================================================
   BACKGROUND PARTICLES
===================================================== */

.background{
    position:fixed;
    inset:0;
    pointer-events:none;
    overflow:hidden;
    z-index:0;
}

.orb{
    position:absolute;
    border-radius:50%;
    filter:blur(80px);
    opacity:.35;
    animation:float 10s ease-in-out infinite;
}

.orb.one{
    width:300px;
    height:300px;
    background:#7c3aed;
    top:-100px;
    left:-100px;
}

.orb.two{
    width:260px;
    height:260px;
    background:#0891b2;
    right:-80px;
    bottom:-80px;
    animation-delay:2s;
}

@keyframes float{

    0%,100%{
        transform:translate(0,0);
    }

    50%{
        transform:translate(25px,-20px);
    }
}


/* =====================================================
   MAIN
===================================================== */

.container{

    position:relative;
    z-index:2;

    width:min(1250px,94%);

    margin:auto;

    padding:30px 0 50px;
}


/* =====================================================
   NAVBAR
===================================================== */

.navbar{

    display:flex;
    justify-content:space-between;
    align-items:center;

    padding:15px 20px;

    border:1px solid var(--border);

    background:rgba(15,23,42,.60);

    backdrop-filter:blur(18px);

    border-radius:18px;

    box-shadow:
        0 10px 40px rgba(0,0,0,.20);
}

.logo{

    display:flex;
    align-items:center;
    gap:12px;

    font-weight:800;
    font-size:17px;
}

.logo-icon{

    width:40px;
    height:40px;

    display:grid;
    place-items:center;

    border-radius:12px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--secondary)
        );

    box-shadow:
        0 8px 25px rgba(139,92,246,.35);
}

.nav-status{

    display:flex;
    align-items:center;
    gap:8px;

    color:var(--muted);

    font-size:13px;
}

.status-dot{

    width:8px;
    height:8px;

    border-radius:50%;

    background:var(--success);

    box-shadow:
        0 0 12px var(--success);

    animation:pulse 2s infinite;
}

@keyframes pulse{

    0%,100%{
        opacity:1;
    }

    50%{
        opacity:.45;
    }
}


/* =====================================================
   HERO
===================================================== */

.hero{

    text-align:center;

    padding:65px 20px 45px;

    animation:fadeUp .8s ease;
}

.badge{

    display:inline-flex;

    padding:7px 14px;

    border-radius:100px;

    border:1px solid rgba(139,92,246,.30);

    background:rgba(139,92,246,.10);

    color:#c4b5fd;

    font-size:12px;

    font-weight:700;

    letter-spacing:.08em;

    text-transform:uppercase;
}

.hero h1{

    margin-top:18px;

    font-size:
        clamp(38px,6vw,68px);

    line-height:1;

    font-weight:900;

    letter-spacing:-.05em;

    background:
        linear-gradient(
            110deg,
            #fff,
            #c4b5fd,
            #67e8f9
        );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero p{

    max-width:650px;

    margin:20px auto 0;

    color:var(--muted);

    line-height:1.8;

    font-size:15px;
}


/* =====================================================
   DASHBOARD
===================================================== */

.dashboard{

    display:grid;

    grid-template-columns:
        minmax(0,1.05fr)
        minmax(350px,.95fr);

    gap:22px;

    animation:fadeUp 1s ease;
}


/* =====================================================
   CARD
===================================================== */

.card{

    border:1px solid var(--border);

    background:var(--panel);

    backdrop-filter:blur(22px);

    border-radius:26px;

    box-shadow:var(--shadow);

    overflow:hidden;
}

.card-header{

    padding:25px 26px 10px;
}

.card-title{

    font-size:18px;

    font-weight:800;
}

.card-subtitle{

    margin-top:6px;

    color:var(--muted);

    font-size:13px;
}


/* =====================================================
   SLIDERS
===================================================== */

.controls{

    padding:10px 26px 26px;
}

.slider-item{

    margin-top:25px;
}

.slider-header{

    display:flex;

    justify-content:space-between;

    align-items:center;

    margin-bottom:12px;
}

.slider-name{

    font-size:14px;

    font-weight:650;
}

.slider-value{

    min-width:67px;

    text-align:center;

    padding:7px 10px;

    border-radius:9px;

    background:
        rgba(139,92,246,.12);

    border:1px solid
        rgba(139,92,246,.25);

    color:#c4b5fd;

    font-weight:800;

    font-variant-numeric:tabular-nums;

    transition:
        transform .2s ease,
        background .2s ease;
}

.slider-value.changed{

    transform:scale(1.10);

    background:
        rgba(139,92,246,.22);
}


/* RANGE */

input[type="range"]{

    appearance:none;

    width:100%;

    height:6px;

    border-radius:20px;

    outline:none;

    cursor:pointer;

    background:

        linear-gradient(
            90deg,
            var(--primary) var(--progress),
            rgba(255,255,255,.10) var(--progress)
        );

    transition:
        background .2s ease;
}

input[type="range"]::-webkit-slider-thumb{

    appearance:none;

    width:21px;
    height:21px;

    border-radius:50%;

    background:#fff;

    border:4px solid var(--primary);

    box-shadow:
        0 0 0 4px rgba(139,92,246,.12),
        0 5px 15px rgba(0,0,0,.35);

    transition:
        transform .2s ease,
        box-shadow .2s ease;
}

input[type="range"]::-webkit-slider-thumb:hover{

    transform:scale(1.18);

    box-shadow:
        0 0 0 7px rgba(139,92,246,.13),
        0 5px 20px rgba(0,0,0,.4);
}

input[type="range"]::-moz-range-thumb{

    width:17px;
    height:17px;

    border-radius:50%;

    background:#fff;

    border:4px solid var(--primary);
}


/* =====================================================
   PROFILE
===================================================== */

.profile{

    margin-top:28px;

    padding:16px;

    border-radius:17px;

    background:
        rgba(255,255,255,.035);

    border:1px solid var(--border);
}

.profile-header{

    display:flex;

    justify-content:space-between;

    margin-bottom:12px;

    font-size:12px;

    color:var(--muted);
}

.profile-bar{

    height:8px;

    border-radius:20px;

    background:rgba(255,255,255,.08);

    overflow:hidden;
}

.profile-fill{

    width:35%;

    height:100%;

    border-radius:20px;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--secondary)
        );

    transition:
        width .6s cubic-bezier(.22,1,.36,1);
}


/* =====================================================
   PRESETS
===================================================== */

.presets-title{

    margin-top:24px;

    font-size:12px;

    color:var(--muted);
}

.presets{

    display:grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:9px;

    margin-top:10px;
}

.preset{

    padding:13px 7px;

    border-radius:13px;

    border:1px solid var(--border);

    background:
        rgba(255,255,255,.035);

    color:white;

    cursor:pointer;

    transition:
        transform .2s ease,
        border .2s ease,
        background .2s ease;
}

.preset:hover{

    transform:translateY(-3px);

    border-color:
        rgba(139,92,246,.45);

    background:
        rgba(139,92,246,.10);
}

.preset-icon{

    font-size:23px;

    margin-bottom:6px;
}

.preset-name{

    font-size:11px;

    font-weight:700;
}

.preset-desc{

    display:block;

    margin-top:3px;

    color:var(--muted);

    font-size:9px;
}


/* =====================================================
   BUTTONS
===================================================== */

.actions{

    display:grid;

    grid-template-columns:1fr auto;

    gap:10px;

    margin-top:20px;
}

.predict{

    min-height:50px;

    border:none;

    border-radius:13px;

    background:
        linear-gradient(
            110deg,
            #7c3aed,
            #4f46e5,
            #0891b2
        );

    color:white;

    font-weight:800;

    cursor:pointer;

    font-size:14px;

    box-shadow:
        0 10px 25px rgba(79,70,229,.30);

    transition:
        transform .2s ease,
        box-shadow .2s ease;
}

.predict:hover{

    transform:translateY(-2px);

    box-shadow:
        0 15px 35px rgba(79,70,229,.40);
}

.predict:active{

    transform:scale(.98);
}

.reset{

    width:50px;

    border:1px solid var(--border);

    border-radius:13px;

    background:
        rgba(255,255,255,.05);

    color:#cbd5e1;

    cursor:pointer;

    font-size:18px;

    transition:
        transform .2s ease,
        background .2s ease;
}

.reset:hover{

    transform:rotate(-25deg);

    background:
        rgba(255,255,255,.10);
}


/* =====================================================
   RESULT CARD
===================================================== */

.result-card{

    min-height:100%;

    display:flex;

    flex-direction:column;

    align-items:center;

    justify-content:center;

    text-align:center;

    padding:30px;
}

.result-label{

    color:var(--muted);

    font-size:11px;

    letter-spacing:.12em;

    text-transform:uppercase;

    font-weight:800;
}


/* FLOWER */

.flower-container{

    width:min(100%,390px);

    aspect-ratio:1.18;

    margin:20px 0;

    border-radius:23px;

    overflow:hidden;

    position:relative;

    background:#111827;

    border:1px solid var(--border);

    box-shadow:
        0 20px 60px rgba(0,0,0,.40);
}

.flower-container::after{

    content:"";

    position:absolute;

    inset:0;

    background:
        linear-gradient(
            to top,
            rgba(0,0,0,.35),
            transparent 45%
        );

    pointer-events:none;
}

.flower-container img{

    width:100%;
    height:100%;

    object-fit:cover;

    display:block;

    transition:
        transform .7s cubic-bezier(.22,1,.36,1),
        opacity .4s ease;
}

.flower-container img.animate{

    animation:
        flowerReveal .75s
        cubic-bezier(.22,1,.36,1);
}

@keyframes flowerReveal{

    0%{

        opacity:0;

        transform:
            scale(1.12)
            rotate(1deg);
    }

    100%{

        opacity:1;

        transform:
            scale(1)
            rotate(0);
    }
}


/* RESULT */

.result-name{

    font-size:32px;

    font-weight:900;

    letter-spacing:-.03em;

    background:
        linear-gradient(
            90deg,
            #fff,
            #c4b5fd,
            #67e8f9
        );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

.result-description{

    max-width:390px;

    margin-top:8px;

    color:var(--muted);

    line-height:1.65;

    font-size:13px;
}


/* =====================================================
   RESULT DATA
===================================================== */

.data-grid{

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    width:100%;

    gap:7px;

    margin-top:23px;
}

.data-box{

    padding:10px 5px;

    border-radius:11px;

    background:
        rgba(255,255,255,.04);

    border:1px solid var(--border);
}

.data-value{

    font-size:14px;

    font-weight:800;
}

.data-label{

    margin-top:3px;

    font-size:8px;

    color:var(--muted);

    text-transform:uppercase;
}


/* =====================================================
   API STATUS
===================================================== */

.api-status{

    margin-top:20px;

    padding:9px 13px;

    border-radius:100px;

    display:inline-flex;

    align-items:center;

    gap:7px;

    background:
        rgba(34,197,94,.08);

    border:
        1px solid rgba(34,197,94,.20);

    color:#86efac;

    font-size:11px;
}


/* =====================================================
   ERROR
===================================================== */

.error{

    min-height:18px;

    margin-top:8px;

    color:#fca5a5;

    font-size:11px;
}


/* =====================================================
   ANIMATION
===================================================== */

.fade-up{

    animation:
        fadeUp .7s
        cubic-bezier(.22,1,.36,1);
}

@keyframes fadeUp{

    from{

        opacity:0;

        transform:translateY(18px);
    }

    to{

        opacity:1;

        transform:translateY(0);
    }
}


/* =====================================================
   RESPONSIVE
===================================================== */

@media(max-width:850px){

    .dashboard{

        grid-template-columns:1fr;
    }

    .result-card{

        min-height:500px;
    }
}

@media(max-width:500px){

    .container{

        width:94%;
    }

    .hero{

        padding:
            45px 10px 30px;
    }

    .hero h1{

        font-size:40px;
    }

    .card-header,
    .controls{

        padding-left:19px;
        padding-right:19px;
    }

    .presets{

        gap:5px;
    }

    .result-name{

        font-size:27px;
    }
}


/* =====================================================
   REDUCED MOTION
===================================================== */

@media(prefers-reduced-motion:reduce){

    *,
    *::before,
    *::after{

        animation-duration:.01ms!important;

        transition-duration:.01ms!important;
    }
}

</style>

</head>


<body>


<!-- BACKGROUND -->

<div class="background">

    <div class="orb one"></div>

    <div class="orb two"></div>

</div>


<div class="container">


<!-- =====================================================
     NAVBAR
===================================================== -->

<nav class="navbar">

    <div class="logo">

        <div class="logo-icon">
            🌸
        </div>

        <span>Iris AI</span>

    </div>


    <div class="nav-status">

        <span class="status-dot"></span>

        SVM Model Online

    </div>

</nav>



<!-- =====================================================
     HERO
===================================================== -->

<section class="hero">

    <div class="badge">
        ✦ Artificial Intelligence · Machine Learning
    </div>


    <h1>
        Iris AI
    </h1>


    <p>

        Hệ thống phân loại hoa Iris sử dụng
        <strong>Support Vector Machine</strong>.
        Điều chỉnh các đặc trưng hình thái và
        khám phá kết quả dự đoán theo thời gian thực.

    </p>

</section>



<!-- =====================================================
     DASHBOARD
===================================================== -->

<main class="dashboard">


<!-- =====================================================
     LEFT
===================================================== -->

<section class="card">


    <div class="card-header">

        <div class="card-title">
            Phân tích hình thái
        </div>

        <div class="card-subtitle">

            Điều chỉnh thông số bằng các thanh trượt

        </div>

    </div>



    <div class="controls">


<!-- SEPAL LENGTH -->

<div class="slider-item">

<div class="slider-header">

<span class="slider-name">
    Chiều dài đài hoa
</span>

<span
    class="slider-value"
    id="sepal_length_value">
    5.1 cm
</span>

</div>


<input
    id="sepal_length"
    type="range"
    min="4"
    max="8"
    step="0.1"
    value="5.1"
>


</div>



<!-- SEPAL WIDTH -->

<div class="slider-item">

<div class="slider-header">

<span class="slider-name">
    Chiều rộng đài hoa
</span>

<span
    class="slider-value"
    id="sepal_width_value">
    3.5 cm
</span>

</div>


<input
    id="sepal_width"
    type="range"
    min="2"
    max="4.5"
    step="0.1"
    value="3.5"
>


</div>



<!-- PETAL LENGTH -->

<div class="slider-item">

<div class="slider-header">

<span class="slider-name">
    Chiều dài cánh hoa
</span>

<span
    class="slider-value"
    id="petal_length_value">
    1.4 cm
</span>

</div>


<input
    id="petal_length"
    type="range"
    min="1"
    max="7"
    step="0.1"
    value="1.4"
>


</div>



<!-- PETAL WIDTH -->

<div class="slider-item">

<div class="slider-header">

<span class="slider-name">
    Chiều rộng cánh hoa
</span>

<span
    class="slider-value"
    id="petal_width_value">
    0.2 cm
</span>

</div>


<input
    id="petal_width"
    type="range"
    min="0.1"
    max="2.5"
    step="0.1"
    value="0.2"
>


</div>



<!-- PROFILE -->

<div class="profile">

    <div class="profile-header">

        <span>
            Morphology Profile
        </span>

        <span id="profileText">
            Compact
        </span>

    </div>

    <div class="profile-bar">

        <div
            id="profileFill"
            class="profile-fill">
        </div>

    </div>

</div>



<!-- PRESETS -->

<div class="presets-title">

    Chọn mẫu nhanh

</div>


<div class="presets">


<button
    type="button"
    class="preset"
    data-type="setosa">

    <div class="preset-icon">
        🌱
    </div>

    <div class="preset-name">
        Setosa
    </div>

    <span class="preset-desc">
        Nhỏ · ngắn
    </span>

</button>



<button
    type="button"
    class="preset"
    data-type="versicolor">

    <div class="preset-icon">
        🌷
    </div>

    <div class="preset-name">
        Versicolor
    </div>

    <span class="preset-desc">
        Trung bình
    </span>

</button>



<button
    type="button"
    class="preset"
    data-type="virginica">

    <div class="preset-icon">
        🌸
    </div>

    <div class="preset-name">
        Virginica
    </div>

    <span class="preset-desc">
        Lớn · dài
    </span>

</button>


</div>



<!-- ACTIONS -->

<div class="actions">


<button
    type="button"
    id="predictBtn"
    class="predict">

    ✦ &nbsp; Phân loại bằng SVM

</button>


<button
    type="button"
    id="resetBtn"
    class="reset"
    title="Đặt lại">

    ↻

</button>


</div>


<div
    id="error"
    class="error">
</div>


</div>

</section>



<!-- =====================================================
     RIGHT RESULT
===================================================== -->

<section class="card result-card">


<div class="result-label">

    Kết quả dự đoán

</div>


<div class="flower-container">

    <img
        id="flowerImage"
        src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
        alt="Iris Setosa"
    >

</div>


<div
    id="resultName"
    class="result-name">

    Iris Setosa

</div>


<div
    id="resultDescription"
    class="result-description">

    Hoa Iris Setosa thường có
    cánh hoa ngắn và hẹp,
    kích thước tổng thể nhỏ hơn
    so với các loài Iris còn lại.

</div>



<!-- DATA -->

<div class="data-grid">


<div class="data-box">

    <div
        id="dataSL"
        class="data-value">
        5.1
    </div>

    <div class="data-label">
        Sepal L
    </div>

</div>


<div class="data-box">

    <div
        id="dataSW"
        class="data-value">
        3.5
    </div>

    <div class="data-label">
        Sepal W
    </div>

</div>


<div class="data-box">

    <div
        id="dataPL"
        class="data-value">
        1.4
    </div>

    <div class="data-label">
        Petal L
    </div>

</div>


<div class="data-box">

    <div
        id="dataPW"
        class="data-value">
        0.2
    </div>

    <div class="data-label">
        Petal W
    </div>

</div>


</div>



<div class="api-status">

    <span class="status-dot"></span>

    <span id="apiStatus">
        SVM prediction ready
    </span>

</div>


</section>


</main>


</div>



<script>

/* =====================================================
   IRIS AI FRONTEND
===================================================== */

const sliders = {

    sepal_length:
        document.getElementById("sepal_length"),

    sepal_width:
        document.getElementById("sepal_width"),

    petal_length:
        document.getElementById("petal_length"),

    petal_width:
        document.getElementById("petal_width")
};


const defaults = {

    sepal_length:5.1,

    sepal_width:3.5,

    petal_length:1.4,

    petal_width:0.2

};


const presets = {

    setosa:{

        sepal_length:5.1,
        sepal_width:3.5,
        petal_length:1.4,
        petal_width:0.2

    },

    versicolor:{

        sepal_length:6.0,
        sepal_width:2.9,
        petal_length:4.5,
        petal_width:1.5

    },

    virginica:{

        sepal_length:6.5,
        sepal_width:3.0,
        petal_length:5.5,
        petal_width:1.8

    }

};


const flowers = {

    0:{

        name:"Iris Setosa",

        description:
            "Hoa Iris Setosa thường có cánh hoa ngắn và hẹp, kích thước tổng thể nhỏ hơn so với các loài Iris còn lại.",

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"

    },

    1:{

        name:"Iris Versicolor",

        description:
            "Iris Versicolor có đặc trưng hình thái ở mức trung gian, với kích thước cánh hoa lớn hơn Setosa.",

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg"

    },

    2:{

        name:"Iris Virginica",

        description:
            "Iris Virginica thường có cánh hoa dài và rộng, với kích thước lớn hơn hai loài Iris còn lại.",

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg"

    }

};


/* =====================================================
   UPDATE SLIDER
===================================================== */

function updateSlider(element){

    const min =
        Number(element.min);

    const max =
        Number(element.max);

    const value =
        Number(element.value);


    const progress =
        ((value-min)/(max-min))*100;


    element.style
        .setProperty(
            "--progress",
            progress + "%"
        );


    const id =
        element.id;


    const valueBox =
        document.getElementById(
            id + "_value"
        );


    if(valueBox){

        valueBox.textContent =
            value.toFixed(1) + " cm";


        valueBox.classList.remove(
            "changed"
        );


        void valueBox.offsetWidth;


        valueBox.classList.add(
            "changed"
        );

    }


    updateData();

    updateProfile();

}


/* =====================================================
   UPDATE DATA
===================================================== */

function updateData(){

    document.getElementById("dataSL")
        .textContent =
        Number(
            sliders.sepal_length.value
        ).toFixed(1);


    document.getElementById("dataSW")
        .textContent =
        Number(
            sliders.sepal_width.value
        ).toFixed(1);


    document.getElementById("dataPL")
        .textContent =
        Number(
            sliders.petal_length.value
        ).toFixed(1);


    document.getElementById("dataPW")
        .textContent =
        Number(
            sliders.petal_width.value
        ).toFixed(1);

}


/* =====================================================
   MORPHOLOGY PROFILE
===================================================== */

function updateProfile(){

    const petalLength =
        Number(
            sliders.petal_length.value
        );

    const petalWidth =
        Number(
            sliders.petal_width.value
        );


    let score =
        ((petalLength-1)/6)*70
        +
        ((petalWidth-.1)/2.4)*30;


    score =
        Math.max(
            0,
            Math.min(100,score)
        );


    document.getElementById(
        "profileFill"
    ).style.width =
        score + "%";


    let text = "Compact";


    if(score > 70){

        text = "Large";

    }

    else if(score > 40){

        text = "Medium";

    }


    document.getElementById(
        "profileText"
    ).textContent =
        text;

}


/* =====================================================
   INIT
===================================================== */

Object.values(sliders).forEach(
    slider => {

        updateSlider(slider);

        slider.addEventListener(
            "input",
            () => {

                updateSlider(slider);

            }
        );

    }
);


/* =====================================================
   PRESET
===================================================== */

document
.querySelectorAll(".preset")
.forEach(button => {

    button.addEventListener(
        "click",
        () => {

            const type =
                button.dataset.type;


            const data =
                presets[type];


            Object.entries(data)
            .forEach(
                ([key,value]) => {

                    sliders[key].value =
                        value;

                    updateSlider(
                        sliders[key]
                    );

                }
            );


            predict();

        }
    );

});


/* =====================================================
   PREDICT
===================================================== */

async function predict(){

    const button =
        document.getElementById(
            "predictBtn"
        );

    const error =
        document.getElementById(
            "error"
        );


    error.textContent = "";


    button.disabled = true;

    button.textContent =
        "⏳  Đang phân tích...";


    const data = {

        sepal_length:
            Number(
                sliders.sepal_length.value
            ),

        sepal_width:
            Number(
                sliders.sepal_width.value
            ),

        petal_length:
            Number(
                sliders.petal_length.value
            ),

        petal_width:
            Number(
                sliders.petal_width.value
            )

    };


    try{

        const response =
            await fetch(
                "/predict",
                {

                    method:"POST",

                    headers:{
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)

                }
            );


        if(!response.ok){

            throw new Error(
                "HTTP " +
                response.status
            );

        }


        const result =
            await response.json();


        const classId =
            Number(
                result.class_id
            );


        const flower =
            flowers[classId];


        if(!flower){

            throw new Error(
                "Class không hợp lệ"
            );

        }


        /* IMAGE */

        const image =
            document.getElementById(
                "flowerImage"
            );


        image.classList.remove(
            "animate"
        );


        void image.offsetWidth;


        image.src =
            flower.image;


        image.alt =
            flower.name;


        image.classList.add(
            "animate"
        );


        /* NAME */

        const name =
            document.getElementById(
                "resultName"
            );


        name.classList.remove(
            "fade-up"
        );


        void name.offsetWidth;


        name.textContent =
            flower.name;


        name.classList.add(
            "fade-up"
        );


        /* DESCRIPTION */

        document.getElementById(
            "resultDescription"
        ).textContent =
            flower.description;


        /* STATUS */

        document.getElementById(
            "apiStatus"
        ).textContent =
            "Prediction completed · Class " +
            classId;


    }

    catch(err){

        error.textContent =
            "Không thể kết nối API: " +
            err.message;

    }

    finally{

        button.disabled =
            false;

        button.textContent =
            "✦   Phân loại bằng SVM";

    }

}


/* =====================================================
   RESET
===================================================== */

document
.getElementById("resetBtn")
.addEventListener(
    "click",
    () => {

        Object.entries(defaults)
        .forEach(
            ([key,value]) => {

                sliders[key].value =
                    value;

                updateSlider(
                    sliders[key]
                );

            }
        );


        predict();

    }
);


/* =====================================================
   PREDICT BUTTON
===================================================== */

document
.getElementById("predictBtn")
.addEventListener(
    "click",
    predict
);

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
        "status": "healthy",
        "model": "SVM"
    }


# =========================
# PREDICT API
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

        "prediction":
            species[prediction]

    }
