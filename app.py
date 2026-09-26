from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("svm_model.pkl")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Iris AI Quantum Studio",
    description="Iris flower classification using SVM",
    version="4.3.0"
)


# =========================================================
# INPUT MODEL
# =========================================================

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# =========================================================
# CLASS INFORMATION
# =========================================================

species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


species_short = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


species_images = {
    0: "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG",
    1: "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg",
    2: "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg"
}


species_descriptions = {
    0: "Iris Setosa thường có cánh hoa ngắn và hẹp, với kích thước tổng thể nhỏ.",
    1: "Iris Versicolor có các đặc trưng hình thái ở mức trung gian.",
    2: "Iris Virginica thường có cánh hoa dài và rộng hơn."
}


# =========================================================
# HOME
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html lang="vi">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Iris AI Quantum Studio</title>


<style>

/* =========================================================
   THEME
========================================================= */

:root{

    --bg:#050816;

    --panel:#0d1326;

    --panel2:#11182e;

    --border:#222d4d;

    --text:#f8fafc;

    --muted:#8490aa;

    --primary:#8b5cf6;

    --secondary:#06b6d4;

    --success:#22c55e;

    --danger:#ef4444;

    --grid:rgba(255,255,255,.055);

    --shadow:0 25px 70px rgba(0,0,0,.45);

}


body.cyber{

    --bg:#030609;

    --panel:#071316;

    --panel2:#091b20;

    --border:#0c4a52;

    --primary:#00e5ff;

    --secondary:#00ff88;

    --grid:rgba(0,255,200,.07);

}


body.matrix{

    --bg:#020602;

    --panel:#061006;

    --panel2:#091609;

    --border:#164d16;

    --primary:#39ff14;

    --secondary:#00cc66;

    --grid:rgba(57,255,20,.06);

}


/* =========================================================
   GLOBAL
========================================================= */

*{

    box-sizing:border-box;

    margin:0;

    padding:0;

}


body{

    min-height:100vh;

    background:

        radial-gradient(
            circle at 10% 0%,
            color-mix(
                in srgb,
                var(--primary) 28%,
                transparent
            ),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 100%,
            color-mix(
                in srgb,
                var(--secondary) 20%,
                transparent
            ),
            transparent 30%
        ),

        var(--bg);

    color:var(--text);

    font-family:
        Inter,
        "Segoe UI",
        Arial,
        sans-serif;

    transition:
        background .5s ease,
        color .3s ease;

}


button,
input{

    font:inherit;

}


button{

    cursor:pointer;

}


/* =========================================================
   BACKGROUND
========================================================= */

.bg-grid{

    position:fixed;

    inset:0;

    pointer-events:none;

    background-image:

        linear-gradient(
            var(--grid) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            var(--grid) 1px,
            transparent 1px
        );

    background-size:42px 42px;

    mask-image:
        linear-gradient(
            to bottom,
            black,
            transparent
        );

}


.orb{

    position:fixed;

    width:300px;

    height:300px;

    border-radius:50%;

    filter:blur(100px);

    opacity:.16;

    pointer-events:none;

    animation:orbFloat 9s ease-in-out infinite;

}


.orb1{

    background:var(--primary);

    top:-100px;

    left:-80px;

}


.orb2{

    background:var(--secondary);

    right:-100px;

    bottom:-100px;

    animation-delay:2s;

}


@keyframes orbFloat{

    0%,100%{
        transform:translate(0,0);
    }

    50%{
        transform:translate(30px,-25px);
    }

}


/* =========================================================
   CONTAINER
========================================================= */

.container{

    position:relative;

    z-index:2;

    width:min(1120px,94%);

    margin:auto;

    padding:20px 0 40px;

}


/* =========================================================
   NAVBAR
========================================================= */

.navbar{

    min-height:57px;

    display:flex;

    align-items:center;

    justify-content:space-between;

    gap:15px;

    padding:10px 16px;

    border:1px solid var(--border);

    background:
        color-mix(
            in srgb,
            var(--panel) 92%,
            transparent
        );

    border-radius:16px;

    box-shadow:var(--shadow);

    backdrop-filter:blur(20px);

}


.brand{

    display:flex;

    align-items:center;

    gap:10px;

    font-size:14px;

    font-weight:800;

}


.brand-icon{

    width:34px;

    height:34px;

    display:grid;

    place-items:center;

    border-radius:10px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--secondary)
        );

    box-shadow:
        0 0 25px
        color-mix(
            in srgb,
            var(--primary) 45%,
            transparent
        );

}


.version{

    color:var(--primary);

    font-size:11px;

}


.theme-switch{

    display:flex;

    align-items:center;

    gap:3px;

    padding:3px;

    border:1px solid var(--border);

    border-radius:9px;

}


.theme-btn{

    border:0;

    color:var(--muted);

    background:transparent;

    padding:5px 9px;

    border-radius:6px;

    font-size:9px;

    font-weight:700;

}


.theme-btn.active{

    background:var(--primary);

    color:white;

}


.online{

    display:flex;

    align-items:center;

    gap:7px;

    padding:7px 11px;

    border:1px solid
        rgba(34,197,94,.25);

    border-radius:100px;

    color:#86efac;

    font-size:10px;

}


.online-dot{

    width:6px;

    height:6px;

    background:var(--success);

    border-radius:50%;

    box-shadow:
        0 0 10px var(--success);

}


/* =========================================================
   HERO
========================================================= */

.hero{

    text-align:center;

    padding:42px 10px 30px;

}


.hero-tag{

    color:var(--primary);

    font-size:10px;

    letter-spacing:.18em;

    text-transform:uppercase;

    font-weight:800;

}


.hero h1{

    margin-top:10px;

    font-size:
        clamp(32px,5vw,52px);

    line-height:1;

    font-weight:900;

    letter-spacing:-.04em;

}


.hero h1 span{

    color:var(--primary);

}


.hero p{

    margin:13px auto 0;

    max-width:650px;

    color:var(--muted);

    font-size:12px;

    line-height:1.7;

}


/* =========================================================
   DASHBOARD
========================================================= */

.dashboard{

    display:grid;

    grid-template-columns:
        minmax(0,.96fr)
        minmax(0,1.04fr);

    gap:16px;

}


/* =========================================================
   CARD
========================================================= */

.card{

    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--panel) 96%,
                transparent
            ),
            var(--panel2)
        );

    border:1px solid var(--border);

    border-radius:18px;

    box-shadow:var(--shadow);

    overflow:hidden;

    backdrop-filter:blur(20px);

}


.card-header{

    padding:18px 18px 8px;

}


.card-title{

    font-size:14px;

    font-weight:800;

}


.card-sub{

    margin-top:4px;

    color:var(--muted);

    font-size:9px;

}


/* =========================================================
   SLIDERS
========================================================= */

.controls{

    padding:5px 18px 18px;

}


.slider{

    margin-top:17px;

}


.slider-head{

    display:flex;

    justify-content:space-between;

    align-items:center;

    margin-bottom:8px;

}


.slider-name{

    font-size:10px;

    font-weight:700;

}


.value{

    min-width:45px;

    padding:4px 7px;

    text-align:center;

    border-radius:5px;

    border:1px solid var(--border);

    background:rgba(255,255,255,.025);

    color:var(--primary);

    font-size:9px;

    font-weight:800;

    transition:
        transform .15s ease;

}


.value.bump{

    transform:scale(1.12);

}


input[type=range]{

    appearance:none;

    width:100%;

    height:4px;

    border-radius:20px;

    outline:none;

    background:

        linear-gradient(
            90deg,
            var(--primary) var(--progress),
            #202942 var(--progress)
        );

}


input[type=range]::-webkit-slider-thumb{

    appearance:none;

    width:14px;

    height:14px;

    border-radius:50%;

    background:white;

    border:3px solid var(--primary);

    box-shadow:
        0 0 0 3px
        color-mix(
            in srgb,
            var(--primary) 18%,
            transparent
        ),
        0 0 15px
        color-mix(
            in srgb,
            var(--primary) 65%,
            transparent
        );

    transition:
        transform .15s ease;

}


input[type=range]::-webkit-slider-thumb:hover{

    transform:scale(1.3);

}


/* =========================================================
   PROFILE
========================================================= */

.profile{

    margin-top:17px;

    padding:10px;

    border-radius:9px;

    background:rgba(255,255,255,.025);

    border:1px solid var(--border);

}


.profile-head{

    display:flex;

    justify-content:space-between;

    color:var(--muted);

    font-size:8px;

    margin-bottom:6px;

}


.profile-bar{

    height:4px;

    background:#202942;

    border-radius:20px;

    overflow:hidden;

}


.profile-fill{

    height:100%;

    width:20%;

    border-radius:20px;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--secondary)
        );

    transition:
        width .55s
        cubic-bezier(.22,1,.36,1);

}


/* =========================================================
   PRESETS
========================================================= */

.section-label{

    margin-top:15px;

    color:var(--muted);

    font-size:8px;

    text-transform:uppercase;

    letter-spacing:.08em;

}


.presets{

    display:grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:6px;

    margin-top:7px;

}


.preset{

    border:1px solid var(--border);

    background:rgba(255,255,255,.025);

    color:var(--text);

    padding:8px 4px;

    border-radius:8px;

    transition:
        transform .2s ease,
        border-color .2s ease,
        background .2s ease;

}


.preset:hover{

    transform:translateY(-2px);

    border-color:var(--primary);

    background:
        color-mix(
            in srgb,
            var(--primary) 10%,
            transparent
        );

}


.preset-icon{

    font-size:14px;

}


.preset-name{

    display:block;

    margin-top:3px;

    font-size:8px;

    font-weight:800;

}


/* =========================================================
   BUTTONS
========================================================= */

.actions{

    display:grid;

    grid-template-columns:1fr 38px;

    gap:7px;

    margin-top:12px;

}


.predict{

    min-height:37px;

    border:0;

    border-radius:8px;

    color:white;

    font-size:10px;

    font-weight:800;

    background:
        linear-gradient(
            110deg,
            var(--primary),
            var(--secondary)
        );

    box-shadow:
        0 8px 22px
        color-mix(
            in srgb,
            var(--primary) 25%,
            transparent
        );

    transition:
        transform .2s ease,
        filter .2s ease;

}


.predict:hover{

    transform:translateY(-2px);

    filter:brightness(1.08);

}


.predict:disabled{

    opacity:.65;

    cursor:wait;

}


.reset{

    border:1px solid var(--border);

    border-radius:8px;

    background:rgba(255,255,255,.035);

    color:var(--muted);

    font-size:15px;

}


.error{

    min-height:14px;

    margin-top:5px;

    text-align:center;

    color:#fca5a5;

    font-size:8px;

}


/* =========================================================
   RESULT HEADER
========================================================= */

.result{

    padding:18px;

}


.result-head{

    display:flex;

    justify-content:space-between;

    align-items:flex-start;

    margin-bottom:11px;

}


.result-title{

    font-size:14px;

    font-weight:800;

}


.result-sub{

    color:var(--muted);

    margin-top:4px;

    font-size:9px;

}


.latency{

    padding:4px 7px;

    border-radius:5px;

    background:
        color-mix(
            in srgb,
            var(--primary) 12%,
            transparent
        );

    color:var(--primary);

    font-size:8px;

}


/* =========================================================
   FLOWER
========================================================= */

.flower-result{

    display:grid;

    grid-template-columns:
        95px 1fr;

    gap:12px;

    align-items:center;

    padding:9px;

    border:1px solid var(--border);

    border-radius:11px;

    background:rgba(255,255,255,.018);

}


.flower-image{

    width:95px;

    height:65px;

    border-radius:7px;

    object-fit:cover;

    background:#080d19;

    transition:
        transform .5s ease,
        opacity .3s ease;

}


.flower-image.animate{

    animation:
        flowerIn .65s
        cubic-bezier(.22,1,.36,1);

}


@keyframes flowerIn{

    0%{

        opacity:0;

        transform:scale(1.12);

    }

    100%{

        opacity:1;

        transform:scale(1);

    }

}


.result-name{

    font-size:18px;

    font-weight:900;

}


.result-desc{

    margin-top:4px;

    color:var(--muted);

    font-size:9px;

    line-height:1.5;

}


/* =========================================================
   CHARTS
========================================================= */

.charts{

    display:grid;

    grid-template-columns:
        1fr 1fr;

    gap:10px;

    margin-top:10px;

}


.chart{

    border:1px solid var(--border);

    border-radius:11px;

    padding:10px;

    min-height:170px;

    background:rgba(255,255,255,.012);

}


.chart-title{

    font-size:9px;

    font-weight:800;

    margin-bottom:8px;

}


/* BAR CHART */

.bars{

    height:135px;

    display:flex;

    align-items:flex-end;

    justify-content:space-around;

    gap:8px;

    border-bottom:1px solid var(--border);

    position:relative;

}


.bar-group{

    height:100%;

    flex:1;

    display:flex;

    flex-direction:column;

    justify-content:flex-end;

    align-items:center;

    gap:4px;

}


.bar{

    width:min(38px,65%);

    height:0%;

    border-radius:5px 5px 0 0;

    background:
        linear-gradient(
            to top,
            var(--primary),
            var(--secondary)
        );

    transition:
        height .7s
        cubic-bezier(.22,1,.36,1);

}


.bar-value{

    font-size:8px;

    color:var(--muted);

}


.bar-label{

    font-size:7px;

    color:var(--muted);

}


/* =========================================================
   RADAR
========================================================= */

.radar-wrap{

    height:145px;

    display:grid;

    place-items:center;

}


#radar{

    width:145px;

    height:145px;

    overflow:visible;

}


.radar-grid{

    fill:none;

    stroke:var(--border);

    stroke-width:1;

}


.radar-axis{

    stroke:var(--border);

    stroke-width:1;

}


.radar-data{

    fill:
        color-mix(
            in srgb,
            var(--primary) 22%,
            transparent
        );

    stroke:var(--primary);

    stroke-width:2;

    transition:
        all .6s
        cubic-bezier(.22,1,.36,1);

}


.radar-point{

    fill:var(--primary);

    transition:
        cx .6s,
        cy .6s;

}


/* =========================================================
   HISTORY
========================================================= */

.history{

    margin-top:10px;

    border:1px solid var(--border);

    border-radius:11px;

    overflow:hidden;

}


.history-head{

    padding:10px;

}


.history-title{

    font-size:10px;

    font-weight:800;

}


.history-sub{

    margin-top:3px;

    color:var(--muted);

    font-size:8px;

}


.table-wrap{

    max-height:155px;

    overflow-y:auto;

}


table{

    width:100%;

    border-collapse:collapse;

    font-size:8px;

}


th{

    position:sticky;

    top:0;

    background:var(--panel2);

    color:var(--muted);

    text-align:left;

    padding:7px;

    font-weight:700;

}


td{

    padding:7px;

    border-top:1px solid var(--border);

}


.class-badge{

    color:var(--primary);

    font-weight:800;

}


/* =========================================================
   ANIMATIONS
========================================================= */

.fade{

    animation:
        fadeUp .55s
        cubic-bezier(.22,1,.36,1);

}


@keyframes fadeUp{

    from{

        opacity:0;

        transform:translateY(10px);

    }

    to{

        opacity:1;

        transform:translateY(0);

    }

}


/* =========================================================
   RESPONSIVE
========================================================= */

@media(max-width:850px){

    .dashboard{

        grid-template-columns:1fr;

    }

}


@media(max-width:600px){

    .navbar{

        flex-wrap:wrap;

    }

    .theme-switch{

        order:3;

        width:100%;

        justify-content:center;

    }

    .charts{

        grid-template-columns:1fr;

    }

    .flower-result{

        grid-template-columns:85px 1fr;

    }

    .flower-image{

        width:85px;

        height:65px;

    }

}


@media(prefers-reduced-motion:reduce){

    *{

        animation-duration:.01ms!important;

        transition-duration:.01ms!important;

    }

}

</style>

</head>


<body>

<div class="bg-grid"></div>

<div class="orb orb1"></div>

<div class="orb orb2"></div>


<div class="container">


<!-- =====================================================
     NAVBAR
===================================================== -->

<nav class="navbar">


<div class="brand">

    <div class="brand-icon">
        ⚡
    </div>

    <div>

        Iris AI Quantum Studio

        <span class="version">
            v4.3
        </span>

    </div>

</div>


<div style="display:flex;align-items:center;gap:9px">


<div class="theme-switch">

    <button
        class="theme-btn active"
        data-theme="quantum">
        Quantum
    </button>

    <button
        class="theme-btn"
        data-theme="cyber">
        Cyberpunk
    </button>

    <button
        class="theme-btn"
        data-theme="matrix">
        Matrix
    </button>

</div>


<div class="online">

    <span class="online-dot"></span>

    SVM Active

</div>


</div>


</nav>


<!-- =====================================================
     HERO
===================================================== -->

<section class="hero">

    <div class="hero-tag">
        Artificial Intelligence · Machine Learning
    </div>

    <h1>
        Iris <span>AI</span>
    </h1>

    <p>
        Phân tích hình thái và nhận diện loài hoa
        bằng mô hình Support Vector Machine.
    </p>

</section>


<!-- =====================================================
     DASHBOARD
===================================================== -->

<div class="dashboard">


<!-- =====================================================
     LEFT CONTROL
===================================================== -->

<section class="card">


<div class="card-header">

    <div class="card-title">
        Bộ điều khiển đặc trưng hình thái
    </div>

    <div class="card-sub">
        Tùy chỉnh thông số sinh học để mô hình AI phân loại thời gian thực
    </div>

</div>


<div class="controls">


<!-- SEPAL LENGTH -->

<div class="slider">

<div class="slider-head">

<span class="slider-name">
    Chiều dài đài hoa (Sepal Length)
</span>

<span
    class="value"
    id="slValue">
    5.1 cm
</span>

</div>

<input
    id="sl"
    type="range"
    min="4"
    max="8"
    step="0.1"
    value="5.1"
>

</div>


<!-- SEPAL WIDTH -->

<div class="slider">

<div class="slider-head">

<span class="slider-name">
    Chiều rộng đài hoa (Sepal Width)
</span>

<span
    class="value"
    id="swValue">
    3.5 cm
</span>

</div>

<input
    id="sw"
    type="range"
    min="2"
    max="4.5"
    step="0.1"
    value="3.5"
>

</div>


<!-- PETAL LENGTH -->

<div class="slider">

<div class="slider-head">

<span class="slider-name">
    Chiều dài cánh hoa (Petal Length)
</span>

<span
    class="value"
    id="plValue">
    1.4 cm
</span>

</div>

<input
    id="pl"
    type="range"
    min="1"
    max="7"
    step="0.1"
    value="1.4"
>

</div>


<!-- PETAL WIDTH -->

<div class="slider">

<div class="slider-head">

<span class="slider-name">
    Chiều rộng cánh hoa (Petal Width)
</span>

<span
    class="value"
    id="pwValue">
    0.2 cm
</span>

</div>

<input
    id="pw"
    type="range"
    min="0.1"
    max="2.5"
    step="0.1"
    value="0.2"
>

</div>


<!-- PROFILE -->

<div class="profile">

<div class="profile-head">

<span>
    MORPHOLOGY PROFILE
</span>

<span id="profileName">
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

<div class="section-label">
    Mẫu chuẩn sinh học
</div>


<div class="presets">


<button
    type="button"
    class="preset"
    data-type="setosa">

    <div class="preset-icon">
        🌱
    </div>

    <span class="preset-name">
        Setosa
    </span>

</button>


<button
    type="button"
    class="preset"
    data-type="versicolor">

    <div class="preset-icon">
        🌷
    </div>

    <span class="preset-name">
        Versicolor
    </span>

</button>


<button
    type="button"
    class="preset"
    data-type="virginica">

    <div class="preset-icon">
        🌸
    </div>

    <span class="preset-name">
        Virginica
    </span>

</button>


</div>


<!-- ACTION -->

<div class="actions">


<button
    type="button"
    id="predictBtn"
    class="predict">

    ✦ Phân tích AI ngay

</button>


<button
    type="button"
    id="resetBtn"
    class="reset"
    title="Reset">

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
     RIGHT
===================================================== -->

<section class="card result">


<div class="result-head">

<div>

<div class="result-title">
    Kết quả phân tích hình thái học
</div>

<div class="result-sub">
    Độ tin cậy từ thuật toán Support Vector Machine
</div>

</div>


<div
    id="latency"
    class="latency">

    ⚡ Ready

</div>


</div>


<!-- FLOWER -->

<div class="flower-result">


<img
    id="flowerImage"
    class="flower-image"
    src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
    alt="Iris Setosa"
>


<div>

<div
    id="resultName"
    class="result-name">

    Iris Setosa

</div>


<div
    id="resultDesc"
    class="result-desc">

    Iris Setosa thường có cánh hoa
    ngắn và hẹp.

</div>

</div>

</div>


<!-- =====================================================
     CHARTS
===================================================== -->

<div class="charts">


<!-- BAR -->

<div class="chart">

<div class="chart-title">
    Xác suất phân loại
</div>


<div class="bars">


<div class="bar-group">

    <div
        id="bar0value"
        class="bar-value">
        0%
    </div>

    <div
        id="bar0"
        class="bar">
    </div>

    <div class="bar-label">
        Setosa
    </div>

</div>


<div class="bar-group">

    <div
        id="bar1value"
        class="bar-value">
        0%
    </div>

    <div
        id="bar1"
        class="bar">
    </div>

    <div class="bar-label">
        Versicolor
    </div>

</div>


<div class="bar-group">

    <div
        id="bar2value"
        class="bar-value">
        0%
    </div>

    <div
        id="bar2"
        class="bar">
    </div>

    <div class="bar-label">
        Virginica
    </div>

</div>


</div>

</div>


<!-- RADAR -->

<div class="chart">

<div class="chart-title">
    Hồ sơ hình thái
</div>


<div class="radar-wrap">


<svg
    id="radar"
    viewBox="0 0 180 180"
    aria-label="Radar chart">


<polygon
    class="radar-grid"
    points="90,18 158,67 132,147 48,147 22,67">
</polygon>


<polygon
    class="radar-grid"
    points="90,38 140,74 121,130 59,130 40,74">
</polygon>


<polygon
    class="radar-grid"
    points="90,58 122,81 110,113 70,113 58,81">
</polygon>


<line
    class="radar-axis"
    x1="90"
    y1="90"
    x2="90"
    y2="18">
</line>


<line
    class="radar-axis"
    x1="90"
    y1="90"
    x2="158"
    y2="67">
</line>


<line
    class="radar-axis"
    x1="90"
    y1="90"
    x2="132"
    y2="147">
</line>


<line
    class="radar-axis"
    x1="90"
    y1="90"
    x2="48"
    y2="147">
</line>


<line
    class="radar-axis"
    x1="90"
    y1="90"
    x2="22"
    y2="67">
</line>


<polygon
    id="radarData"
    class="radar-data"
    points="90,55 125,78 110,115 60,115 45,78">
</polygon>


</svg>

</div>

</div>


</div>


<!-- =====================================================
     HISTORY
===================================================== -->

<div class="history">


<div class="history-head">

<div class="history-title">
    Lịch sử phiên làm việc
</div>

<div class="history-sub">
    Lưu trữ các phiên phân tích gần đây
</div>

</div>


<div class="table-wrap">

<table>

<thead>

<tr>

<th>Loài</th>

<th>SL / SW</th>

<th>PL / PW</th>

<th>Xác suất</th>

<th>Thời gian</th>

</tr>

</thead>


<tbody id="historyBody">


<tr>

<td class="class-badge">
    Iris Setosa
</td>

<td>
    5.1 / 3.5
</td>

<td>
    1.4 / 0.2
</td>

<td>
    —
</td>

<td>
    —
</td>

</tr>


</tbody>

</table>

</div>


</div>


</section>


</div>


</div>


<script>

/* =========================================================
   DATA
========================================================= */

const sliders = {

    sl:document.getElementById("sl"),

    sw:document.getElementById("sw"),

    pl:document.getElementById("pl"),

    pw:document.getElementById("pw")

};


const values = {

    sl:document.getElementById("slValue"),

    sw:document.getElementById("swValue"),

    pl:document.getElementById("plValue"),

    pw:document.getElementById("pwValue")

};


const presets = {

    setosa:{
        sl:5.1,
        sw:3.5,
        pl:1.4,
        pw:.2
    },

    versicolor:{
        sl:6.0,
        sw:2.9,
        pl:4.5,
        pw:1.5
    },

    virginica:{
        sl:6.5,
        sw:3.0,
        pl:5.5,
        pw:1.8
    }

};


const flowerInfo = {

    0:{
        name:"Iris Setosa",

        description:
            "Iris Setosa thường có cánh hoa ngắn và hẹp, với kích thước tổng thể nhỏ.",

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
    },

    1:{
        name:"Iris Versicolor",

        description:
            "Iris Versicolor có các đặc trưng hình thái ở mức trung gian.",

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg"
    },

    2:{
        name:"Iris Virginica",

        description:
            "Iris Virginica thường có cánh hoa dài và rộng hơn.",

        image:
            "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg"
    }

};


/* =========================================================
   SLIDER UPDATE
========================================================= */

function updateSlider(key){

    const slider =
        sliders[key];

    const min =
        Number(slider.min);

    const max =
        Number(slider.max);

    const value =
        Number(slider.value);


    const percent =
        ((value-min)/(max-min))*100;


    slider.style.setProperty(
        "--progress",
        percent + "%"
    );


    values[key].textContent =
        value.toFixed(1) + " cm";


    values[key].classList.remove(
        "bump"
    );

    void values[key].offsetWidth;

    values[key].classList.add(
        "bump"
    );


    updateProfile();

    updateRadar();

}


/* =========================================================
   PROFILE
========================================================= */

function updateProfile(){

    const pl =
        Number(sliders.pl.value);

    const pw =
        Number(sliders.pw.value);


    let score =

        ((pl-1)/6)*70 +

        ((pw-.1)/2.4)*30;


    score =
        Math.max(
            0,
            Math.min(100,score)
        );


    document.getElementById(
        "profileFill"
    ).style.width =
        score + "%";


    let label =
        "Compact";


    if(score >= 70){

        label =
            "Large";

    }
    else if(score >= 40){

        label =
            "Medium";

    }


    document.getElementById(
        "profileName"
    ).textContent =
        label;

}


/* =========================================================
   RADAR
========================================================= */

function updateRadar(){

    const data = [

        Number(sliders.sl.value)/8,

        Number(sliders.sw.value)/4.5,

        Number(sliders.pl.value)/7,

        Number(sliders.pw.value)/2.5,

        Number(sliders.sl.value)/8

    ];


    const centerX = 90;

    const centerY = 90;

    const radius = 72;


    const points =
        data.map(
            (value,index)=>{

                const angle =
                    (-90 +
                    index*72)
                    *Math.PI/180;


                const r =
                    value * radius;


                const x =
                    centerX +
                    Math.cos(angle)*r;


                const y =
                    centerY +
                    Math.sin(angle)*r;


                return `${x},${y}`;

            }
        ).join(" ");


    document.getElementById(
        "radarData"
    ).setAttribute(
        "points",
        points
    );

}


/* =========================================================
   INITIALIZE
========================================================= */

Object.keys(sliders).forEach(
    key=>{

        updateSlider(key);

        sliders[key].addEventListener(
            "input",
            ()=>{
                updateSlider(key);
            }
        );

    }
);


/* =========================================================
   PRESETS
========================================================= */

document
.querySelectorAll(".preset")
.forEach(
    button=>{

        button.addEventListener(
            "click",
            ()=>{

                const type =
                    button.dataset.type;


                const data =
                    presets[type];


                Object.entries(data)
                .forEach(
                    ([key,value])=>{

                        sliders[key].value =
                            value;

                        updateSlider(key);

                    }
                );


                predict();

            }
        );

    }
);


/* =========================================================
   PREDICT
========================================================= */

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
        "⏳ Đang phân tích...";


    const start =
        performance.now();


    const payload = {

        sepal_length:
            Number(
                sliders.sl.value
            ),

        sepal_width:
            Number(
                sliders.sw.value
            ),

        petal_length:
            Number(
                sliders.pl.value
            ),

        petal_width:
            Number(
                sliders.pw.value
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
                        JSON.stringify(payload)

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


        const elapsed =
            Math.round(
                performance.now() -
                start
            );


        const id =
            Number(
                result.class_id
            );


        const flower =
            flowerInfo[id];


        if(!flower){

            throw new Error(
                "Class không hợp lệ"
            );

        }


        /* -----------------------------------------
           FLOWER
        ----------------------------------------- */

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


        /* -----------------------------------------
           RESULT
        ----------------------------------------- */

        const resultName =
            document.getElementById(
                "resultName"
            );


        resultName.textContent =
            flower.name;


        resultName.classList.remove(
            "fade"
        );

        void resultName.offsetWidth;

        resultName.classList.add(
            "fade"
        );


        document.getElementById(
            "resultDesc"
        ).textContent =
            flower.description;


        /* -----------------------------------------
           PROBABILITIES
        ----------------------------------------- */

        const probabilities =
            result.probabilities ||
            [0,0,0];


        probabilities.forEach(
            (prob,index)=>{

                const percent =
                    Math.round(
                        Number(prob)*100
                    );


                document.getElementById(
                    "bar"+index
                ).style.height =
                    percent + "%";


                document.getElementById(
                    "bar"+index+"value"
                ).textContent =
                    percent + "%";

            }
        );


        /* -----------------------------------------
           LATENCY
        ----------------------------------------- */

        document.getElementById(
            "latency"
        ).textContent =
            "⚡ Latency: " +
            elapsed +
            "ms";


        /* -----------------------------------------
           HISTORY
        ----------------------------------------- */

        addHistory(
            flower.name,
            payload,
            probabilities[id]
        );


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
            "✦ Phân tích AI ngay";

    }

}


/* =========================================================
   HISTORY
========================================================= */

function addHistory(
    name,
    data,
    probability
){

    const body =
        document.getElementById(
            "historyBody"
        );


    const row =
        document.createElement(
            "tr"
        );


    const now =
        new Date();


    const time =
        now.toLocaleTimeString(
            "vi-VN",
            {
                hour:"2-digit",
                minute:"2-digit",
                second:"2-digit"
            }
        );


    row.innerHTML = `

        <td class="class-badge">
            ${name}
        </td>

        <td>
            ${data.sepal_length.toFixed(1)}
            /
            ${data.sepal_width.toFixed(1)}
        </td>

        <td>
            ${data.petal_length.toFixed(1)}
            /
            ${data.petal_width.toFixed(1)}
        </td>

        <td>
            ${(Number(probability)*100).toFixed(1)}%
        </td>

        <td>
            ${time}
        </td>

    `;


    body.prepend(row);


    /* giữ tối đa 15 dòng */

    while(
        body.children.length > 15
    ){

        body.removeChild(
            body.lastElementChild
        );

    }

}


/* =========================================================
   RESET
========================================================= */

document
.getElementById("resetBtn")
.addEventListener(
    "click",
    ()=>{

        Object.entries(
            presets.setosa
        ).forEach(
            ([key,value])=>{

                sliders[key].value =
                    value;

                updateSlider(key);

            }
        );


        predict();

    }
);


/* =========================================================
   PREDICT BUTTON
========================================================= */

document
.getElementById("predictBtn")
.addEventListener(
    "click",
    predict
);


/* =========================================================
   THEMES
========================================================= */

document
.querySelectorAll(".theme-btn")
.forEach(
    button=>{

        button.addEventListener(
            "click",
            ()=>{

                const theme =
                    button.dataset.theme;


                document.body.classList.remove(
                    "cyber",
                    "matrix"
                );


                if(theme === "cyber"){

                    document.body.classList.add(
                        "cyber"
                    );

                }


                if(theme === "matrix"){

                    document.body.classList.add(
                        "matrix"
                    );

                }


                document
                .querySelectorAll(
                    ".theme-btn"
                )
                .forEach(
                    item=>{

                        item.classList.remove(
                            "active"
                        );

                    }
                );


                button.classList.add(
                    "active"
                );

            }
        );

    }
);


/* =========================================================
   FIRST PREDICTION
========================================================= */

updateRadar();


</script>

</body>

</html>
"""


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "SVM",
        "probability": getattr(
            model,
            "probability",
            False
        )
    }


# =========================================================
# PREDICT API
# =========================================================

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


    # Lấy xác suất nếu model được train
    # với probability=True

    if getattr(
        model,
        "probability",
        False
    ):

        probabilities = (
            model.predict_proba(features)[0]
            .tolist()
        )

    else:

        probabilities = [
            0,
            0,
            0
        ]


    return {

        "class_id":
            prediction,

        "prediction":
            species[prediction],

        "probabilities":
            probabilities,

        "image":
            species_images[prediction],

        "description":
            species_descriptions[prediction]

    }
