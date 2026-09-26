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
    description="Iris Flower Classification using SVM",
    version="5.0.0"
)


# =========================================================
# INPUT
# =========================================================

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# =========================================================
# DATA
# =========================================================

species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


images = {
    0: "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG",
    1: "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg",
    2: "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg"
}


descriptions = {
    0: "Đặc trưng bởi cánh hoa ngắn, hẹp và kích thước tổng thể nhỏ.",
    1: "Có kích thước hình thái ở mức trung gian giữa Setosa và Virginica.",
    2: "Thường có cánh hoa dài và rộng, kích thước tổng thể lớn."
}


# =========================================================
# HOME
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return r"""
<!DOCTYPE html>

<html lang="vi">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Iris AI Quantum Studio</title>

<style>

*{
    box-sizing:border-box;
    margin:0;
    padding:0;
}

:root{

    --bg:#050816;
    --panel:#0d1326;
    --panel2:#11182e;
    --border:#263252;

    --primary:#8b5cf6;
    --secondary:#06b6d4;

    --text:#f8fafc;
    --muted:#7f8ba8;

}

body{

    min-height:100vh;

    font-family:
        Inter,
        Segoe UI,
        Arial,
        sans-serif;

    color:var(--text);

    background:

        radial-gradient(
            circle at 10% 0%,
            rgba(139,92,246,.28),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 100%,
            rgba(6,182,212,.22),
            transparent 30%
        ),

        var(--bg);

    transition:.4s;

}

body.cyber{

    --bg:#020708;
    --panel:#071517;
    --panel2:#0a2024;
    --border:#0d5960;

    --primary:#00e5ff;
    --secondary:#00ff88;

}

body.matrix{

    --bg:#010501;
    --panel:#061006;
    --panel2:#091609;
    --border:#164d16;

    --primary:#39ff14;
    --secondary:#00cc66;

}


.bg{

    position:fixed;
    inset:0;

    pointer-events:none;

    background-image:

        linear-gradient(
            rgba(255,255,255,.035) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(255,255,255,.035) 1px,
            transparent 1px
        );

    background-size:40px 40px;

    mask-image:
        linear-gradient(
            black,
            transparent
        );

}


/* =====================================================
NAVBAR
===================================================== */

.container{

    position:relative;

    z-index:2;

    width:min(1120px,94%);

    margin:auto;

    padding:18px 0 45px;

}


.nav{

    height:58px;

    padding:10px 16px;

    border:1px solid var(--border);

    border-radius:16px;

    background:
        rgba(10,15,35,.88);

    display:flex;

    align-items:center;

    justify-content:space-between;

    backdrop-filter:blur(20px);

    box-shadow:
        0 20px 60px rgba(0,0,0,.35);

}


.brand{

    display:flex;

    align-items:center;

    gap:10px;

    font-weight:800;

    font-size:14px;

}


.logo{

    width:35px;
    height:35px;

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
        rgba(139,92,246,.35);

}


.version{

    color:var(--primary);

    font-size:10px;

}


.nav-right{

    display:flex;

    align-items:center;

    gap:8px;

}


.theme{

    display:flex;

    padding:3px;

    border:1px solid var(--border);

    border-radius:8px;

}


.theme button{

    border:0;

    padding:6px 9px;

    border-radius:6px;

    color:var(--muted);

    background:transparent;

    font-size:9px;

    cursor:pointer;

}


.theme button.active{

    background:var(--primary);

    color:white;

}


.status{

    border:1px solid rgba(34,197,94,.3);

    color:#86efac;

    padding:7px 10px;

    border-radius:30px;

    font-size:9px;

}


.dot{

    display:inline-block;

    width:6px;
    height:6px;

    margin-right:5px;

    border-radius:50%;

    background:#22c55e;

    box-shadow:
        0 0 10px #22c55e;

}


/* =====================================================
HERO
===================================================== */

.hero{

    text-align:center;

    padding:40px 0 28px;

}


.hero-tag{

    color:var(--primary);

    font-size:9px;

    font-weight:800;

    letter-spacing:.18em;

}


.hero h1{

    margin-top:9px;

    font-size:
        clamp(36px,6vw,58px);

    font-weight:900;

    letter-spacing:-.05em;

}


.hero h1 span{

    color:var(--primary);

}


.hero p{

    margin:10px auto 0;

    color:var(--muted);

    font-size:11px;

}


/* =====================================================
GRID
===================================================== */

.grid{

    display:grid;

    grid-template-columns:
        .95fr
        1.05fr;

    gap:16px;

}


.card{

    border:1px solid var(--border);

    border-radius:18px;

    background:
        linear-gradient(
            145deg,
            rgba(13,19,38,.96),
            rgba(9,14,30,.96)
        );

    box-shadow:
        0 25px 70px rgba(0,0,0,.35);

    overflow:hidden;

}


/* =====================================================
CONTROL
===================================================== */

.header{

    padding:18px 18px 5px;

}


.title{

    font-size:13px;

    font-weight:800;

}


.subtitle{

    margin-top:4px;

    font-size:9px;

    color:var(--muted);

}


.controls{

    padding:5px 18px 18px;

}


.slider{

    margin-top:17px;

}


.slider-top{

    display:flex;

    align-items:center;

    justify-content:space-between;

    margin-bottom:8px;

}


.label{

    font-size:9px;

    font-weight:700;

}


.val{

    color:var(--primary);

    font-size:9px;

    font-weight:800;

    padding:4px 7px;

    border:1px solid var(--border);

    border-radius:5px;

}


input[type=range]{

    appearance:none;

    width:100%;

    height:4px;

    border-radius:20px;

    background:
        linear-gradient(
            90deg,
            var(--primary) var(--progress),
            #202943 var(--progress)
        );

    outline:none;

}


input[type=range]::-webkit-slider-thumb{

    appearance:none;

    width:15px;

    height:15px;

    border-radius:50%;

    background:#fff;

    border:3px solid var(--primary);

    box-shadow:
        0 0 15px
        rgba(139,92,246,.75);

    cursor:pointer;

    transition:.2s;

}


input[type=range]::-webkit-slider-thumb:hover{

    transform:scale(1.3);

}


/* =====================================================
PROFILE
===================================================== */

.profile{

    margin-top:18px;

    padding:10px;

    border:1px solid var(--border);

    border-radius:9px;

}


.profile-top{

    display:flex;

    justify-content:space-between;

    font-size:8px;

    color:var(--muted);

}


.profile-bar{

    margin-top:7px;

    height:5px;

    border-radius:20px;

    background:#202943;

    overflow:hidden;

}


.profile-fill{

    height:100%;

    width:20%;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--secondary)
        );

    transition:
        width .5s
        cubic-bezier(.22,1,.36,1);

}


/* =====================================================
PRESET
===================================================== */

.small-title{

    margin-top:15px;

    color:var(--muted);

    font-size:8px;

    text-transform:uppercase;

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

    border-radius:8px;

    background:
        rgba(255,255,255,.025);

    color:white;

    padding:8px;

    cursor:pointer;

    transition:.25s;

}


.preset:hover{

    transform:translateY(-3px);

    border-color:var(--primary);

}


.preset-icon{

    font-size:15px;

}


.preset span{

    display:block;

    margin-top:3px;

    font-size:8px;

    font-weight:800;

}


/* =====================================================
BUTTON
===================================================== */

.actions{

    display:grid;

    grid-template-columns:1fr 40px;

    gap:7px;

    margin-top:12px;

}


.predict{

    height:38px;

    border:0;

    border-radius:8px;

    color:white;

    font-weight:800;

    font-size:10px;

    background:
        linear-gradient(
            110deg,
            var(--primary),
            var(--secondary)
        );

    cursor:pointer;

    transition:.2s;

}


.predict:hover{

    transform:translateY(-2px);

}


.reset{

    border:1px solid var(--border);

    border-radius:8px;

    background:rgba(255,255,255,.04);

    color:var(--muted);

    cursor:pointer;

}


/* =====================================================
RESULT
===================================================== */

.result{

    padding:18px;

}


.result-top{

    display:flex;

    justify-content:space-between;

    margin-bottom:10px;

}


.latency{

    color:var(--primary);

    font-size:8px;

    padding:4px 7px;

    border-radius:5px;

    background:
        rgba(139,92,246,.12);

}


/* =====================================================
FLOWER
===================================================== */

.flower{

    display:grid;

    grid-template-columns:95px 1fr;

    gap:12px;

    align-items:center;

    padding:9px;

    border:1px solid var(--border);

    border-radius:11px;

    background:
        rgba(255,255,255,.025);

}


.flower img{

    width:95px;

    height:65px;

    object-fit:cover;

    border-radius:7px;

    animation:
        imageIn .5s ease;

}


@keyframes imageIn{

    from{

        opacity:0;

        transform:scale(.9);

    }

    to{

        opacity:1;

        transform:scale(1);

    }

}


.flower-name{

    font-size:17px;

    font-weight:900;

}


.flower-desc{

    margin-top:5px;

    color:var(--muted);

    font-size:9px;

    line-height:1.5;

}


/* =====================================================
CHART AREA
===================================================== */

.charts{

    display:grid;

    grid-template-columns:
        1fr 1fr;

    gap:9px;

    margin-top:9px;

}


.chart{

    height:190px;

    border:1px solid var(--border);

    border-radius:11px;

    padding:10px;

}


.chart-title{

    font-size:9px;

    font-weight:800;

    margin-bottom:5px;

}


/* =====================================================
PROBABILITY
===================================================== */

.prob-list{

    margin-top:14px;

}


.prob-item{

    margin-bottom:13px;

}


.prob-row{

    display:flex;

    justify-content:space-between;

    font-size:8px;

    margin-bottom:5px;

}


.prob-name{

    color:var(--muted);

}


.prob-value{

    color:var(--primary);

    font-weight:800;

}


.prob-track{

    width:100%;

    height:7px;

    border-radius:20px;

    background:#1b243b;

    overflow:hidden;

}


.prob-fill{

    height:100%;

    width:0%;

    border-radius:20px;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--secondary)
        );

    transition:
        width .8s
        cubic-bezier(.22,1,.36,1);

}


/* =====================================================
RADAR
===================================================== */

.radar-box{

    height:155px;

    display:flex;

    justify-content:center;

    align-items:center;

}


#radar{

    width:175px;

    height:175px;

    overflow:visible;

}


.grid-line{

    fill:none;

    stroke:var(--border);

    stroke-width:1;

}


.axis{

    stroke:var(--border);

    stroke-width:1;

}


.radar-shape{

    fill:
        rgba(139,92,246,.18);

    stroke:var(--primary);

    stroke-width:2;

    transition:
        .5s
        cubic-bezier(.22,1,.36,1);

}


.radar-point{

    fill:var(--primary);

}


.radar-label{

    fill:#7f8ba8;

    font-size:7px;

    font-weight:700;

}


/* =====================================================
HISTORY
===================================================== */

.history{

    margin-top:9px;

    border:1px solid var(--border);

    border-radius:11px;

    overflow:hidden;

}


.history-title{

    padding:10px;

    font-size:9px;

    font-weight:800;

}


.history-scroll{

    max-height:145px;

    overflow:auto;

}


table{

    width:100%;

    border-collapse:collapse;

    font-size:8px;

}


th{

    padding:7px;

    color:var(--muted);

    text-align:left;

    background:var(--panel2);

    position:sticky;

    top:0;

}


td{

    padding:7px;

    border-top:1px solid var(--border);

}


.badge{

    color:var(--primary);

    font-weight:800;

}


/* =====================================================
RESPONSIVE
===================================================== */

@media(max-width:850px){

    .grid{

        grid-template-columns:1fr;

    }

}


@media(max-width:600px){

    .nav-right{

        display:none;

    }

    .charts{

        grid-template-columns:1fr;

    }

}

</style>

</head>


<body>

<div class="bg"></div>


<div class="container">


<!-- NAV -->

<nav class="nav">

<div class="brand">

<div class="logo">⚡</div>

<div>
Iris AI Quantum Studio
<span class="version">v5.0</span>
</div>

</div>


<div class="nav-right">

<div class="theme">

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


<div class="status">

<span class="dot"></span>

SVM Active

</div>

</div>

</nav>


<!-- HERO -->

<section class="hero">

<div class="hero-tag">
ARTIFICIAL INTELLIGENCE · MACHINE LEARNING
</div>

<h1>
Iris <span>AI</span>
</h1>

<p>
Phân tích hình thái và nhận diện loài hoa
bằng mô hình Support Vector Machine.
</p>

</section>


<!-- DASHBOARD -->

<div class="grid">


<!-- LEFT -->

<section class="card">

<div class="header">

<div class="title">
Bộ điều khiển đặc trưng hình thái
</div>

<div class="subtitle">
Tùy chỉnh thông số sinh học để mô hình AI phân loại thời gian thực
</div>

</div>


<div class="controls">


<!-- 1 -->

<div class="slider">

<div class="slider-top">

<span class="label">
Chiều dài đài hoa (Sepal Length)
</span>

<span class="val" id="slVal">
5.1 cm
</span>

</div>

<input
id="sl"
type="range"
min="4"
max="8"
step="0.1"
value="5.1">

</div>


<!-- 2 -->

<div class="slider">

<div class="slider-top">

<span class="label">
Chiều rộng đài hoa (Sepal Width)
</span>

<span class="val" id="swVal">
3.5 cm
</span>

</div>

<input
id="sw"
type="range"
min="2"
max="4.5"
step="0.1"
value="3.5">

</div>


<!-- 3 -->

<div class="slider">

<div class="slider-top">

<span class="label">
Chiều dài cánh hoa (Petal Length)
</span>

<span class="val" id="plVal">
1.4 cm
</span>

</div>

<input
id="pl"
type="range"
min="1"
max="7"
step="0.1"
value="1.4">

</div>


<!-- 4 -->

<div class="slider">

<div class="slider-top">

<span class="label">
Chiều rộng cánh hoa (Petal Width)
</span>

<span class="val" id="pwVal">
0.2 cm
</span>

</div>

<input
id="pw"
type="range"
min="0.1"
max="2.5"
step="0.1"
value="0.2">

</div>


<!-- PROFILE -->

<div class="profile">

<div class="profile-top">

<span>MORPHOLOGY PROFILE</span>

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

<div class="small-title">
Mẫu chuẩn sinh học
</div>


<div class="presets">

<button
class="preset"
data-type="setosa">

<div class="preset-icon">🌱</div>

<span>Setosa</span>

</button>


<button
class="preset"
data-type="versicolor">

<div class="preset-icon">🌷</div>

<span>Versicolor</span>

</button>


<button
class="preset"
data-type="virginica">

<div class="preset-icon">🌸</div>

<span>Virginica</span>

</button>

</div>


<!-- ACTION -->

<div class="actions">

<button
id="predictBtn"
class="predict">

✦ Phân tích AI ngay

</button>


<button
id="resetBtn"
class="reset">

↻

</button>

</div>


</div>

</section>


<!-- RIGHT -->

<section class="card result">


<div class="result-top">

<div>

<div class="title">
Kết quả phân tích hình thái học
</div>

<div class="subtitle">
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

<div class="flower">

<img
id="flowerImage"
src="https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
alt="Iris">

<div>

<div
id="flowerName"
class="flower-name">

Iris Setosa

</div>

<div
id="flowerDesc"
class="flower-desc">

Đặc trưng bởi cánh hoa ngắn,
hẹp và kích thước tổng thể nhỏ.

</div>

</div>

</div>


<!-- CHARTS -->

<div class="charts">


<!-- PROBABILITY -->

<div class="chart">

<div class="chart-title">
Xác suất phân loại
</div>


<div class="prob-list">


<div class="prob-item">

<div class="prob-row">

<span class="prob-name">
Setosa
</span>

<span
id="p0"
class="prob-value">
0.0%
</span>

</div>

<div class="prob-track">

<div
id="f0"
class="prob-fill">
</div>

</div>

</div>


<div class="prob-item">

<div class="prob-row">

<span class="prob-name">
Versicolor
</span>

<span
id="p1"
class="prob-value">
0.0%
</span>

</div>

<div class="prob-track">

<div
id="f1"
class="prob-fill">
</div>

</div>

</div>


<div class="prob-item">

<div class="prob-row">

<span class="prob-name">
Virginica
</span>

<span
id="p2"
class="prob-value">
0.0%
</span>

</div>

<div class="prob-track">

<div
id="f2"
class="prob-fill">
</div>

</div>

</div>


</div>

</div>


<!-- RADAR -->

<div class="chart">

<div class="chart-title">
Hồ sơ hình thái
</div>


<div class="radar-box">


<svg
id="radar"
viewBox="0 0 200 200">


<!-- outer -->

<polygon
class="grid-line"
points="
100,20
176,75
147,164
53,164
24,75
">
</polygon>


<!-- middle -->

<polygon
class="grid-line"
points="
100,47
147,81
129,144
71,144
53,81
">
</polygon>


<!-- inner -->

<polygon
class="grid-line"
points="
100,70
126,89
116,123
84,123
74,89
">
</polygon>


<!-- axes -->

<line
class="axis"
x1="100"
y1="100"
x2="100"
y2="20">
</line>

<line
class="axis"
x1="100"
y1="100"
x2="176"
y2="75">
</line>

<line
class="axis"
x1="100"
y1="100"
x2="147"
y2="164">
</line>

<line
class="axis"
x1="100"
y1="100"
x2="53"
y2="164">
</line>

<line
class="axis"
x1="100"
y1="100"
x2="24"
y2="75">
</line>


<!-- DATA -->

<polygon
id="radarShape"
class="radar-shape"
points="
100,50
140,86
120,127
80,127
60,86
">
</polygon>


<!-- labels -->

<text
class="radar-label"
x="100"
y="12"
text-anchor="middle">
Sepal L
</text>


<text
class="radar-label"
x="184"
y="72"
text-anchor="start">
Sepal W
</text>


<text
class="radar-label"
x="151"
y="177"
text-anchor="middle">
Petal L
</text>


<text
class="radar-label"
x="49"
y="177"
text-anchor="middle">
Petal W
</text>


<text
class="radar-label"
x="16"
y="72"
text-anchor="end">
Sepal
</text>


</svg>

</div>

</div>

</div>


<!-- HISTORY -->

<div class="history">

<div class="history-title">
Lịch sử phiên làm việc
</div>


<div class="history-scroll">

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

</tbody>

</table>

</div>

</div>


</section>

</div>

</div>


<script>

/* =====================================================
DATA
===================================================== */

const sliderData = {

    sl:{
        el:document.getElementById("sl"),
        val:document.getElementById("slVal")
    },

    sw:{
        el:document.getElementById("sw"),
        val:document.getElementById("swVal")
    },

    pl:{
        el:document.getElementById("pl"),
        val:document.getElementById("plVal")
    },

    pw:{
        el:document.getElementById("pw"),
        val:document.getElementById("pwVal")
    }

};


const presets = {

    setosa:{
        sl:5.1,
        sw:3.5,
        pl:1.4,
        pw:0.2
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


/* =====================================================
SLIDERS
===================================================== */

function updateSliders(){

    Object.values(sliderData).forEach(item=>{

        const el = item.el;

        const min =
            Number(el.min);

        const max =
            Number(el.max);

        const value =
            Number(el.value);

        const percent =
            ((value-min)/(max-min))*100;

        el.style.setProperty(
            "--progress",
            percent + "%"
        );

        item.val.textContent =
            value.toFixed(1) + " cm";

    });


    updateProfile();

    updateRadar();

}


Object.values(sliderData).forEach(item=>{

    item.el.addEventListener(
        "input",
        updateSliders
    );

});


/* =====================================================
PROFILE
===================================================== */

function updateProfile(){

    const pl =
        Number(sliderData.pl.el.value);

    const pw =
        Number(sliderData.pw.el.value);


    const score =
        Math.max(
            0,
            Math.min(
                100,
                ((pl-1)/6)*70 +
                ((pw-.1)/2.4)*30
            )
        );


    document.getElementById(
        "profileFill"
    ).style.width =
        score + "%";


    let name="Compact";

    if(score > 70)
        name="Large";

    else if(score > 40)
        name="Medium";


    document.getElementById(
        "profileText"
    ).textContent =
        name;

}


/* =====================================================
RADAR
===================================================== */

function updateRadar(){

    const values = [

        Number(sliderData.sl.el.value),
        Number(sliderData.sw.el.value),
        Number(sliderData.pl.el.value),
        Number(sliderData.pw.el.value),
        Number(sliderData.sl.el.value)

    ];


    const min = [
        4,
        2,
        1,
        0.1,
        4
    ];


    const max = [
        8,
        4.5,
        7,
        2.5,
        8
    ];


    const centerX = 100;
    const centerY = 100;
    const radius = 78;


    const points = [];


    for(
        let i=0;
        i<5;
        i++
    ){

        let normalized =
            (values[i]-min[i]) /
            (max[i]-min[i]);


        normalized =
            Math.max(
                .08,
                Math.min(
                    1,
                    normalized
                )
            );


        const angle =
            (-90 + i*72)
            * Math.PI / 180;


        const r =
            normalized * radius;


        const x =
            centerX +
            Math.cos(angle)*r;


        const y =
            centerY +
            Math.sin(angle)*r;


        points.push(
            `${x},${y}`
        );

    }


    document.getElementById(
        "radarShape"
    ).setAttribute(
        "points",
        points.join(" ")
    );

}


/* =====================================================
PREDICT
===================================================== */

async function predict(){

    const button =
        document.getElementById(
            "predictBtn"
        );


    button.disabled=true;

    button.textContent=
        "⏳ Đang phân tích...";


    const start =
        performance.now();


    const data = {

        sepal_length:
            Number(
                sliderData.sl.el.value
            ),

        sepal_width:
            Number(
                sliderData.sw.el.value
            ),

        petal_length:
            Number(
                sliderData.pl.el.value
            ),

        petal_width:
            Number(
                sliderData.pw.el.value
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


        const result =
            await response.json();


        if(!response.ok){

            throw new Error(
                result.detail ||
                "API Error"
            );

        }


        console.log(
            "API RESULT:",
            result
        );


        /* =========================
           FLOWER
        ========================= */

        document.getElementById(
            "flowerName"
        ).textContent =
            result.prediction;


        document.getElementById(
            "flowerDesc"
        ).textContent =
            result.description;


        const image =
            document.getElementById(
                "flowerImage"
            );


        image.src =
            result.image;


        image.alt =
            result.prediction;


        /* =========================
           PROBABILITY
        ========================= */

        if(
            !result.probabilities ||
            result.probabilities.length !== 3
        ){

            throw new Error(
                "Model chưa có xác suất. Hãy upload svm_model.pkl mới với probability=True."
            );

        }


        result.probabilities.forEach(
            (prob,index)=>{

                const percent =
                    Number(prob)*100;


                document.getElementById(
                    "p"+index
                ).textContent =
                    percent.toFixed(1)+"%";


                setTimeout(()=>{

                    document.getElementById(
                        "f"+index
                    ).style.width =
                        percent+"%";

                },50);

            }
        );


        /* =========================
           LATENCY
        ========================= */

        const latency =
            Math.round(
                performance.now() -
                start
            );


        document.getElementById(
            "latency"
        ).textContent =
            "⚡ Latency: " +
            latency +
            "ms";


        /* =========================
           HISTORY
        ========================= */

        addHistory(
            result.prediction,
            data,
            result.probabilities[
                result.class_id
            ]
        );

    }

    catch(error){

        alert(
            error.message
        );

        console.error(error);

    }


    finally{

        button.disabled=false;

        button.textContent=
            "✦ Phân tích AI ngay";

    }

}


/* =====================================================
HISTORY
===================================================== */

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


    const time =
        new Date()
        .toLocaleTimeString(
            "vi-VN"
        );


    row.innerHTML=`

        <td class="badge">
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
            ${(probability*100).toFixed(1)}%
        </td>

        <td>
            ${time}
        </td>

    `;


    body.prepend(row);

}


/* =====================================================
PRESETS
===================================================== */

document
.querySelectorAll(".preset")
.forEach(button=>{

    button.addEventListener(
        "click",
        ()=>{

            const data =
                presets[
                    button.dataset.type
                ];


            sliderData.sl.el.value =
                data.sl;

            sliderData.sw.el.value =
                data.sw;

            sliderData.pl.el.value =
                data.pl;

            sliderData.pw.el.value =
                data.pw;


            updateSliders();

            predict();

        }
    );

});


/* =====================================================
RESET
===================================================== */

document
.getElementById("resetBtn")
.addEventListener(
    "click",
    ()=>{

        sliderData.sl.el.value=5.1;
        sliderData.sw.el.value=3.5;
        sliderData.pl.el.value=1.4;
        sliderData.pw.el.value=0.2;

        updateSliders();

        predict();

    }
);


/* =====================================================
BUTTON
===================================================== */

document
.getElementById("predictBtn")
.addEventListener(
    "click",
    predict
);


/* =====================================================
THEMES
===================================================== */

document
.querySelectorAll(".theme-btn")
.forEach(button=>{

    button.addEventListener(
        "click",
        ()=>{

            document.body.classList.remove(
                "cyber",
                "matrix"
            );


            const theme =
                button.dataset.theme;


            if(theme==="cyber")
                document.body.classList.add("cyber");


            if(theme==="matrix")
                document.body.classList.add("matrix");


            document
            .querySelectorAll(".theme-btn")
            .forEach(
                x=>x.classList.remove("active")
            );


            button.classList.add("active");

        }
    );

});


/* =====================================================
INIT
===================================================== */

updateSliders();

predict();

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
        "probability": bool(
            getattr(
                model,
                "probability",
                False
            )
        )
    }


# =========================================================
# PREDICT
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


    # Không cho phép model cũ
    # âm thầm trả 0%

    if not getattr(
        model,
        "probability",
        False
    ):

        return {
            "error":
                "svm_model.pkl chưa được train với probability=True"
        }


    probabilities = (
        model
        .predict_proba(features)[0]
        .tolist()
    )


    return {

        "class_id":
            prediction,

        "prediction":
            species[prediction],

        "probabilities":
            probabilities,

        "image":
            images[prediction],

        "description":
            descriptions[prediction]

    }
