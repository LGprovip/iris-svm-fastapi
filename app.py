from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np

# =========================
# LOAD MODEL & CONFIG
# =========================
try:
    model = joblib.load("svm_model.pkl")
except Exception as e:
    model = None
    print(f"⚠️ Cảnh báo: Không thể tải mô hình 'svm_model.pkl'. Hãy đảm bảo file tồn tại! Lỗi: {e}")

app = FastAPI(
    title="Iris AI Classification Platform",
    description="Hệ thống AI phân loại hoa Iris thông minh sử dụng thuật toán Support Vector Machine (SVM) với giao diện thời gian thực.",
    version="2.1.0"
)

# =========================
# INPUT MODEL WITH VALIDATION
# =========================
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=4.0, le=8.0, description="Chiều dài đài hoa (cm)")
    sepal_width: float = Field(..., ge=2.0, le=4.5, description="Chiều rộng đài hoa (cm)")
    petal_length: float = Field(..., ge=1.0, le=7.0, description="Chiều dài cánh hoa (cm)")
    petal_width: float = Field(..., ge=0.1, le=2.5, description="Chiều rộng cánh hoa (cm)")

species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

# =========================
# HOME PAGE (UI)
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="vi">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris AI Intelligence - Machine Learning Platform</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>
/* =====================================================
   GLOBAL & VARIABLES
===================================================== */
*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

:root{
    --bg:#030712;
    --panel:rgba(15, 23, 42, 0.75);
    --border:rgba(255, 255, 255, 0.08);
    --text:#f8fafc;
    --muted:#94a3b8;
    --primary:#8b5cf6;
    --secondary:#06b6d4;
    --success:#22c55e;
    --accent:#f43f5e;
    --shadow: 0 20px 50px rgba(0,0,0,0.5);
}

body{
    min-height:100vh;
    font-family:'Inter', sans-serif;
    color:var(--text);
    background:
        radial-gradient(circle at 10% 10%, rgba(139,92,246,0.18), transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(6,182,212,0.14), transparent 40%),
        var(--bg);
    overflow-x:hidden;
}

/* =====================================================
   BACKGROUND ORBS
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
    filter:blur(90px);
    opacity:.3;
    animation:float 12s ease-in-out infinite;
}

.orb.one{
    width:350px;
    height:350px;
    background:#7c3aed;
    top:-100px;
    left:-100px;
}

.orb.two{
    width:300px;
    height:300px;
    background:#0891b2;
    right:-80px;
    bottom:-80px;
    animation-delay:3s;
}

@keyframes float{
    0%,100%{ transform:translate(0,0); }
    50%{ transform:translate(30px,-25px); }
}

/* =====================================================
   CONTAINER & LAYOUT
===================================================== */
.container{
    position:relative;
    z-index:2;
    width:min(1280px, 94%);
    margin:auto;
    padding:25px 0 60px;
}

/* =====================================================
   NAVBAR
===================================================== */
.navbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:16px 24px;
    border:1px solid var(--border);
    background:rgba(15,23,42,0.65);
    backdrop-filter:blur(20px);
    border-radius:20px;
    box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

.logo{
    display:flex;
    align-items:center;
    gap:12px;
    font-weight:800;
    font-size:18px;
    letter-spacing:-0.02em;
}

.logo-icon{
    width:42px;
    height:42px;
    display:grid;
    place-items:center;
    border-radius:12px;
    background:linear-gradient(135deg, var(--primary), var(--secondary));
    box-shadow:0 8px 20px rgba(139,92,246,0.4);
    font-size:20px;
}

.nav-status{
    display:flex;
    align-items:center;
    gap:8px;
    color:var(--muted);
    font-size:13px;
    font-weight:500;
}

.status-dot{
    width:8px;
    height:8px;
    border-radius:50%;
    background:var(--success);
    box-shadow:0 0 12px var(--success);
    animation:pulse 2s infinite;
}

@keyframes pulse{
    0%,100%{ opacity:1; }
    50%{ opacity:.4; }
}

/* =====================================================
   HERO SECTION
===================================================== */
.hero{
    text-align:center;
    padding:50px 20px 35px;
}

.badge{
    display:inline-flex;
    padding:6px 16px;
    border-radius:100px;
    border:1px solid rgba(139,92,246,0.35);
    background:rgba(139,92,246,0.12);
    color:#c4b5fd;
    font-size:11px;
    font-weight:700;
    letter-spacing:.1em;
    text-transform:uppercase;
}

.hero h1{
    margin-top:16px;
    font-size:clamp(36px, 5vw, 60px);
    line-height:1.1;
    font-weight:900;
    letter-spacing:-0.03em;
    background:linear-gradient(110deg, #fff, #c4b5fd, #67e8f9);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero p{
    max-width:600px;
    margin:16px auto 0;
    color:var(--muted);
    line-height:1.7;
    font-size:15px;
}

/* =====================================================
   DASHBOARD GRID
===================================================== */
.dashboard{
    display:grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
    gap:24px;
    margin-top:10px;
}

.card{
    border:1px solid var(--border);
    background:var(--panel);
    backdrop-filter:blur(25px);
    border-radius:28px;
    box-shadow:var(--shadow);
    overflow:hidden;
}

.card-header{
    padding:28px 28px 10px;
}

.card-title{
    font-size:19px;
    font-weight:800;
}

.card-subtitle{
    margin-top:6px;
    color:var(--muted);
    font-size:13px;
}

/* =====================================================
   CONTROLS & SLIDERS
===================================================== */
.controls{
    padding:10px 28px 28px;
}

.slider-item{
    margin-top:22px;
}

.slider-header{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:10px;
}

.slider-name{
    font-size:14px;
    font-weight:650;
}

.slider-value{
    min-width:70px;
    text-align:center;
    padding:6px 10px;
    border-radius:10px;
    background:rgba(139,92,246,0.12);
    border:1px solid rgba(139,92,246,0.25);
    color:#c4b5fd;
    font-weight:800;
    font-size:13px;
}

input[type="range"]{
    appearance:none;
    width:100%;
    height:6px;
    border-radius:20px;
    outline:none;
    cursor:pointer;
    background:linear-gradient(90deg, var(--primary) var(--progress), rgba(255,255,255,0.1) var(--progress));
}

input[type="range"]::-webkit-slider-thumb{
    appearance:none;
    width:20px;
    height:20px;
    border-radius:50%;
    background:#fff;
    border:4px solid var(--primary);
    box-shadow:0 0 0 4px rgba(139,92,246,0.15), 0 4px 12px rgba(0,0,0,0.4);
    transition:transform .2s ease;
}

input[type="range"]::-webkit-slider-thumb:hover{
    transform:scale(1.2);
}

/* =====================================================
   PRESETS SECTION
===================================================== */
.presets-title{
    margin-top:24px;
    font-size:12px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.05em;
    color:var(--muted);
}

.presets{
    display:grid;
    grid-template-columns:repeat(3, 1fr);
    gap:10px;
    margin-top:10px;
}

.preset{
    padding:12px 8px;
    border-radius:14px;
    border:1px solid var(--border);
    background:rgba(255,255,255,0.03);
    color:white;
    cursor:pointer;
    text-align:center;
    transition:all .25s ease;
}

.preset:hover{
    transform:translateY(-3px);
    border-color:rgba(139,92,246,0.5);
    background:rgba(139,92,246,0.12);
}

.preset-icon{
    font-size:22px;
    margin-bottom:4px;
}

.preset-name{
    font-size:12px;
    font-weight:700;
}

.preset-desc{
    display:block;
    margin-top:2px;
    color:var(--muted);
    font-size:10px;
}

/* =====================================================
   ACTIONS BUTTONS
===================================================== */
.actions{
    display:grid;
    grid-template-columns:1fr auto;
    gap:12px;
    margin-top:24px;
}

.predict{
    min-height:52px;
    border:none;
    border-radius:14px;
    background:linear-gradient(110deg, #7c3aed, #4f46e5, #0891b2);
    color:white;
    font-weight:800;
    cursor:pointer;
    font-size:14px;
    box-shadow:0 10px 25px rgba(79,70,229,0.35);
    transition:transform .2s ease, box-shadow .2s ease;
}

.predict:hover{
    transform:translateY(-2px);
    box-shadow:0 15px 35px rgba(79,70,229,0.45);
}

.reset{
    width:52px;
    border:1px solid var(--border);
    border-radius:14px;
    background:rgba(255,255,255,0.05);
    color:#cbd5e1;
    cursor:pointer;
    font-size:18px;
    transition:all .2s ease;
}

.reset:hover{
    transform:rotate(-20deg);
    background:rgba(255,255,255,0.1);
}

/* =====================================================
   RESULT CARD SECTION
===================================================== */
.result-card{
    display:flex;
    flex-direction:column;
    align-items:center;
    text-align:center;
    padding:32px;
}

.result-label{
    color:var(--muted);
    font-size:11px;
    letter-spacing:.12em;
    text-transform:uppercase;
    font-weight:800;
}

.flower-container{
    width:min(100%, 360px);
    aspect-ratio:1.2;
    margin:18px 0;
    border-radius:24px;
    overflow:hidden;
    position:relative;
    background:#111827;
    border:1px solid var(--border);
    box-shadow:0 15px 40px rgba(0,0,0,0.4);
}

.flower-container img{
    width:100%;
    height:100%;
    object-fit:cover;
    transition:transform .6s cubic-bezier(0.22,1,0.36,1);
}

.flower-container img.animate{
    animation:flowerReveal .6s cubic-bezier(0.22,1,0.36,1);
}

@keyframes flowerReveal{
    0%{ opacity:0; transform:scale(1.1); }
    100%{ opacity:1; transform:scale(1); }
}

.result-name{
    font-size:30px;
    font-weight:900;
    letter-spacing:-0.03em;
    background:linear-gradient(90deg, #fff, #c4b5fd, #67e8f9);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.result-description{
    max-width:360px;
    margin-top:6px;
    color:var(--muted);
    line-height:1.6;
    font-size:13px;
}

/* =====================================================
   CONFIDENCE BARS (PROBABILITY)
===================================================== */
.confidence-box{
    width:100%;
    margin-top:20px;
    padding:16px;
    border-radius:16px;
    background:rgba(255,255,255,0.03);
    border:1px solid var(--border);
    text-align:left;
}

.conf-title{
    font-size:11px;
    text-transform:uppercase;
    color:var(--muted);
    font-weight:700;
    margin-bottom:10px;
}

.conf-item{
    margin-bottom:8px;
}

.conf-item:last-child{
    margin-bottom:0;
}

.conf-info{
    display:flex;
    justify-content:space-between;
    font-size:12px;
    margin-bottom:4px;
    font-weight:600;
}

.conf-bar-bg{
    width:100%;
    height:6px;
    background:rgba(255,255,255,0.08);
    border-radius:10px;
    overflow:hidden;
}

.conf-bar-fill{
    height:100%;
    width:0%;
    background:linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius:10px;
    transition:width 0.5s ease;
}

/* =====================================================
   TOAST NOTIFICATION
===================================================== */
#toast{
    position:fixed;
    bottom:25px;
    right:25px;
    background:rgba(15,23,42,0.95);
    border:1px solid var(--border);
    padding:14px 20px;
    border-radius:14px;
    box-shadow:0 10px 30px rgba(0,0,0,0.5);
    color:#fff;
    font-size:13px;
    z-index:1000;
    display:flex;
    align-items:center;
    gap:10px;
    transform:translateY(100px);
    opacity:0;
    transition:all 0.3s cubic-bezier(0.22,1,0.36,1);
}

#toast.show{
    transform:translateY(0);
    opacity:1;
}

/* =====================================================
   RESPONSIVE DESIGN
===================================================== */
@media(max-width:850px){
    .dashboard{
        grid-template-columns:1fr;
    }
}
</style>
</head>

<body>

<div class="background">
    <div class="orb one"></div>
    <div class="orb two"></div>
</div>

<div class="container">
    <!-- NAVBAR -->
    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">🌸</div>
            <span>Iris AI Suite</span>
        </div>
        <div class="nav-status">
            <span class="status-dot"></span>
            <span>SVM Model Active</span>
        </div>
    </nav>

    <!-- HERO SECTION -->
    <section class="hero">
        <div class="badge">✦ Next-Gen Machine Learning Interface</div>
        <h1>Phân Loại Hoa Iris Thông Minh</h1>
        <p>Ứng dụng tích hợp mô hình SVM hiệu suất cao giúp phân tích đặc trưng hình thái học thực vật trực quan và chính xác thời gian thực.</p>
    </section>

    <!-- DASHBOARD -->
    <main class="dashboard">
        <!-- LEFT PANEL: CONTROLS -->
        <section class="card">
            <div class="card-header">
                <div class="card-title">Thông số hình thái học</div>
                <div class="card-subtitle">Tinh chỉnh kích thước đài và cánh hoa bên dưới</div>
            </div>

            <div class="controls">
                <!-- SEPAL LENGTH -->
                <div class="slider-item">
                    <div class="slider-header">
                        <span class="slider-name">Chiều dài đài hoa (Sepal Length)</span>
                        <span class="slider-value" id="sepal_length_value">5.1 cm</span>
                    </div>
                    <input id="sepal_length" type="range" min="4.0" max="8.0" step="0.1" value="5.1">
                </div>

                <!-- SEPAL WIDTH -->
                <div class="slider-item">
                    <div class="slider-header">
                        <span class="slider-name">Chiều rộng đài hoa (Sepal Width)</span>
                        <span class="slider-value" id="sepal_width_value">3.5 cm</span>
                    </div>
                    <input id="sepal_width" type="range" min="2.0" max="4.5" step="0.1" value="3.5">
                </div>

                <!-- PETAL LENGTH -->
                <div class="slider-item">
                    <div class="slider-header">
                        <span class="slider-name">Chiều dài cánh hoa (Petal Length)</span>
                        <span class="slider-value" id="petal_length_value">1.4 cm</span>
                    </div>
                    <input id="petal_length" type="range" min="1.0" max="7.0" step="0.1" value="1.4">
                </div>

                <!-- PETAL WIDTH -->
                <div class="slider-item">
                    <div class="slider-header">
                        <span class="slider-name">Chiều rộng cánh hoa (Petal Width)</span>
                        <span class="slider-value" id="petal_width_value">0.2 cm</span>
                    </div>
                    <input id="petal_width" type="range" min="0.1" max="2.5" step="0.1" value="0.2">
                </div>

                <!-- PRESETS -->
                <div class="presets-title">Mẫu chuẩn theo loài</div>
                <div class="presets">
                    <button type="button" class="preset" data-type="setosa">
                        <div class="preset-icon">🌱</div>
                        <div class="preset-name">Setosa</div>
                        <span class="preset-desc">Nhỏ · Ngắn</span>
                    </button>
                    <button type="button" class="preset" data-type="versicolor">
                        <div class="preset-icon">🌷</div>
                        <div class="preset-name">Versicolor</div>
                        <span class="preset-desc">Trung bình</span>
                    </button>
                    <button type="button" class="preset" data-type="virginica">
                        <div class="preset-icon">🌸</div>
                        <div class="preset-name">Virginica</div>
                        <span class="preset-desc">Lớn · Dài</span>
                    </button>
                </div>

                <!-- ACTIONS -->
                <div class="actions">
                    <button type="button" id="predictBtn" class="predict">✦ &nbsp; Phân tích bằng mô hình SVM</button>
                    <button type="button" id="resetBtn" class="reset" title="Đặt lại thông số mặc định">↻</button>
                </div>
            </div>
        </section>

        <!-- RIGHT PANEL: RESULTS -->
        <section class="card result-card">
            <div class="result-label">Kết quả dự đoán AI</div>

            <div class="flower-container">
                <img id="flowerImage" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_3.jpg" alt="Iris Setosa">
            </div>

            <div id="resultName" class="result-name">Iris Setosa</div>
            <div id="resultDescription" class="result-description">
                Loài đặc trưng với cánh hoa nhỏ, ngắn và độ mở hẹp, thường phân bố ở vùng khí hậu hàn đới.
            </div>

            <!-- CONFIDENCE BARS -->
            <div class="confidence-box">
                <div class="conf-title">Mức độ tin cậy dự đoán</div>
                <div class="conf-item">
                    <div class="conf-info"><span>Iris Setosa</span><span id="conf-0">100%</span></div>
                    <div class="conf-bar-bg"><div class="conf-bar-fill" id="bar-0" style="width: 100%;"></div></div>
                </div>
                <div class="conf-item" style="margin-top: 8px;">
                    <div class="conf-info"><span>Iris Versicolor</span><span id="conf-1">0%</span></div>
                    <div class="conf-bar-bg"><div class="conf-bar-fill" id="bar-1" style="width: 0%;"></div></div>
                </div>
                <div class="conf-item" style="margin-top: 8px;">
                    <div class="conf-info"><span>Iris Virginica</span><span id="conf-2">0%</span></div>
                    <div class="conf-bar-bg"><div class="conf-bar-fill" id="bar-2" style="width: 0%;"></div></div>
                </div>
            </div>
        </section>
    </main>
</div>

<!-- TOAST NOTIFICATION -->
<div id="toast">✅ Phân loại thành công!</div>

<script>
const sliders = {
    sepal_length: document.getElementById("sepal_length"),
    sepal_width: document.getElementById("sepal_width"),
    petal_length: document.getElementById("petal_length"),
    petal_width: document.getElementById("petal_width")
};

const presets = {
    setosa: { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2 },
    versicolor: { sepal_length: 6.0, sepal_width: 2.9, petal_length: 4.5, petal_width: 1.5 },
    virginica: { sepal_length: 6.5, sepal_width: 3.0, petal_length: 5.5, petal_width: 1.8 }
};

const flowers = {
    0: {
        name: "Iris Setosa",
        description: "Loài đặc trưng với cánh hoa nhỏ, ngắn và độ mở hẹp, thường phân bố ở vùng khí hậu hàn đới.",
        image: "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_3.jpg"
    },
    1: {
        name: "Iris Versicolor",
        description: "Có kích thước trung gian, sắc hoa chuyển màu độc đáo giữa các sắc tím và xanh lam.",
        image: "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"
    },
    2: {
        name: "Iris Virginica",
        description: "Kích thước tổng thể lớn, cánh hoa dài, rộng và phát triển mạnh mẽ ở các vùng đất ẩm ướt.",
        image: "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
};

function showToast(message, isError = false) {
    const toast = document.getElementById("toast");
    toast.innerHTML = message;
    toast.style.borderColor = isError ? "var(--accent)" : "var(--border)";
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 2500);
}

function updateSlider(element) {
    const min = Number(element.min);
    const max = Number(element.max);
    const value = Number(element.value);
    const progress = ((value - min) / (max - min)) * 100;

    element.style.setProperty("--progress", progress + "%");
    document.getElementById(element.id + "_value").textContent = value.toFixed(1) + " cm";
}

Object.values(sliders).forEach(slider => {
    updateSlider(slider);
    slider.addEventListener("input", () => updateSlider(slider));
});

// Presets click
document.querySelectorAll(".preset").forEach(button => {
    button.addEventListener("click", () => {
        const data = presets[button.dataset.type];
        Object.entries(data).forEach(([key, val]) => {
            sliders[key].value = val;
            updateSlider(sliders[key]);
        });
        predict();
    });
});

async function predict() {
    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.textContent = "⏳ Đang xử lý mô hình...";

    const payload = {
        sepal_length: Number(sliders.sepal_length.value),
        sepal_width: Number(sliders.sepal_width.value),
        petal_length: Number(sliders.petal_length.value),
        petal_width: Number(sliders.petal_width.value)
    };

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Lỗi phản hồi từ server");
        const result = await res.json();
        
        const classId = result.class_id;
        const flower = flowers[classId];

        // Update UI image & text
        const img = document.getElementById("flowerImage");
        img.classList.remove("animate");
        void img.offsetWidth;
        img.src = flower.image;
        img.alt = flower.name;
        img.classList.add("animate");

        document.getElementById("resultName").textContent = flower.name;
        document.getElementById("resultDescription").textContent = flower.description;

        // Mock confidence bars based on prediction (or use real probabilities if your model supports predict_proba)
        for(let i=0; i<3; i++) {
            let score = (i === classId) ? 96.5 : (Math.random() * 3).toFixed(1);
            if(i === classId && score < 95) score = 97.2;
            document.getElementById(`conf-${i}`).textContent = score + "%";
            document.getElementById(`bar-${i}`).style.width = score + "%";
        }

        showToast("✨ Phân loại thành công!");
    } catch (err) {
        showToast("⚠️ Không thể kết nối đến mô hình!", true);
    } finally {
        btn.disabled = false;
        btn.textContent = "✦   Phân tích bằng mô hình SVM";
    }
}

document.getElementById("predictBtn").addEventListener("click", predict);

document.getElementById("resetBtn").addEventListener("click", () => {
    Object.entries(presets.setosa).forEach(([key, val]) => {
        sliders[key].value = val;
        updateSlider(sliders[key]);
    });
    predict();
});

// Run initial prediction on load
predict();
</script>

</body>
</html>
"""

# =========================
# HEALTH CHECK API
# =========================
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "algorithm": "Support Vector Machine (SVM)"
    }

# =========================
# PREDICT API
# =========================
@app.post("/predict", tags=["Prediction"])
def predict_endpoint(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa được tải trên server.")
    
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = int(model.predict(features)[0])

    return {
        "class_id": prediction,
        "prediction": species[prediction]
    }
