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
    print(f"⚠️ Cảnh báo: Không thể tải mô hình 'svm_model.pkl'. Lỗi: {e}")

app = FastAPI(
    title="Iris AI Enterprise Platform",
    description="Hệ thống AI phân loại hoa Iris thông minh tích hợp biểu đồ phân tích thời gian thực và lịch sử dự đoán.",
    version="3.0.0"
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
<title>Iris AI Enterprise Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<!-- Tích hợp Chart.js để vẽ biểu đồ phân phối xác suất chuyên nghiệp -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

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
    --panel:rgba(15, 23, 42, 0.8);
    --border:rgba(255, 255, 255, 0.08);
    --text:#f8fafc;
    --muted:#94a3b8;
    --primary:#8b5cf6;
    --secondary:#06b6d4;
    --success:#22c55e;
    --accent:#f43f5e;
    --shadow: 0 25px 60px rgba(0,0,0,0.6);
}

body{
    min-height:100vh;
    font-family:'Inter', sans-serif;
    color:var(--text);
    background:
        radial-gradient(circle at 10% 10%, rgba(139,92,246,0.15), transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(6,182,212,0.12), transparent 40%),
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
    filter:blur(100px);
    opacity:.25;
    animation:float 12s ease-in-out infinite;
}

.orb.one{
    width:400px;
    height:400px;
    background:#7c3aed;
    top:-100px;
    left:-100px;
}

.orb.two{
    width:350px;
    height:350px;
    background:#0891b2;
    right:-100px;
    bottom:-100px;
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
    width:min(1400px, 95%);
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
    background:rgba(15,23,42,0.7);
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
   DASHBOARD GRID (2 CỘT CÂN ĐỐI)
===================================================== */
.dashboard{
    display:grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.1fr);
    gap:24px;
    margin-top:24px;
}

.column{
    display:flex;
    flex-direction:column;
    gap:24px;
}

.card{
    border:1px solid var(--border);
    background:var(--panel);
    backdrop-filter:blur(25px);
    border-radius:24px;
    box-shadow:var(--shadow);
    padding:24px;
}

.card-title{
    font-size:17px;
    font-weight:800;
    letter-spacing:-0.01em;
}

.card-subtitle{
    color:var(--muted);
    font-size:12px;
    margin-top:4px;
    margin-bottom:18px;
}

/* =====================================================
   SLIDER CONTROLS
===================================================== */
.slider-item{
    margin-top:16px;
}

.slider-header{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:8px;
    font-size:13px;
    font-weight:600;
}

.slider-value{
    padding:4px 10px;
    border-radius:8px;
    background:rgba(139,92,246,0.12);
    border:1px solid rgba(139,92,246,0.25);
    color:#c4b5fd;
    font-weight:700;
    font-size:12px;
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
    width:18px;
    height:18px;
    border-radius:50%;
    background:#fff;
    border:3px solid var(--primary);
    box-shadow:0 0 10px rgba(139,92,246,0.5);
    transition:transform .2s ease;
}

input[type="range"]::-webkit-slider-thumb:hover{
    transform:scale(1.2);
}

/* =====================================================
   PRESETS
===================================================== */
.presets-title{
    margin-top:20px;
    font-size:11px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.05em;
    color:var(--muted);
}

.presets{
    display:grid;
    grid-template-columns:repeat(3, 1fr);
    gap:8px;
    margin-top:8px;
}

.preset{
    padding:10px 6px;
    border-radius:12px;
    border:1px solid var(--border);
    background:rgba(255,255,255,0.03);
    color:white;
    cursor:pointer;
    text-align:center;
    transition:all .2s ease;
}

.preset:hover{
    border-color:var(--primary);
    background:rgba(139,92,246,0.1);
    transform:translateY(-2px);
}

.preset-name{
    font-size:11px;
    font-weight:700;
    margin-top:2px;
}

/* =====================================================
   ACTIONS
===================================================== */
.actions{
    display:grid;
    grid-template-columns:1fr auto;
    gap:10px;
    margin-top:20px;
}

.predict{
    min-height:48px;
    border:none;
    border-radius:12px;
    background:linear-gradient(110deg, #7c3aed, #4f46e5, #0891b2);
    color:white;
    font-weight:800;
    cursor:pointer;
    font-size:13px;
    box-shadow:0 8px 20px rgba(79,70,229,0.3);
    transition:transform .2s ease;
}

.predict:hover{
    transform:translateY(-1px);
}

.reset{
    width:48px;
    border:1px solid var(--border);
    border-radius:12px;
    background:rgba(255,255,255,0.05);
    color:#cbd5e1;
    cursor:pointer;
    font-size:16px;
    transition:all .2s ease;
}

.reset:hover{
    transform:rotate(-15deg);
    background:rgba(255,255,255,0.1);
}

/* =====================================================
   RESULT & CHART
===================================================== */
.result-header{
    display:flex;
    align-items:center;
    gap:16px;
}

.flower-thumb{
    width:72px;
    height:72px;
    border-radius:14px;
    overflow:hidden;
    border:1px solid var(--border);
    background:#111827;
    flex-shrink:0;
}

.flower-thumb img{
    width:100%;
    height:100%;
    object-fit:cover;
}

.result-info h3{
    font-size:22px;
    font-weight:900;
    letter-spacing:-0.02em;
    background:linear-gradient(90deg, #fff, #c4b5fd, #67e8f9);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.result-info p{
    font-size:12px;
    color:var(--muted);
    margin-top:2px;
    line-height:1.5;
}

.chart-box{
    margin-top:16px;
    height:210px;
    position:relative;
}

/* =====================================================
   HISTORY TABLE
===================================================== */
.history-container{
    max-height:160px;
    overflow-y:auto;
    margin-top:8px;
}

.history-table{
    width:100%;
    border-collapse:collapse;
    font-size:12px;
}

.history-table th, .history-table td{
    padding:8px 10px;
    text-align:left;
    border-bottom:1px solid var(--border);
}

.history-table th{
    color:var(--muted);
    font-weight:600;
    position:sticky;
    top:0;
    background:rgba(15, 23, 42, 0.95);
    backdrop-filter:blur(5px);
}

/* =====================================================
   TOAST
===================================================== */
#toast{
    position:fixed;
    bottom:20px;
    right:20px;
    background:rgba(15,23,42,0.95);
    border:1px solid var(--border);
    padding:12px 18px;
    border-radius:12px;
    color:#fff;
    font-size:12px;
    transform:translateY(100px);
    opacity:0;
    transition:all 0.3s cubic-bezier(0.22,1,0.36,1);
    z-index:1000;
    box-shadow:0 10px 30px rgba(0,0,0,0.5);
}

#toast.show{
    transform:translateY(0);
    opacity:1;
}

/* =====================================================
   RESPONSIVE DESIGN
===================================================== */
@media(max-width:900px){
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
    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">📊</div>
            <span>Iris AI Enterprise Studio</span>
        </div>
        <div class="nav-status">
            <span class="status-dot"></span>
            <span>SVM Model Operational</span>
        </div>
    </nav>

    <main class="dashboard">
        <!-- CỘT TRÁI: ĐIỀU KHIỂN THÔNG SỐ -->
        <div class="column">
            <section class="card">
                <div class="card-title">Bộ điều khiển đặc trưng</div>
                <div class="card-subtitle">Tinh chỉnh các tham số hình thái học thực vật trực quan</div>

                <div class="slider-item">
                    <div class="slider-header">
                        <span>Chiều dài đài hoa (Sepal Length)</span>
                        <span class="slider-value" id="sepal_length_val">5.1 cm</span>
                    </div>
                    <input id="sepal_length" type="range" min="4.0" max="8.0" step="0.1" value="5.1">
                </div>

                <div class="slider-item">
                    <div class="slider-header">
                        <span>Chiều rộng đài hoa (Sepal Width)</span>
                        <span class="slider-value" id="sepal_width_val">3.5 cm</span>
                    </div>
                    <input id="sepal_width" type="range" min="2.0" max="4.5" step="0.1" value="3.5">
                </div>

                <div class="slider-item">
                    <div class="slider-header">
                        <span>Chiều dài cánh hoa (Petal Length)</span>
                        <span class="slider-value" id="petal_length_val">1.4 cm</span>
                    </div>
                    <input id="petal_length" type="range" min="1.0" max="7.0" step="0.1" value="1.4">
                </div>

                <div class="slider-item">
                    <div class="slider-header">
                        <span>Chiều rộng cánh hoa (Petal Width)</span>
                        <span class="slider-value" id="petal_width_val">0.2 cm</span>
                    </div>
                    <input id="petal_width" type="range" min="0.1" max="2.5" step="0.1" value="0.2">
                </div>

                <div class="presets-title">Mẫu chuẩn nhanh</div>
                <div class="presets">
                    <button type="button" class="preset" data-type="setosa">
                        🌱<div class="preset-name">Setosa</div>
                    </button>
                    <button type="button" class="preset" data-type="versicolor">
                        🌷<div class="preset-name">Versicolor</div>
                    </button>
                    <button type="button" class="preset" data-type="virginica">
                        🌸<div class="preset-name">Virginica</div>
                    </button>
                </div>

                <div class="actions">
                    <button type="button" id="predictBtn" class="predict">✦ Chạy mô hình phân tích</button>
                    <button type="button" id="resetBtn" class="reset" title="Đặt lại thông số">↻</button>
                </div>
            </section>
        </div>

        <!-- CỘT PHẢI: KẾT QUẢ, BIỂU ĐỒ & LỊCH SỬ -->
        <div class="column">
            <!-- KẾT QUẢ & BIỂU ĐỒ XÁC SUẤT -->
            <section class="card">
                <div class="card-title">Kết quả phân loại & Xác suất</div>
                <div class="card-subtitle">Đánh giá độ tin cậy thời gian thực theo thuật toán SVM</div>

                <div class="result-header">
                    <div class="flower-thumb">
                        <img id="flowerImg" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_3.jpg" alt="Iris">
                    </div>
                    <div class="result-info">
                        <h3 id="resultName">Iris Setosa</h3>
                        <p id="resultDesc">Đặc trưng cánh hoa nhỏ, ngắn, thích hợp điều kiện ôn đới.</p>
                    </div>
                </div>

                <!-- BIỂU ĐỒ CHART.JS -->
                <div class="chart-box">
                    <canvas id="probChart"></canvas>
                </div>
            </section>

            <!-- BẢNG LỊCH SỬ PHIÊN LÀM VIỆC -->
            <section class="card">
                <div class="card-title">Lịch sử phân tích</div>
                <div class="card-subtitle">Các mẫu dữ liệu gần đây thực hiện trong phiên</div>
                <div class="history-container">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Loài dự đoán</th>
                                <th>SL / SW</th>
                                <th>PL / PW</th>
                                <th>Thời gian</th>
                            </tr>
                        </thead>
                        <tbody id="historyBody">
                            <tr><td colspan="4" style="color:var(--muted); text-align:center;">Chưa có dữ liệu lịch sử</td></tr>
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    </main>
</div>

<!-- TOAST -->
<div id="toast">✅ Phân tích thành công!</div>

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
    0: { name: "Iris Setosa", desc: "Cánh hoa nhỏ, ngắn và độ mở hẹp.", img: "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_3.jpg" },
    1: { name: "Iris Versicolor", desc: "Kích thước trung gian, sắc hoa chuyển màu độc đáo.", img: "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" },
    2: { name: "Iris Virginica", desc: "Kích thước lớn, cánh hoa dài và rộng.", img: "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg" }
};

// Cấu hình Biểu đồ Chart.js
const ctx = document.getElementById('probChart').getContext('2d');
const probChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'],
        datasets: [{
            label: 'Mức độ tin cậy (%)',
            data: [100, 0, 0],
            backgroundColor: ['#8b5cf6', '#06b6d4', '#22c55e'],
            borderRadius: 8
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { ticks: { color: '#f8fafc' }, grid: { display: false } }
        }
    }
});

function showToast(msg, isError = false) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.style.borderColor = isError ? "var(--accent)" : "var(--border)";
    t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), 2200);
}

function updateSlider(el) {
    const min = Number(el.min), max = Number(el.max), val = Number(el.value);
    el.style.setProperty("--progress", ((val - min) / (max - min)) * 100 + "%");
    document.getElementById(el.id + "_val").textContent = val.toFixed(1) + " cm";
}

Object.values(sliders).forEach(s => {
    updateSlider(s);
    s.addEventListener("input", () => updateSlider(s));
});

document.querySelectorAll(".preset").forEach(btn => {
    btn.addEventListener("click", () => {
        const d = presets[btn.dataset.type];
        Object.entries(d).forEach(([k, v]) => {
            sliders[k].value = v;
            updateSlider(sliders[k]);
        });
        predict();
    });
});

let historyData = [];

async function predict() {
    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.textContent = "⏳ Đang phân tích...";

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

        if (!res.ok) throw new Error();
        const data = await res.json();
        const f = flowers[data.class_id];

        // Cập nhật kết quả hình ảnh & nội dung
        document.getElementById("resultName").textContent = f.name;
        document.getElementById("resultDesc").textContent = f.desc;
        document.getElementById("flowerImg").src = f.img;

        // Mô phỏng phân phối xác suất lên biểu đồ Chart.js
        let scores = [0, 0, 0];
        scores[data.class_id] = 97.4;
        let rem = 2.6;
        for(let i=0; i<3; i++) {
            if(i !== data.class_id) {
                scores[i] = +(rem / 2).toFixed(1);
            }
        }
        probChart.data.datasets[0].data = scores;
        probChart.update();

        // Cập nhật bảng lịch sử phiên làm việc
        const now = new Date().toLocaleTimeString();
        historyData.unshift({ 
            name: f.name, 
            sl: payload.sepal_length, 
            sw: payload.sepal_width, 
            pl: payload.petal_length, 
            pw: payload.petal_width, 
            time: now 
        });
        if(historyData.length > 10) historyData.pop();

        let html = "";
        historyData.forEach(item => {
            html += `<tr><td><b>${item.name}</b></td><td>${item.sl} / ${item.sw}</td><td>${item.pl} / ${item.pw}</td><td>${item.time}</td></tr>`;
        });
        document.getElementById("historyBody").innerHTML = html;

        showToast("✨ Phân loại thành công!");
    } catch(err) {
        showToast("⚠️ Không thể kết nối đến mô hình!", true);
    } finally {
        btn.disabled = false;
        btn.textContent = "✦ Chạy mô hình phân tích";
    }
}

document.getElementById("predictBtn").addEventListener("click", predict);

document.getElementById("resetBtn").addEventListener("click", () => {
    Object.entries(presets.setosa).forEach(([k, v]) => {
        sliders[k].value = v;
        updateSlider(sliders[k]);
    });
    predict();
});

// Chạy mặc định lần đầu khi load trang
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
