import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# ===============================================
# PAGE CONFIG
# ===============================================
st.set_page_config(
    page_title="Klasifikasi Aduan Masyarakat",
    page_icon="📢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================================
# CUSTOM CSS
# ===============================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 1rem 0;
    }
    .pinalti {
        background-color: #f5d5e0;
        border-color: #e83e8c;
        color: #6d1f3e;
    }
    .darurat {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .prioritas {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .umum {
        background-color: #d1ecf1;
        border-color: #17a2b8;
        color: #0c5460;
    }
    .lainnya {
        background-color: #e2e3e5;
        border-color: #6c757d;
        color: #383d41;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ===============================================
# LOAD MODEL (dengan caching)
# ===============================================
@st.cache_resource
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        st.info("🔄 Loading model from Hugging Face Hub...")
        model_name = "Zulkifli1409/aduan-model"

        # load tokenizer dan config
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 5  # pastikan sesuai jumlah kelas

        # load model (sekarang otomatis baca model.safetensors)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            use_safetensors=True,
            ignore_mismatched_sizes=True
        )

        model = model.to(device)
        model.eval()

        st.success("✅ Model IndoBERT berhasil dimuat dari Hugging Face!")
        return model, tokenizer, device, "huggingface"

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

# ===============================================
# PREDIKSI FUNCTION - BASIC
# ===============================================
def predict_aduan(text, model, tokenizer, device):
    """Prediksi kategori aduan - 5 kelas"""
    # 5 labels sesuai training
    label_map = {
        0: "PINALTI",
        1: "DARURAT", 
        2: "PRIORITAS", 
        3: "UMUM", 
        4: "LAINNYA"
    }

    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    result = {
        "label": label_map[pred_idx],
        "confidence": confidence * 100,
        "all_probs": {
            "PINALTI": probs[0].item() * 100,
            "DARURAT": probs[1].item() * 100,
            "PRIORITAS": probs[2].item() * 100,
            "UMUM": probs[3].item() * 100,
            "LAINNYA": probs[4].item() * 100,
        },
    }

    return result


# ===============================================
# PREDIKSI FUNCTION - ADVANCED
# ===============================================
def predict_aduan_advanced(text, model, tokenizer, device):
    """Prediksi dengan informasi tambahan (token analysis, attention)"""
    result = predict_aduan(text, model, tokenizer, device)
    
    # Tokenize untuk analisis
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    
    # Hitung statistik
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits
        
        # Temperature scaling untuk confidence calibration
        temperature = 1.5
        probs_calibrated = torch.nn.functional.softmax(logits / temperature, dim=1)[0]
    
    # Advanced metrics
    result["advanced"] = {
        "text_length": len(text),
        "token_count": attention_mask.sum().item(),
        "top_2_labels": sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)[:2],
        "confidence_gap": max(result["all_probs"].values()) - sorted(result["all_probs"].values())[-2],
        "entropy": -sum([p/100 * torch.log(torch.tensor(p/100 + 1e-10)) for p in result["all_probs"].values()]).item(),
        "calibrated_confidence": probs_calibrated[torch.argmax(probs_calibrated)].item() * 100,
    }
    
    return result


# ===============================================
# PREDIKSI FUNCTION - EXPERT
# ===============================================
def predict_aduan_expert(text, model, tokenizer, device):
    """Prediksi dengan analisis mendalam dan rekomendasi"""
    result = predict_aduan_advanced(text, model, tokenizer, device)
    
    # Expert analysis
    expert_info = {
        "urgency_level": get_urgency_level(result),
        "recommended_action": get_recommended_action(result),
        "confidence_assessment": assess_confidence(result),
        "potential_misclassification": check_misclassification_risk(result),
        "content_flags": analyze_content_flags(text, result),
    }
    
    result["expert"] = expert_info
    
    return result


def get_urgency_level(result):
    """Tentukan level urgensi berdasarkan kategori"""
    urgency_map = {
        "PINALTI": "⚠️ IMMEDIATE MODERATION",
        "DARURAT": "🚨 CRITICAL - IMMEDIATE ACTION",
        "PRIORITAS": "⚡ HIGH - QUICK RESPONSE",
        "UMUM": "📋 NORMAL - STANDARD PROCESS",
        "LAINNYA": "📌 LOW - GENERAL HANDLING"
    }
    return urgency_map.get(result["label"], "UNKNOWN")


def get_recommended_action(result):
    """Rekomendasi tindakan berdasarkan kategori"""
    actions = {
        "PINALTI": "Block/review content immediately. Alert moderation team. Check for policy violations.",
        "DARURAT": "Dispatch emergency response team. Alert relevant authorities. Provide immediate assistance.",
        "PRIORITAS": "Create ticket with high priority. Assign to relevant department within 24 hours.",
        "UMUM": "Route to appropriate department. Standard response time applies.",
        "LAINNYA": "Review and categorize manually if needed. Standard handling procedures."
    }
    return actions.get(result["label"], "Contact supervisor for guidance.")


def assess_confidence(result):
    """Evaluasi kualitas confidence"""
    conf = result["confidence"]
    gap = result["advanced"]["confidence_gap"]
    
    if conf >= 90 and gap >= 50:
        return "🟢 VERY HIGH - Prediction is highly reliable"
    elif conf >= 80 and gap >= 30:
        return "🟡 HIGH - Prediction is reliable"
    elif conf >= 70 and gap >= 20:
        return "🟠 MODERATE - Consider manual review"
    else:
        return "🔴 LOW - Manual review recommended"


def check_misclassification_risk(result):
    """Deteksi risiko misklasifikasi"""
    conf = result["confidence"]
    entropy = result["advanced"]["entropy"]
    gap = result["advanced"]["confidence_gap"]
    
    risks = []
    
    if conf < 75:
        risks.append("Low confidence score")
    if entropy > 1.2:
        risks.append("High entropy (uncertain prediction)")
    if gap < 20:
        risks.append("Small gap between top predictions")
    
    if not risks:
        return "✅ Low risk - Prediction appears solid"
    else:
        return "⚠️ " + ", ".join(risks)


def analyze_content_flags(text, result):
    """Analisis konten untuk flag tambahan"""
    flags = []
    
    text_lower = text.lower()
    
    # Check for emergency keywords
    emergency_keywords = ["kebakaran", "banjir", "gempa", "kecelakaan", "darurat", "meninggal", "mati"]
    if any(kw in text_lower for kw in emergency_keywords):
        flags.append("🚨 Contains emergency keywords")
    
    # Check for profanity indicators (basic)
    profanity_indicators = ["tolol", "bodoh", "bangsat", "kampret", "brengsek", "sialan"]
    if any(kw in text_lower for kw in profanity_indicators):
        flags.append("⚠️ May contain profanity")
    
    # Check length
    if len(text) < 10:
        flags.append("⚠️ Very short text")
    elif len(text) > 500:
        flags.append("📝 Long text")
    
    # Check if mostly uppercase (shouting)
    if sum(1 for c in text if c.isupper()) > len(text) * 0.5:
        flags.append("📢 Excessive caps (shouting)")
    
    return flags if flags else ["✅ No special flags detected"]


# ===============================================
# VISUALISASI
# ===============================================
def create_probability_chart(probs):
    """Bar chart untuk probabilitas - 5 kelas"""
    labels = list(probs.keys())
    values = list(probs.values())

    # Warna sesuai kategori (5 warna)
    colors = ["#e83e8c", "#dc3545", "#ffc107", "#17a2b8", "#6c757d"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.2f}%" for v in values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Probabilitas Setiap Kategori",
        xaxis_title="Kategori",
        yaxis_title="Probabilitas (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False,
    )

    return fig


def create_confidence_gauge(confidence):
    """Gauge chart untuk confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 90], 'color': "lightgreen"},
                {'range': [90, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


# ===============================================
# MAIN APP
# ===============================================
def main():
    # Header
    st.markdown(
        '<div class="main-header">📢 Sistem Klasifikasi Aduan Masyarakat</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Powered by IndoBERT - Deteksi otomatis tingkat urgensi aduan dengan Content Moderation</div>',
        unsafe_allow_html=True,
    )

    # Load model
    with st.spinner("🔄 Loading model..."):
        model, tokenizer, device, source = load_model()

    if model is None:
        st.error("❌ Gagal memuat model. Pastikan koneksi internet stabil.")
        return

    st.success(f"✅ Model berhasil dimuat dari Hugging Face! (Device: {device})")

    # Sidebar - Info
    with st.sidebar:
        st.header("ℹ️ Informasi")
        st.markdown(
            """
        **Kategori Aduan:**
        - 🟣 **PINALTI**: Konten mengandung kata kasar, SARA, pornografi, atau ujaran kebencian
        - 🔴 **DARURAT**: Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)
        - 🟡 **PRIORITAS**: Perlu penanganan cepat (infrastruktur rusak, kebersihan)
        - 🔵 **UMUM**: Informasi/pertanyaan umum
        - ⚫ **LAINNYA**: Aduan lain yang tidak termasuk kategori di atas
        
        ---
        **Mode Prediksi:**
        - **Basic**: Prediksi standar
        - **Advanced**: Dengan analisis detail
        - **Expert**: Rekomendasi lengkap
        
        ---
        **Akurasi Model:**
        - Overall: 96.10%
        - Pinalti: 96.45%
        - Darurat: 96.03%
        - Prioritas: 96.75%
        - Umum: 95.93%
        - Lainnya: 95.00%
        """
        )

        st.divider()

        # Statistics
        if "history" in st.session_state and st.session_state.history:
            st.subheader("📊 Statistik Sesi")
            df_hist = pd.DataFrame(st.session_state.history)

            st.metric("Total Prediksi", len(df_hist))

            # Count by category
            cat_counts = df_hist["Kategori"].value_counts()
            for cat, count in cat_counts.items():
                emoji_map = {
                    "PINALTI": "🟣",
                    "DARURAT": "🔴",
                    "PRIORITAS": "🟡",
                    "UMUM": "🔵",
                    "LAINNYA": "⚫"
                }
                emoji = emoji_map.get(cat, "⚪")
                st.metric(f"{emoji} {cat}", count)

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Prediksi Tunggal", "📋 Prediksi Batch", "📜 Riwayat", "📊 Analisis"]
    )

    # ===============================================
    # TAB 1: PREDIKSI TUNGGAL
    # ===============================================
    with tab1:
        # Mode selection
        mode = st.radio(
            "Pilih Mode Prediksi:",
            ["Basic", "Advanced", "Expert"],
            horizontal=True,
            help="Basic: Prediksi standar | Advanced: Dengan analisis detail | Expert: Rekomendasi lengkap"
        )
        
        st.divider()
        
        st.subheader("Masukkan Aduan")

        # Text input
        text_input = st.text_area(
            "Teks Aduan:",
            height=150,
            placeholder="Contoh: Ada kebakaran besar di pasar tradisional...",
            help="Masukkan teks aduan yang ingin diklasifikasi",
        )

        # Predict button
        if st.button("🚀 Klasifikasi Aduan", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("⚠️ Mohon masukkan teks aduan terlebih dahulu!")
            else:
                with st.spinner("🔄 Menganalisis aduan..."):
                    # Pilih fungsi prediksi sesuai mode
                    if mode == "Basic":
                        result = predict_aduan(text_input, model, tokenizer, device)
                    elif mode == "Advanced":
                        result = predict_aduan_advanced(text_input, model, tokenizer, device)
                    else:  # Expert
                        result = predict_aduan_expert(text_input, model, tokenizer, device)

                # Display result
                st.divider()

                # Result box
                box_class = result["label"].lower()
                emoji_map = {
                    "PINALTI": "🟣",
                    "DARURAT": "🔴",
                    "PRIORITAS": "🟡",
                    "UMUM": "🔵",
                    "LAINNYA": "⚫"
                }
                emoji = emoji_map[result["label"]]

                st.markdown(
                    f"""
                <div class="result-box {box_class}">
                    <h2>{emoji} Kategori: {result['label']}</h2>
                    <h3>Confidence: {result['confidence']:.2f}%</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Warning untuk PINALTI
                if result["label"] == "PINALTI":
                    st.markdown(
                        """
                        <div class="warning-box">
                            <strong>⚠️ CONTENT MODERATION ALERT</strong><br>
                            Konten ini terdeteksi mengandung kata kasar, SARA, atau konten yang melanggar norma.
                            Diperlukan review manual dan tindakan moderasi.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Metrics (5 kolom)
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("🟣 PINALTI", f"{result['all_probs']['PINALTI']:.2f}%")
                with col2:
                    st.metric("🔴 DARURAT", f"{result['all_probs']['DARURAT']:.2f}%")
                with col3:
                    st.metric("🟡 PRIORITAS", f"{result['all_probs']['PRIORITAS']:.2f}%")
                with col4:
                    st.metric("🔵 UMUM", f"{result['all_probs']['UMUM']:.2f}%")
                with col5:
                    st.metric("⚫ LAINNYA", f"{result['all_probs']['LAINNYA']:.2f}%")

                # Charts
                col_chart1, col_chart2 = st.columns([2, 1])
                
                with col_chart1:
                    st.plotly_chart(
                        create_probability_chart(result["all_probs"]),
                        use_container_width=True,
                    )
                
                with col_chart2:
                    st.plotly_chart(
                        create_confidence_gauge(result["confidence"]),
                        use_container_width=True,
                    )

                # Advanced/Expert Information
                if mode == "Advanced" or mode == "Expert":
                    st.divider()
                    st.subheader("📊 Analisis Lanjutan")
                    
                    adv_col1, adv_col2, adv_col3 = st.columns(3)
                    
                    with adv_col1:
                        st.metric("Panjang Teks", result["advanced"]["text_length"])
                        st.metric("Jumlah Token", result["advanced"]["token_count"])
                    
                    with adv_col2:
                        st.metric("Confidence Gap", f"{result['advanced']['confidence_gap']:.2f}%")
                        st.metric("Entropy", f"{result['advanced']['entropy']:.4f}")
                    
                    with adv_col3:
                        st.metric("Calibrated Conf.", f"{result['advanced']['calibrated_confidence']:.2f}%")
                    
                    # Top 2 predictions
                    st.markdown("**Top 2 Predictions:**")
                    for i, (label, prob) in enumerate(result["advanced"]["top_2_labels"], 1):
                        st.write(f"{i}. {label}: {prob:.2f}%")

                # Expert Information
                if mode == "Expert":
                    st.divider()
                    st.subheader("🎯 Expert Analysis")
                    
                    exp_col1, exp_col2 = st.columns(2)
                    
                    with exp_col1:
                        st.markdown("**Urgency Level:**")
                        st.info(result["expert"]["urgency_level"])
                        
                        st.markdown("**Confidence Assessment:**")
                        st.info(result["expert"]["confidence_assessment"])
                        
                        st.markdown("**Misclassification Risk:**")
                        st.info(result["expert"]["potential_misclassification"])
                    
                    with exp_col2:
                        st.markdown("**Recommended Action:**")
                        st.success(result["expert"]["recommended_action"])
                        
                        st.markdown("**Content Flags:**")
                        for flag in result["expert"]["content_flags"]:
                            st.write(f"- {flag}")

                # Save to history
                st.session_state.history.append(
                    {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Teks": (
                            text_input[:50] + "..."
                            if len(text_input) > 50
                            else text_input
                        ),
                        "Kategori": result["label"],
                        "Confidence": f"{result['confidence']:.2f}%",
                        "Mode": mode,
                    }
                )

    # ===============================================
    # TAB 2: PREDIKSI BATCH
    # ===============================================
    with tab2:
        st.subheader("Prediksi Multiple Aduan Sekaligus")

        # Option 1: Text area
        st.markdown("**Opsi 1: Masukkan beberapa aduan (satu per baris)**")
        batch_text = st.text_area(
            "Aduan (pisahkan dengan enter):",
            height=200,
            placeholder="Ada kebakaran di rumah warga\nJalan berlubang perlu diperbaiki\nMohon info jadwal posyandu",
        )

        # Option 2: File upload
        st.markdown("**Opsi 2: Upload file (.txt atau .csv)**")
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["txt", "csv"],
            help="Format: satu aduan per baris (txt) atau kolom 'teks_aduan' (csv)",
        )

        if st.button("🚀 Prediksi Semua", type="primary"):
            texts = []

            # Process batch_text
            if batch_text.strip():
                texts.extend(
                    [line.strip() for line in batch_text.split("\n") if line.strip()]
                )

            # Process uploaded file
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".txt"):
                    texts.extend(
                        [
                            line.decode("utf-8").strip()
                            for line in uploaded_file
                            if line.strip()
                        ]
                    )
                elif uploaded_file.name.endswith(".csv"):
                    df_upload = pd.read_csv(uploaded_file)
                    if "teks_aduan" in df_upload.columns:
                        texts.extend(df_upload["teks_aduan"].dropna().tolist())
                    else:
                        st.error("❌ File CSV harus memiliki kolom 'teks_aduan'")

            if not texts:
                st.warning("⚠️ Tidak ada teks yang dimasukkan!")
            else:
                st.info(f"📊 Memproses {len(texts)} aduan...")

                results = []
                progress_bar = st.progress(0)

                for i, text in enumerate(texts):
                    result = predict_aduan(text, model, tokenizer, device)
                    results.append(
                        {
                            "No": i + 1,
                            "Teks": text[:50] + "..." if len(text) > 50 else text,
                            "Kategori": result["label"],
                            "Confidence": f"{result['confidence']:.2f}%",
                            "Teks_Lengkap": text,
                        }
                    )
                    progress_bar.progress((i + 1) / len(texts))

                # Display results
                df_results = pd.DataFrame(results)

                st.success(f"✅ Selesai! {len(texts)} aduan telah diklasifikasi")

                # Summary (5 kolom)
                col1, col2, col3, col4, col5 = st.columns(5)

                pinalti_count = (df_results["Kategori"] == "PINALTI").sum()
                darurat_count = (df_results["Kategori"] == "DARURAT").sum()
                prioritas_count = (df_results["Kategori"] == "PRIORITAS").sum()
                umum_count = (df_results["Kategori"] == "UMUM").sum()
                lainnya_count = (df_results["Kategori"] == "LAINNYA").sum()

                with col1:
                    st.metric("🟣 PINALTI", pinalti_count)
                with col2:
                    st.metric("🔴 DARURAT", darurat_count)
                with col3:
                    st.metric("🟡 PRIORITAS", prioritas_count)
                with col4:
                    st.metric("🔵 UMUM", umum_count)
                with col5:
                    st.metric("⚫ LAINNYA", lainnya_count)

                # Warning jika ada PINALTI
                if pinalti_count > 0:
                    st.warning(f"⚠️ Ditemukan {pinalti_count} aduan yang memerlukan moderasi konten!")

                # Display table
                st.dataframe(
                    df_results[["No", "Teks", "Kategori", "Confidence"]],
                    use_container_width=True,
                    hide_index=True,
                )

                # Download button
                csv = df_results.to_csv(index=False, encoding="utf-8")
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv,
                    file_name=f"hasil_klasifikasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

    # ===============================================
    # TAB 3: RIWAYAT
    # ===============================================
    with tab3:
        st.subheader("📜 Riwayat Prediksi")

        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)

            # Display dataframe
            st.dataframe(df_history, use_container_width=True, hide_index=True)

            # Download button
            csv = df_history.to_csv(index=False, encoding="utf-8")
            col1, col2 = st.columns([1, 5])

            with col1:
                st.download_button(
                    label="📥 Download",
                    data=csv,
                    file_name=f"riwayat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            with col2:
                if st.button("🗑️ Hapus Riwayat", type="secondary"):
                    st.session_state.history = []
                    st.rerun()
        else:
            st.info(
                "👋 Belum ada riwayat prediksi. Mulai klasifikasi aduan untuk melihat riwayat!"
            )

    # ===============================================
    # TAB 4: ANALISIS
    # ===============================================
    with tab4:
        st.subheader("📊 Analisis Dashboard")

        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)

            # Overview metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Prediksi", len(df_history))
            
            with col2:
                avg_conf = df_history["Confidence"].str.rstrip("%").astype(float).mean()
                st.metric("Rata-rata Confidence", f"{avg_conf:.2f}%")
            
            with col3:
                most_common = df_history["Kategori"].mode()[0]
                st.metric("Kategori Terbanyak", most_common)

            st.divider()

            # Distribution chart
            st.subheader("📈 Distribusi Kategori")
            
            category_counts = df_history["Kategori"].value_counts()
            
            # Pie chart
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribusi Kategori Aduan",
                color_discrete_sequence=["#e83e8c", "#dc3545", "#ffc107", "#17a2b8", "#6c757d"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Bar chart by mode
            if "Mode" in df_history.columns:
                st.divider()
                st.subheader("🎯 Distribusi Berdasarkan Mode Prediksi")
                
                mode_counts = df_history.groupby(["Kategori", "Mode"]).size().reset_index(name="Count")
                
                fig_mode = px.bar(
                    mode_counts,
                    x="Kategori",
                    y="Count",
                    color="Mode",
                    title="Kategori per Mode Prediksi",
                    barmode="group"
                )
                st.plotly_chart(fig_mode, use_container_width=True)

            # Timeline
            st.divider()
            st.subheader("⏱️ Timeline Prediksi")
            
            df_history["Timestamp"] = pd.to_datetime(df_history["Timestamp"])
            df_history["Hour"] = df_history["Timestamp"].dt.hour
            
            hourly_counts = df_history.groupby("Hour").size().reset_index(name="Count")
            
            fig_timeline = px.line(
                hourly_counts,
                x="Hour",
                y="Count",
                title="Aktivitas Prediksi per Jam",
                markers=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Confidence distribution
            st.divider()
            st.subheader("📊 Distribusi Confidence Score")
            
            df_history["Confidence_Numeric"] = df_history["Confidence"].str.rstrip("%").astype(float)
            
            fig_conf = px.histogram(
                df_history,
                x="Confidence_Numeric",
                nbins=20,
                title="Distribusi Confidence Score",
                labels={"Confidence_Numeric": "Confidence (%)"},
                color_discrete_sequence=["#1f77b4"]
            )
            st.plotly_chart(fig_conf, use_container_width=True)

            # High priority alerts
            st.divider()
            st.subheader("⚠️ High Priority Alerts")
            
            pinalti_df = df_history[df_history["Kategori"] == "PINALTI"]
            darurat_df = df_history[df_history["Kategori"] == "DARURAT"]
            
            alert_col1, alert_col2 = st.columns(2)
            
            with alert_col1:
                st.markdown("**🟣 PINALTI (Content Moderation)**")
                if len(pinalti_df) > 0:
                    st.error(f"⚠️ {len(pinalti_df)} aduan memerlukan moderasi")
                    st.dataframe(
                        pinalti_df[["Timestamp", "Teks", "Confidence"]].head(5),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success("✅ Tidak ada konten yang perlu dimoderasi")
            
            with alert_col2:
                st.markdown("**🔴 DARURAT (Emergency)**")
                if len(darurat_df) > 0:
                    st.error(f"🚨 {len(darurat_df)} situasi darurat")
                    st.dataframe(
                        darurat_df[["Timestamp", "Teks", "Confidence"]].head(5),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success("✅ Tidak ada situasi darurat")

            # Export summary
            st.divider()
            st.subheader("📥 Export Analisis")
            
            summary_data = {
                "Total Prediksi": len(df_history),
                "Rata-rata Confidence": f"{avg_conf:.2f}%",
                "Kategori Terbanyak": most_common,
                "PINALTI": len(pinalti_df),
                "DARURAT": len(darurat_df),
                "PRIORITAS": len(df_history[df_history["Kategori"] == "PRIORITAS"]),
                "UMUM": len(df_history[df_history["Kategori"] == "UMUM"]),
                "LAINNYA": len(df_history[df_history["Kategori"] == "LAINNYA"]),
            }
            
            summary_json = json.dumps(summary_data, indent=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Summary (JSON)",
                    data=summary_json,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
            
            with col2:
                st.download_button(
                    label="📥 Download Full Data (CSV)",
                    data=df_history.to_csv(index=False, encoding="utf-8"),
                    file_name=f"full_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        else:
            st.info(
                "👋 Belum ada data untuk dianalisis. Mulai klasifikasi aduan terlebih dahulu!"
            )
            
            # Demo visualization
            st.divider()
            st.subheader("📊 Contoh Dashboard (Demo)")
            
            demo_data = {
                "Kategori": ["PINALTI", "DARURAT", "PRIORITAS", "UMUM", "LAINNYA"],
                "Count": [12, 45, 78, 123, 34]
            }
            
            fig_demo = px.bar(
                demo_data,
                x="Kategori",
                y="Count",
                title="Contoh Distribusi Kategori",
                color="Kategori",
                color_discrete_sequence=["#e83e8c", "#dc3545", "#ffc107", "#17a2b8", "#6c757d"]
            )
            st.plotly_chart(fig_demo, use_container_width=True)

    # Footer
    st.divider()
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <small>🤖 Sistem Klasifikasi Aduan Masyarakat v2.0 | Powered by IndoBERT & Streamlit</small><br>
        <small>⚠️ Disclaimer: Model ini adalah alat bantu. Untuk konten sensitif, selalu lakukan review manual.</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ===============================================
# RUN APP
# ===============================================
if __name__ == "__main__":
    main()
