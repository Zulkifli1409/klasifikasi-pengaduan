import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ===============================================
# LOAD MODEL (dengan caching)
# ===============================================
@st.cache_resource
def load_model():
    """Load model dan tokenizer (cached)"""
    try:
        model_name = "indobenchmark/indobert-base-p1"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            use_safetensors=True,
        )

        # Load trained weights
        state_dict = load_file("best_model.safetensors", device=str(device))
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None


# ===============================================
# PREDIKSI FUNCTION
# ===============================================
def predict_aduan(text, model, tokenizer, device):
    """Prediksi kategori aduan"""
    label_map = {0: "DARURAT", 1: "PRIORITAS", 2: "UMUM"}

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
            "DARURAT": probs[0].item() * 100,
            "PRIORITAS": probs[1].item() * 100,
            "UMUM": probs[2].item() * 100,
        },
    }

    return result


# ===============================================
# VISUALISASI
# ===============================================
def create_probability_chart(probs):
    """Buat bar chart untuk probabilitas"""
    labels = list(probs.keys())
    values = list(probs.values())

    # Warna sesuai kategori
    colors = ["#dc3545", "#ffc107", "#17a2b8"]

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
        '<div class="sub-header">Powered by IndoBERT - Deteksi otomatis tingkat urgensi aduan</div>',
        unsafe_allow_html=True,
    )

    # Load model
    with st.spinner("⏳ Loading model..."):
        model, tokenizer, device = load_model()

    if model is None:
        st.error(
            "❌ Gagal memuat model. Pastikan file 'best_model.safetensors' ada di folder yang sama."
        )
        return

    st.success(f"✅ Model berhasil dimuat! (Device: {device})")

    # Sidebar - Info
    with st.sidebar:
        st.header("ℹ️ Informasi")
        st.markdown(
            """
        **Kategori Aduan:**
        - 🔴 **DARURAT**: Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)
        - 🟡 **PRIORITAS**: Perlu penanganan cepat (infrastruktur rusak, kebersihan)
        - 🔵 **UMUM**: Informasi/pertanyaan umum
        
        ---
        **Cara Pakai:**
        1. Pilih mode di tab
        2. Masukkan teks aduan
        3. Klik tombol prediksi
        4. Lihat hasil klasifikasi
        """
        )

        st.divider()

        # Statistics
        if "history" in st.session_state and st.session_state.history:
            st.subheader("📊 Statistik")
            df_hist = pd.DataFrame(st.session_state.history)

            st.metric("Total Prediksi", len(df_hist))

            # Count by category
            cat_counts = df_hist["Kategori"].value_counts()
            for cat, count in cat_counts.items():
                emoji = (
                    "🔴" if cat == "DARURAT" else "🟡" if cat == "PRIORITAS" else "🔵"
                )
                st.metric(f"{emoji} {cat}", count)

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Prediksi Tunggal", "📋 Prediksi Batch", "📜 Riwayat"]
    )

    # ===============================================
    # TAB 1: PREDIKSI TUNGGAL
    # ===============================================
    with tab1:
        st.subheader("Masukkan Aduan")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Text input
            text_input = st.text_area(
                "Teks Aduan:",
                height=150,
                placeholder="Contoh: Ada kebakaran besar di pasar tradisional...",
                help="Masukkan teks aduan yang ingin diklasifikasi",
            )

        with col2:
            st.markdown("**Contoh Aduan:**")
            examples = [
                "Ada kebakaran besar di pasar",
                "Jalan berlubang perlu diperbaiki",
                "Mohon info jadwal posyandu",
                "Tolong ada orang kecelakaan parah",
                "Sampah menumpuk di jalan",
            ]

            for i, ex in enumerate(examples):
                if st.button(f"📌 Contoh {i+1}", key=f"ex_{i}"):
                    st.session_state.example_text = ex

            if "example_text" in st.session_state:
                text_input = st.session_state.example_text
                del st.session_state.example_text
                st.rerun()

        # Predict button
        if st.button("🚀 Klasifikasi Aduan", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("⚠️ Mohon masukkan teks aduan terlebih dahulu!")
            else:
                with st.spinner("🔄 Menganalisis aduan..."):
                    result = predict_aduan(text_input, model, tokenizer, device)

                # Display result
                st.divider()

                # Result box
                box_class = result["label"].lower()
                emoji = (
                    "🔴"
                    if result["label"] == "DARURAT"
                    else "🟡" if result["label"] == "PRIORITAS" else "🔵"
                )

                st.markdown(
                    f"""
                <div class="result-box {box_class}">
                    <h2>{emoji} Kategori: {result['label']}</h2>
                    <h3>Confidence: {result['confidence']:.2f}%</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🔴 DARURAT", f"{result['all_probs']['DARURAT']:.2f}%")

                with col2:
                    st.metric(
                        "🟡 PRIORITAS", f"{result['all_probs']['PRIORITAS']:.2f}%"
                    )

                with col3:
                    st.metric("🔵 UMUM", f"{result['all_probs']['UMUM']:.2f}%")

                # Chart
                st.plotly_chart(
                    create_probability_chart(result["all_probs"]),
                    use_container_width=True,
                )

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

                # Summary
                col1, col2, col3 = st.columns(3)

                darurat_count = (df_results["Kategori"] == "DARURAT").sum()
                prioritas_count = (df_results["Kategori"] == "PRIORITAS").sum()
                umum_count = (df_results["Kategori"] == "UMUM").sum()

                with col1:
                    st.metric("🔴 DARURAT", darurat_count)
                with col2:
                    st.metric("🟡 PRIORITAS", prioritas_count)
                with col3:
                    st.metric("🔵 UMUM", umum_count)

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

    # Footer
    st.divider()
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <small>🤖 Sistem Klasifikasi Aduan Masyarakat | Powered by IndoBERT & Streamlit</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ===============================================
# RUN APP
# ===============================================
if __name__ == "__main__":
    main()
