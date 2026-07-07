import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Prediksi Harga Cabai Rawit Merah",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("model/model_mlr.pkl")
    return model

model = load_model()

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/dataset_cabe_rawit_merah.csv",
        parse_dates=["tanggal_lengkap"]
    )
    return df

df = load_data()

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
padding-left:3rem;
padding-right:3rem;
}

.card{

background:#ffffff;

padding:25px;

border-radius:15px;

box-shadow:0px 4px 12px rgba(0,0,0,0.1);

border-left:6px solid #d62828;

margin-bottom:20px;

}

.big-font{

font-size:40px;

font-weight:bold;

color:#d62828;

}

.sub-font{

font-size:18px;

color:#555555;

}

.metric{

background:#fff5f5;

padding:20px;

border-radius:12px;

text-align:center;

border:1px solid #ffd6d6;

}

.info-box{

background:#fff8f8;

padding:15px;

border-radius:10px;

border-left:5px solid red;

}

</style>
""", unsafe_allow_html=True)

# ==========================================
# NAVBAR
# ==========================================

selected = option_menu(

    menu_title=None,

    options=[
        "Home",
        "Dataset",
        "Visualisasi",
        "Prediksi",
        "Tentang"
    ],

    icons=[
        "house-fill",
        "table",
        "bar-chart-fill",
        "graph-up-arrow",
        "person-fill"
    ],

    orientation="horizontal",

    default_index=0
)
    
# ==========================================
# HOME
# ==========================================

if selected == "Home":

    # =========================
    # Hitung jumlah data
    # =========================

    X = df[['hari_ke', 'lag_1', 'lag_3', 'lag_7', 'ma_7']]
    y = df['cabe_rawit_merah']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # =========================
    # HEADER
    # =========================

    st.markdown("""
    <div class="card">

        <div class="big-font">
            🌶️ Prediksi Harga Cabai Rawit Merah
        </div>

        <div class="sub-font" style="margin-top:10px;">
            Dashboard implementasi <b>Multiple Linear Regression</b> untuk memprediksi
            harga cabai rawit merah di Kabupaten Bekasi berdasarkan data historis.
        </div>

        <div style="margin-top:20px;">

            <span style="
                background:#d62828;
                color:white;
                padding:8px 16px;
                border-radius:20px;
                font-size:14px;
                font-weight:600;
                margin-right:8px;">
                📅 Periode 2023–2025
            </span>

            <span style="
                background:#f1f3f5;
                color:#343a40;
                padding:8px 16px;
                border-radius:20px;
                font-size:14px;
                margin-right:8px;">
                📊 {len(df)} Data
            </span>
    
            <span style="
                background:#f1f3f5;
                color:#343a40;
                padding:8px 16px;
                border-radius:20px;
                font-size:14px;">
                🎯 Target: Harga Cabai Rawit Merah
            </span>
    </div>

    </div>
    """, 
                unsafe_allow_html=True
    )

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "📄 Jumlah Data",
            len(df)
        )

    with col2:
        st.metric(
            "📚 Data Training",
            len(X_train)
        )

    with col3:
        st.metric(
            "🧪 Data Testing",
            len(X_test)
        )

    st.divider()

    st.subheader("📌 Informasi Penelitian")

    st.markdown(f"""
<div style="background:#f8f9fa;
padding:20px;
border-radius:12px;
border-left:6px solid #dc3545;">

<b>Metode</b><br>
Multiple Linear Regression

<hr>

<b>Dataset</b><br>
Open Data Kabupaten Bekasi

<hr>

<b>Target</b><br>
Harga Cabai Rawit Merah

<hr>

<b>Periode</b><br>
2023–2025

</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("📈 Variabel Model")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.success("hari_ke")

    with col2:
        st.success("lag-1")

    with col3:
        st.success("lag-3")

    with col4:
        st.success("lag-7")

    with col5:
        st.success("MA7")

    st.write("")
    st.divider()

    st.subheader("ℹ️ Tentang Dashboard")

    st.info("""
Dashboard ini merupakan implementasi model **Multiple Linear Regression**
yang dibangun untuk memprediksi harga cabai rawit merah di Kabupaten Bekasi.

Model menggunakan lima variabel independen yaitu:

• hari_ke

• lag-1

• lag-3

• lag-7

• Moving Average 7 Hari (MA7)

Prediksi hanya dapat dilakukan pada tanggal yang memiliki data historis
yang diperlukan untuk membentuk variabel tersebut.

Pengembangan selanjutnya dapat dilakukan dengan mengintegrasikan dashboard
dengan sumber data harga harian sehingga pembentukan variabel historis dapat
dilakukan secara otomatis.
""")
    
# ==========================================
# MENU DATASET
# ==========================================

if selected == "Dataset":

    st.title("📊 Dataset")

    st.write(
        """
        Dataset yang digunakan pada penelitian ini merupakan data harga
        cabai rawit merah Kabupaten Bekasi periode tahun 2023–2025 yang
        diperoleh dari Open Data Kabupaten Bekasi.
        """
    )

    st.divider()

# ==========================================
# STATISTIK DATASET
# ==========================================
    col1,col2,col3=st.columns(3)

    with col1:
        st.metric(
            "Jumlah Data",
            len(df)
        )

    with col2:
        st.metric(
            "Jumlah Variabel",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Periode",
            "2023-2025"
        )

# ==========================================
# INFORMASI DATASET
# ==========================================
    st.subheader("Informasi Dataset")

    info=pd.DataFrame({

        "Kolom":df.columns,

        "Tipe Data":df.dtypes.astype(str)

    })

    st.dataframe(
        info,
        use_container_width=True
    )

# ==========================================
# DATASET
# ==========================================
    st.subheader("Data")

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

# ==========================================
# DOWNLOAD DATASET
# ==========================================
    csv=df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="⬇️ Download Dataset",

        data=csv,

        file_name="dataset_cabe_rawit_merah.csv",

        mime="text/csv"

    )

# ==========================================
# MENU VISUALISASI
# ==========================================

if selected=="Visualisasi":

    st.title("📈 Visualisasi Dataset")

    st.write("""

Visualisasi berikut menunjukkan pola harga aktual serta
variabel hasil feature engineering yang digunakan dalam
pembangunan model Multiple Linear Regression.

""")

    st.divider()

# ==========================================
# GRAFIK HARGA AKTUAL
# ==========================================
    st.subheader("Harga Aktual")

    fig,ax=plt.subplots(figsize=(12,4))

    ax.plot(

        df["tanggal_lengkap"],

        df["cabe_rawit_merah"],

        linewidth=2

    )

    ax.set_xlabel("Tanggal")

    ax.set_ylabel("Harga")

    ax.grid(True)

    st.pyplot(fig)

# ==========================================
# LAG-1
# ==========================================
    st.subheader("Lag-1")

    fig,ax=plt.subplots(figsize=(12,4))

    ax.plot(

        df["tanggal_lengkap"],

        df["lag_1"]

    )

    ax.grid(True)

    st.pyplot(fig)

# ==========================================
# LAG-3
# ==========================================
    st.subheader("Lag-3")

    fig,ax=plt.subplots(figsize=(12,4))

    ax.plot(

        df["tanggal_lengkap"],

        df["lag_3"]

    )

    ax.grid(True)

    st.pyplot(fig)

# ==========================================
# LAG-7
# ==========================================
    st.subheader("Lag-7")

    fig,ax=plt.subplots(figsize=(12,4))

    ax.plot(

        df["tanggal_lengkap"],

        df["lag_7"]

    )

    ax.grid(True)

    st.pyplot(fig)

# ==========================================
# MOVING AVERAGE 7
# ==========================================
    st.subheader("Moving Average 7 Hari (MA7)")

    fig,ax=plt.subplots(figsize=(12,4))

    ax.plot(

        df["tanggal_lengkap"],

        df["ma_7"]

    )

    ax.grid(True)

    st.pyplot(fig)

# ==========================================
# STATISTIK DESKRIPTIF
# ==========================================

    st.divider()

    st.subheader("Statistik Deskriptif")

    st.dataframe(

        df.describe(),

        use_container_width=True

    )

# ==========================================
# MENU PREDIKSI
# ==========================================

if selected == "Prediksi":

    st.title("🔮 Prediksi Harga Cabai Rawit Merah")

    st.write("""
    Pilih tanggal yang tersedia pada dataset untuk melakukan prediksi harga
    cabai rawit merah menggunakan model Multiple Linear Regression.
    """)

    st.divider()

    min_date = df["tanggal_lengkap"].min().date()
    max_date = df["tanggal_lengkap"].max().date()

    tanggal = st.date_input(
    "📅 Pilih Tanggal",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)
    data = df[
    df["tanggal_lengkap"].dt.date == tanggal
]
    if data.empty:
        st.error("Data untuk tanggal tersebut tidak tersedia.")
        st.stop()

    data = data.iloc[0]

    hari_ke = data["hari_ke"]

    lag1 = data["lag_1"]

    lag3 = data["lag_3"]

    lag7 = data["lag_7"]

    ma7 = data["ma_7"]

    aktual = data["cabe_rawit_merah"]

    st.subheader("📋 Variabel Model")

    variabel = pd.DataFrame({

        "Variabel":[

            "hari_ke",

            "lag_1",

            "lag_3",

            "lag_7",

            "MA7"

        ],

        "Nilai":[

            hari_ke,

            lag1,

            lag3,

            lag7,

            ma7

        ]

    })

    st.dataframe(
        variabel,
        use_container_width=True
    )

    if st.button("🚀 Prediksi Harga"):

        with st.spinner("Sedang melakukan prediksi..."):

            X = np.array([[

                hari_ke,
                lag1,
                lag3,
                lag7,
                ma7

            ]])

            prediksi = model.predict(X)[0]

        st.success("Prediksi berhasil dilakukan.")

    col1,col2,col3=st.columns(3)
    with col1:
        st.metric(
            "Harga Aktual",
            f"Rp {aktual:,.0f}"
        )

    with col2:
        st.metric(
            "Harga Prediksi",
            f"Rp {prediksi:,.0f}"               
        )

    with col3:
        selisih = prediksi - aktual
        st.metric(
            "Selisih",
            f"{selisih:,.0f}"
        )

    st.subheader("📊 Error Prediksi")

    error = aktual - prediksi
    persen = abs(error)/aktual*100

    col1,col2 = st.columns(2)
    with col1:
        st.metric(
            "Error",
            f"{error:,.0f}"
        )

    with col2:
        st.metric(
            "APE",
            f"{persen:.2f}%"
        )

    st.subheader("📐 Persamaan Model")

    st.latex(r'''
Y=
875.968
+
0.638992(hari\_ke)
+
0.615494(lag_1)
-
0.487867(lag_3)
-
0.229140(lag_7)
+
1.078612(MA7)
''')
            
    st.info("""

**Catatan**

Dashboard ini menggunakan model Multiple Linear Regression dengan
variabel **hari_ke**, **lag-1**, **lag-3**, **lag-7**, dan
**Moving Average 7 Hari (MA7)**.

Prediksi hanya dapat dilakukan pada tanggal yang memiliki data historis
yang diperlukan untuk membentuk variabel tersebut.

Dashboard ini belum dapat melakukan prediksi pada tanggal di luar
rentang data penelitian.

Pengembangan lebih lanjut dapat dilakukan dengan mengintegrasikan
dashboard dengan sumber data harga harian sehingga proses pembentukan
variabel historis dapat dilakukan secara otomatis dan prediksi untuk
hari berikutnya dapat dihasilkan.

""")

# ==========================================
# MENU TENTANG
# ==========================================

if selected == "Tentang":

    st.title("ℹ️ Tentang")

    st.markdown("""
    <div class="card">

    <h2 style="color:#d62828;">
    🌶️ Prediksi Harga Cabai Rawit Merah
    </h2>

    Dashboard ini merupakan implementasi model
    <b>Multiple Linear Regression</b>
    untuk memprediksi harga cabai rawit merah
    di Kabupaten Bekasi berdasarkan data historis.

    </div>
    """, unsafe_allow_html=True)

    st.subheader("👩‍🎓 Profil Peneliti")

    col1, col2 = st.columns([1,2])

    with col1:

        st.markdown("## 👩")

    with col2:

        st.write("**Nama**")
        st.write("Raina Andriani Putri")

        st.write("**NPM**")
        st.write("202210715283")

        st.write("**Universitas**")
        st.write("Universitas Bhayangkara Jakarta Raya")

        st.write("**Program Studi**")
        st.write("Informatika")

    st.subheader("📄 Judul Skripsi")

    st.success("""

Prediksi Harga Cabai Rawit Merah
Menggunakan Metode
Multiple Linear Regression

""")
    
    st.subheader("🧠 Metode")

    st.write("""

Model dibangun menggunakan metode
**Multiple Linear Regression**
dengan variabel:

- hari_ke
- lag-1
- lag-3
- lag-7
- Moving Average 7 Hari (MA7)

""")

    st.subheader("📊 Dataset")

    st.write("""

Sumber data:

Open Data Kabupaten Bekasi

Periode:

2023–2025

""")

    st.subheader("💻 Teknologi")

    st.write("""

Dashboard dibuat menggunakan:

- Python

- Streamlit

- Scikit-learn

- Pandas

- Matplotlib

""")

    st.subheader("⚠️ Keterbatasan Dashboard")

    st.warning("""

Dashboard ini merupakan implementasi model penelitian.

Prediksi hanya dapat dilakukan pada tanggal yang
memiliki data historis yang diperlukan untuk
membentuk variabel:

• hari_ke

• lag-1

• lag-3

• lag-7

• MA7

Dashboard belum dapat melakukan prediksi
untuk tanggal di luar rentang data penelitian.

Pengembangan lebih lanjut dapat dilakukan
dengan mengintegrasikan dashboard dengan
database harga harian sehingga proses
feature engineering dapat dilakukan
secara otomatis.

""")
    
st.divider()

st.markdown(
    """
    <center>

    © 2026

    Prediksi Harga Cabai Rawit Merah

    Multiple Linear Regression

    Universitas Bhayangkara Jakarta Raya

    </center>
    """,
    unsafe_allow_html=True
)
