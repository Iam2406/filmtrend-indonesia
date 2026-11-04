
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from urllib.parse import quote

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="FilmTrend Indonesia",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# LOKASI DATASET DAN MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "dataset_film_indonesia_master.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "filmtrend_random_forest.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "genre_encoder.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "model_features.pkl"
)

MEDIAN_PATH = os.path.join(
    BASE_DIR, "median_popularity.pkl"
)

# ==========================================
# MEMUAT DATA DAN MODEL
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURE_PATH)
    median = joblib.load(MEDIAN_PATH)

    return model, encoder, features, median

df = load_data()
model, encoder, features, median = load_model()

# Pastikan kolom numerik dapat diproses
for col in [
    "release_year",
    "vote_average",
    "vote_count",
    "popularity"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎬 FilmTrend Indonesia")

menu = st.sidebar.radio(
    "Menu Utama",
    [
        "Dashboard",
        "Analisis Tren Film",
        "Prediksi Popularitas",
        "Rekomendasi Film",
        "Data Film"
    ]
)

# ==========================================
# JUDUL APLIKASI
# ==========================================

st.title("🎬 FilmTrend Indonesia")

st.caption(
    "Analisis Tren Pasar Film Indonesia "
    "Menggunakan Random Forest"
)

# ==========================================
# DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.header("Dashboard")

    total_film = len(df)

    film_rating = len(
        df[df["vote_count"] > 0]
    )

    film_tanpa_rating = total_film - film_rating

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Film", f"{total_film:,}")
    col2.metric("Film Memiliki Rating", f"{film_rating:,}")
    col3.metric("Film Belum Memiliki Rating", f"{film_tanpa_rating:,}")

    st.divider()

    st.subheader("Tren Jumlah Film Berdasarkan Tahun")

    tren = (
        df.groupby("release_year")
        .size()
        .reset_index(name="jumlah_film")
        .sort_values("release_year")
    )

    fig = px.line(
        tren,
        x="release_year",
        y="jumlah_film",
        markers=True,
        labels={
            "release_year": "Tahun",
            "jumlah_film": "Jumlah Film"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ANALISIS TREN FILM
# ==========================================

elif menu == "Analisis Tren Film":

    st.header("Analisis Tren Film")

    st.subheader("Distribusi Genre")

    genre_data = df[
        df["genres"].notna()
    ].copy()

    genre_data["genre"] = genre_data["genres"].str.split(",")

    genre_data = genre_data.explode("genre")

    genre_data["genre"] = genre_data["genre"].str.strip()

    genre_count = (
        genre_data.groupby("genre")
        .size()
        .reset_index(name="jumlah")
        .sort_values("jumlah", ascending=False)
        .head(10)
    )

    fig_genre = px.bar(
        genre_count,
        x="jumlah",
        y="genre",
        orientation="h",
        title="10 Genre Film Terbanyak"
    )

    st.plotly_chart(fig_genre, use_container_width=True)

    st.subheader("Distribusi Rating")

    df_rating = df[
        (df["vote_count"] > 0) &
        (df["vote_average"] > 0)
    ]

    fig_rating = px.histogram(
        df_rating,
        x="vote_average",
        nbins=20,
        title="Distribusi Rating Film"
    )

    st.plotly_chart(fig_rating, use_container_width=True)

    st.subheader("10 Film dengan Popularitas Tertinggi")

    top_film = df.nlargest(10, "popularity")

    fig_popular = px.bar(
        top_film,
        x="popularity",
        y="title",
        orientation="h"
    )

    st.plotly_chart(fig_popular, use_container_width=True)

    st.info(
        "Popularitas merupakan metrik TMDB, "
        "bukan jumlah penonton bioskop."
    )

# ==========================================
# PREDIKSI POPULARITAS
# ==========================================

elif menu == "Prediksi Popularitas":

    st.header("Prediksi Popularitas Film")

    st.write(
        "Masukkan karakteristik film untuk "
        "memperoleh prediksi kategori popularitas."
    )

    tahun = st.number_input(
        "Tahun Rilis",
        min_value=1900,
        max_value=2100,
        value=2026
    )

    rating = st.number_input(
        "Rating Film",
        min_value=0.1,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    vote_count = st.number_input(
        "Jumlah Vote",
        min_value=1,
        value=100
    )

    daftar_genre = list(encoder.classes_)

    genre_pilihan = st.multiselect(
        "Pilih Genre Film",
        daftar_genre
    )

    if st.button("Prediksi Popularitas"):

        if not genre_pilihan:
            st.warning("Pilih minimal satu genre.")

        else:

            genre_encoded = encoder.transform(
                [genre_pilihan]
            )[0]

            input_data = pd.DataFrame(
                [[tahun, rating, vote_count]],
                columns=[
                    "release_year",
                    "vote_average",
                    "vote_count"
                ]
            )

            genre_input = pd.DataFrame(
                [genre_encoded],
                columns=[
                    f"genre_{g}"
                    for g in encoder.classes_
                ]
            )

            input_final = pd.concat(
                [input_data, genre_input],
                axis=1
            )

            input_final = input_final.reindex(
                columns=features,
                fill_value=0
            )

            prediksi = model.predict(input_final)[0]

            probabilitas = model.predict_proba(
                input_final
            )[0]

            if prediksi == 1:
                st.success("Hasil Prediksi: Popular")
            else:
                st.warning("Hasil Prediksi: Kurang Popular")

            st.write(
                "Probabilitas kelas Popular:",
                f"{probabilitas[1] * 100:.2f}%"
            )

            st.caption(
                "Hasil ini merupakan prediksi model berdasarkan "
                "pola data TMDB, bukan jaminan popularitas aktual."
            )

# ==========================================
# REKOMENDASI FILM
# ==========================================

elif menu == "Rekomendasi Film":

    st.header("🎬 Rekomendasi Film Indonesia")

    st.write(
        "Temukan film berdasarkan genre, rating, "
        "tahun rilis, dan preferensi tontonan."
    )

    st.divider()

    # Mengambil genre yang tersedia
    genre_tersedia = set()

    for daftar in df["genres"].dropna():
        for genre in str(daftar).split(","):
            genre = genre.strip()

            if genre and genre != "Tidak diketahui":
                genre_tersedia.add(genre)

    genre_tersedia = sorted(genre_tersedia)

    # Mengambil rentang tahun
    tahun_valid = df["release_year"].dropna()

    tahun_min = int(tahun_valid.min())
    tahun_max = int(tahun_valid.max())

    col1, col2 = st.columns(2)

    with col1:

        genre_rekomendasi = st.multiselect(
            "Pilih Genre Film",
            genre_tersedia,
            help="Pilih satu atau beberapa genre film."
        )

        rating_minimum = st.slider(
            "Rating Minimum",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.5
        )

    with col2:

        rentang_tahun = st.slider(
            "Rentang Tahun Rilis",
            min_value=tahun_min,
            max_value=tahun_max,
            value=(max(tahun_min, tahun_max - 10), tahun_max)
        )

        preferensi = st.selectbox(
            "Urutkan Rekomendasi Berdasarkan",
            [
                "Popularitas Tertinggi",
                "Rating Tertinggi",
                "Film Terbaru"
            ]
        )

    jumlah_rekomendasi = st.selectbox(
        "Jumlah Rekomendasi",
        [5, 10, 15, 20],
        index=1
    )

    if st.button("🔍 Cari Rekomendasi Film", use_container_width=True):

        hasil = df.copy()

        # Filter tahun
        hasil = hasil[
            hasil["release_year"].between(
                rentang_tahun[0],
                rentang_tahun[1]
            )
        ]

        # Filter rating
        hasil = hasil[
            (hasil["vote_average"] >= rating_minimum) &
            (hasil["vote_count"] > 0)
        ]

        # Filter genre
        if genre_rekomendasi:

            def cocok_genre(genres):
                if pd.isna(genres):
                    return False

                daftar = [
                    g.strip()
                    for g in str(genres).split(",")
                ]

                return any(
                    genre in daftar
                    for genre in genre_rekomendasi
                )

            hasil = hasil[
                hasil["genres"].apply(cocok_genre)
            ]

        # Pengurutan hasil
        if preferensi == "Popularitas Tertinggi":

            hasil = hasil.sort_values(
                "popularity",
                ascending=False
            )

        elif preferensi == "Rating Tertinggi":

            hasil = hasil.sort_values(
                ["vote_average", "vote_count"],
                ascending=[False, False]
            )

        elif preferensi == "Film Terbaru":

            hasil = hasil.sort_values(
                "release_year",
                ascending=False
            )

        hasil = hasil.head(jumlah_rekomendasi)

        st.divider()

        st.subheader("Hasil Rekomendasi")

        st.write(
            f"Ditemukan **{len(hasil)} film** "
            "sesuai kriteria pilihan."
        )

        if hasil.empty:

            st.warning(
                "Belum ditemukan film yang sesuai. "
                "Coba turunkan rating minimum atau "
                "perluas rentang tahun."
            )

        else:

            for _, film in hasil.iterrows():

                with st.container(border=True):

                    col_gambar, col_info = st.columns(
                        [1, 3]
                    )

                    # Poster film jika tersedia
                    with col_gambar:

                        poster = film.get(
                            "poster_path",
                            None
                        )

                        if (
                            isinstance(poster, str)
                            and poster.startswith("/")
                        ):

                            st.image(
                                "https://image.tmdb.org/t/p/w500"
                                + poster,
                                use_container_width=True
                            )

                        else:

                            st.write("🎬 Poster tidak tersedia")

                    with col_info:

                        st.subheader(
                            str(film["title"])
                        )

                        st.write(
                            f"**Tahun:** "
                            f"{int(film['release_year'])}"
                            if pd.notna(film["release_year"])
                            else "**Tahun:** Tidak diketahui"
                        )

                        st.write(
                            f"**Genre:** {film['genres']}"
                        )

                        st.write(
                            f"**Rating TMDB:** "
                            f"{film['vote_average']:.1f}/10"
                        )

                        st.write(
                            f"**Jumlah Vote:** "
                            f"{int(film['vote_count']):,}"
                        )

                        st.write(
                            f"**Popularitas:** "
                            f"{film['popularity']:.2f}"
                        )

                        # Membuat tautan detail TMDB
                        film_id = film.get("id", None)

                        if pd.notna(film_id):

                            link_film = (
                                "https://www.themoviedb.org/movie/"
                                + str(int(film_id))
                            )

                        else:

                            link_film = (
                                "https://www.themoviedb.org/search?query="
                                + quote(str(film["title"]))
                            )

                        st.markdown(
                            f"[🔗 Lihat Detail Film di TMDB]({link_film})"
                        )

        st.caption(
            "Rekomendasi disusun berdasarkan dataset TMDB "
            "dan preferensi yang dipilih. Tautan TMDB "
            "merupakan halaman informasi film, bukan "
            "jaminan ketersediaan layanan streaming."
        )

# ==========================================
# DATA FILM
# ==========================================

elif menu == "Data Film":

    st.header("Data Film Indonesia")

    pencarian = st.text_input(
        "Cari judul film"
    )

    data_tampil = df.copy()

    if pencarian:

        data_tampil = data_tampil[
            data_tampil["title"].str.contains(
                pencarian,
                case=False,
                na=False
            )
        ]

    st.write(
        f"Jumlah hasil: {len(data_tampil)} film"
    )

    st.dataframe(
        data_tampil[
            [
                "title",
                "release_year",
                "genres",
                "vote_average",
                "vote_count",
                "popularity"
            ]
        ],
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "FilmTrend Indonesia | Proyek Analisis Data "
    "dan Prediksi Popularitas Film"
)
