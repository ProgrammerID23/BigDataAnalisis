# Klasifikasi Lagu Berdasarkan Mood

## Pendahuluan
Musik memiliki kemampuan unik untuk membangkitkan emosi dan menciptakan suasana hati tertentu. Dalam era digital, data musik yang kaya memungkinkan kita untuk memahami dan mengklasifikasikan lagu berdasarkan mood secara otomatis. Proyek ini bertujuan untuk mengembangkan model machine learning yang mampu mengelompokkan lagu ke dalam kategori mood tertentu, seperti "Bahagia", "Sedih", "Santai", atau "Bersemangat", dengan memanfaatkan data atribut audio dari Spotify.

Aplikasi dari sistem ini mencakup:
- Pembuatan playlist otomatis
- Rekomendasi musik
- Analisis tren musik

## Pemahaman Data

### Sumber Data
Dataset diambil dari Spotify dan mencakup 5.000 lagu dari berbagai genre dan periode. Dataset ini berisi atribut audio seperti:
- **Tempo**: Kecepatan lagu (dalam BPM).
- **Energi**: Tingkat intensitas lagu (skala 0-1).
- **Valensi**: Tingkat positif atau negatif emosional lagu (skala 0-1).
- **Danceability**: Kemudahan lagu untuk digunakan menari.
- **Acousticness**: Probabilitas bahwa lagu bersifat akustik.

### Analisis Data Awal
Proyek ini dimulai dengan eksplorasi data untuk memahami distribusi variabel, outlier, dan korelasi antar atribut. Berikut adalah beberapa temuan utama:
- Valensi dan energi menunjukkan korelasi yang kuat dengan mood lagu.
- Tempo memiliki variasi distribusi yang signifikan antar genre.

#### Visualisasi Distribusi Variabel
![image](https://github.com/user-attachments/assets/1e036c9b-e2e8-4ba5-90e5-1d11dd2940a1)

![image](https://github.com/user-attachments/assets/1d8ede80-7ed3-4c97-bc02-3df1b4cb8418)

![image](https://github.com/user-attachments/assets/67e078a1-3b7a-43a5-8e0f-f0d93282cb0e)




#### Korelasi Antar Variabel
![image](https://github.com/user-attachments/assets/a0664b63-e69e-4f20-9c63-ba501a7a8de7)

### Temuan Utama dari Analisis Data
#### Party Suitability by Genre
- Genre seperti **Latin** dan **Rap** memiliki kelayakan pesta (party suitability) tertinggi, terutama pada kategori tempo "Moderate".
- **EDM** menunjukkan kelayakan pesta tinggi pada kategori tempo "Fast", mencerminkan sifat energik dan cocok untuk aktivitas intens.

#### Distribusi Tempo dalam Genre
- Lagu-lagu genre seperti **Pop**, **R&B**, dan **Rock** sebagian besar berada dalam kategori tempo "Moderate".
- Genre **Rap** menunjukkan distribusi yang seimbang antara kategori tempo "Fast" dan "Moderate", mencerminkan fleksibilitas dalam suasana lagu.
- Genre seperti **EDM** memiliki proporsi lagu "Fast" yang lebih tinggi dibandingkan genre lain.

#### Korelasi Antar Fitur
- "Party Suitability" sangat berkorelasi dengan **danceability** dan **energy**, menegaskan pentingnya kombinasi kedua fitur ini dalam menciptakan suasana pesta.
- **Tempo** tidak menunjukkan korelasi tinggi dengan fitur lain, menjadikannya atribut independen yang penting untuk personalisasi rekomendasi lagu.

## Model yang Dapat Dibangun
Berdasarkan wawasan dari analisis data, berikut adalah model yang dapat dikembangkan:

### Sistem Rekomendasi Berbasis Konten (Content-Based Recommendation)
- **Deskripsi**: Menggunakan fitur seperti "party_suitability", "tempo_category", "danceability", dan "energy" untuk merekomendasikan lagu yang sesuai dengan preferensi pengguna, seperti genre, suasana, atau tempo tertentu.
- **Alasan**: Wawasan dari analisis menunjukkan bahwa fitur-fitur ini sangat relevan dalam menentukan kelayakan lagu untuk aktivitas tertentu seperti pesta atau relaksasi.

### Sistem Klasifikasi Mood
- **Deskripsi**: Membuat model klasifikasi untuk memprediksi mood lagu berdasarkan kombinasi fitur audio seperti valensi, energi, danceability, dan tempo.
- **Alasan**: Analisis menunjukkan bahwa valensi dan energi memiliki hubungan kuat dengan mood lagu.

### Fungsi Subkelompok
Untuk memberikan pemahaman yang lebih dalam, data dikelompokkan ke dalam subkategori berdasarkan:

#### Genre Musik
Analisis per genre membantu mengidentifikasi pola mood unik, seperti:
- Musik **pop** cenderung memiliki valensi dan energi tinggi.
- Musik **klasik** menunjukkan tingkat acousticness yang lebih tinggi.

#### Tahun Rilis
Lagu-lagu yang dirilis pada dekade tertentu dibandingkan untuk memahami evolusi karakteristik audio dari waktu ke waktu.

### Preprocessing Data
Proses preprocessing mencakup beberapa langkah utama:
1. **Penanganan Nilai Hilang**: Data numerik seperti valensi, energi, danceability, tempo, dan durasi diisi dengan nilai rata-rata kolom untuk memastikan konsistensi.
2. **Rekayasa Fitur (Feature Engineering)**:
   - **Kategorisasi Tempo**: Tempo dikategorikan menjadi tiga jenis (Lambat, Sedang, Cepat) berdasarkan skala normalisasi.
   - **Keselarasan untuk Pesta**: Fitur baru "party_suitability" dibuat dengan mengalikan energi dan danceability untuk mencerminkan kesesuaian lagu dalam suasana pesta.
   - **Kategorisasi Mood**: Mood diklasifikasikan menjadi empat kategori utama (Happy, Energetic, Calm, Sad) berdasarkan kombinasi valensi dan energi.
3. **Encoding Variabel Kategorikal**:
   - Variabel "playlist_genre" diencoding menggunakan One-Hot Encoding untuk menghasilkan representasi vektor yang sesuai.
   - Variabel "tempo_category" diencoding menggunakan Label Encoding untuk menghasilkan nilai numerik.
4. **Seleksi Fitur**: Fitur yang dipilih meliputi atribut numerik utama seperti valensi, energi, danceability, party_suitability, kategori tempo, dan hasil encoding genre.
5. **Pemisahan Data**: Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20.

## Pemodelan

### Hyperparameter Tuning dengan Random Forest
Proses berikut digunakan untuk melakukan tuning hyperparameter dan evaluasi model Random Forest:

1. **Inisialisasi Model**: Random Forest diinisialisasi dengan pengaturan default.
2. **Definisi Grid Hyperparameter**: Parameter yang diuji meliputi:
   - Jumlah estimator (`n_estimators`): 50, 100, 150.
   - Kedalaman maksimum (`max_depth`): None, 10, 20, 30.
   - Minimum sampel untuk split (`min_samples_split`): 2, 5, 10.
   - Minimum sampel untuk leaf (`min_samples_leaf`): 1, 2, 4.
3. **Grid Search**: Dilakukan pencarian menggunakan `GridSearchCV` dengan validasi silang sebanyak 5 kali untuk menemukan kombinasi parameter terbaik.
4. **Pelatihan Model Terbaik**: Model terbaik dilatih ulang pada data latih.
5. **Evaluasi Model**:
   - Akurasi dihitung pada data uji.
   - Laporan klasifikasi ditampilkan untuk menunjukkan performa model pada setiap kelas.
6. **Visualisasi Hasil**:
   - Hasil grid search dirangkum untuk melihat kombinasi parameter terbaik.
   - Matriks kebingungan divisualisasikan untuk mengevaluasi kesalahan klasifikasi.

#### Hasil Evaluasi Random Forest
- **Akurasi**: Random Forest memberikan hasil akurasi terbaik dengan kombinasi hyperparameter yang dioptimalkan.

#### Classification Report
```plaintext
              precision    recall  f1-score   support

        Calm       1.00      1.00      1.00       719
   Energetic       1.00      1.00      1.00      2115
       Happy       1.00      1.00      1.00      2714
         Sad       1.00      1.00      1.00      1019

    accuracy                           1.00      6567
   macro avg       1.00      1.00      1.00      6567
weighted avg       1.00      1.00      1.00      6567
```

#### Confusion Matrix
![image](https://github.com/user-attachments/assets/6e31b678-404a-4e40-9b59-cca3a7fa05b5)


## Kesimpulan
Proyek ini berhasil membangun sistem klasifikasi lagu berdasarkan mood dengan tingkat akurasi yang sempurna (100%). Hasil evaluasi menunjukkan bahwa model Random Forest mampu mengklasifikasikan setiap kategori mood tanpa kesalahan pada data uji. Matriks kebingungan juga memperkuat performa ini, dengan semua prediksi berada pada kelas yang benar.

### Rekomendasi Lanjutan
- **Pemanfaatan pada Data Lebih Besar**: Mengaplikasikan model ini pada dataset yang lebih besar dan lebih beragam untuk menguji kemampuan generalisasi.
- **Integrasi dengan Metadata Musik**: Menambahkan metadata lain seperti artis, album, atau popularitas untuk meningkatkan kemampuan rekomendasi.
- **Analisis Lirik Lagu**: Menggunakan analisis teks pada lirik untuk menambahkan dimensi baru dalam penentuan mood.
- **Implementasi Sistem Rekomendasi**: Menggunakan hasil model untuk membangun sistem rekomendasi berbasis suasana hati (mood-based music recommendation).

## Cara Menjalankan Proyek
1. Clone repository ini:
   ```bash
   git clone https://github.com/ProgrammerID23/BigDataAnalisis.git
   ```
2. Instal dependensi:
   ```bash
   pip install -r requiremens.txt
   ```
3. Jalankan notebook:
   ```bash
   jupyter notebook Bigdata_Spotify.ipynb
   ```
## Nama Kelompok

**1.** Muchamad Naufal Aziz(202110370311348)

**2.** Yoga Alisyahbana (202110370311356)
