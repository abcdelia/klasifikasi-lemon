import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Klasifikasi Lemon"
)

model = joblib.load("model_klasifikasi_lemon.joblib")

st.title("Belajar Klasifikasi Lemon")
st.markdown("Aplikasi machine learning classification untuk memprediksi kualitas lemon")

diameter = st.slider("Diameter", 40.0, 60.0, 65.5)
berat = st.slider("Berat", 100, 150, 110)
tebal_kulit = st.slider("Tebal Kulit", 3.2, 5.0, 4.8)
kadar_gula = st.slider("Kadar Gula", 8.0, 9.0, 7.5)
asal_daerah = st.pills("Asal Daerah", ["California", "Medan", "Malang"], default="Malang" )
musim_panen = st.pills("Musim Panen", ["Puncak","Akhir"], default="Puncak")
warna = st.pills("Warna", ["Hijau pekat", "Kuning kehijauan", "Kuning cerah"], default="Hijau pekat")


if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,musim_panen,warna]], columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","musim_panen","warna"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat oleh **Adel**")