from streamlit_option_menu import option_menu
import streamlit as st
import joblib
import pandas as pd

st.header("PROYEK UAS PPW", divider='rainbow')
text = st.text_area("Masukkan Artikel Berita")

button = st.button("Submit")

if "nb_reduksi" not in st.session_state:
    st.session_state.nb_reduksi = []
    st.session_state.nb_asli = []

if button:
    vectorizer = joblib.load("vectorizer.pkl")
    tfidf_matrics = vectorizer.transform([text]).toarray()
    
    # Predict Model Naive Bayes Reduksi
    model_reduksi = joblib.load("naive bayes.pkl")
    lda = joblib.load("lda.pkl")
    lda_transform = lda.transform(tfidf_matrics)
    prediction_reduksi = model_reduksi.predict(lda_transform)
    st.session_state.nb_reduksi = prediction_reduksi[0]
    
    # Predict Model Naive Bayes Tanpa Reduksi
    model_asli = joblib.load("Naive Bayes (Asli).pkl")
    prediction_asli = model_asli.predict(tfidf_matrics)
    st.session_state.nb_asli = prediction_asli[0]

selected = option_menu(
  menu_title="",
  options=["Dataset Information", "History Uji Coba" ,"Klasifikasi"],
  icons=["data", "Process", "model", "implemen", "Test", "sa"],
  orientation="horizontal"
  )

if selected == "Dataset Information":
    st.write("Dataset Asli")
    st.dataframe(pd.read_csv('3kategori.csv'), use_container_width=True)
    st.write("Dataset Hasil Reduksi Dimensi")
    st.dataframe(pd.read_csv('reduksi dimensi.csv'), use_container_width=True)


elif selected == "Klasifikasi":
  if st.session_state.nb_reduksi:
      nb_lda, nb_NonLDA = st.tabs(["Model Naive Bayes(LDA)", "Model Naive Bayes (Tanpa LDA)"])
      
      with nb_lda:
        st.write(f"Prediction Category : {st.session_state.nb_reduksi}")
        
      with nb_NonLDA:
        st.write(f"Prediction Category : {st.session_state.nb_asli}")
        
elif selected == "History Uji Coba":
    st.write("Hasil Uji Coba")
    st.dataframe(pd.read_csv('history.csv'), use_container_width=True)
