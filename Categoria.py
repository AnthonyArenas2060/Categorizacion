import pandas as pd
import streamlit as st
import requests
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import re

st.title("Subir un archivo de Excel")
# Crear el widget para subir archivos
archivo = st.file_uploader("Elige un archivo de Excel", type=["xlsx"])
data = pd.read_excel(archivo) if archivo else None

if archivo is not None:
    try:
        tweets = data['Full Text'].to_list()
        df = pd.DataFrame({"tweet": tweets})
        def limpiar_tweet(texto):
            texto = texto.replace("\n", " ").replace("\\n", " ")
            texto = re.sub(r"^RT\s+@\w+\s+", "", texto)
            texto = re.sub(r"@\w+", "", texto)
            texto = re.sub(r"https..*", "", texto)
            texto = texto.strip()
            return texto
        df["tweet_limpio"] = df["tweet"].apply(limpiar_tweet)
        coments_gente = df["tweet_limpio"].to_list()
        st.dataframe(coments_gente)
        
        MODELO_ZERO_SHOT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" 
    
        #clasificador = pipeline("zero-shot-classification",model=MODELO_ZERO_SHOT,tokenizer=MODELO_ZERO_SHOT)
    
        #comments = coments_gente
    
        #candidate_labels = ["salud (medicamentos)", "deporte (fútbol, jugador, defensa, delantero, equipo, gol, partido)", "Recomendacion"]
    
        #results = clasificador(comments, candidate_labels)
        #df = pd.DataFrame(results)
        #st.dataframe(df)
    
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

