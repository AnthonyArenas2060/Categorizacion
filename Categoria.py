import pandas as pd
import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import re

st.title("Subir un archivo de Excel")

# Crear el widget para subir archivos
archivo = st.file_uploader("Elige un archivo de Excel", type=["xlsx"])
data = pd.read_excel(archivo) if archivo else None


if "labels" not in st.session_state:
    st.session_state.labels = []

# --- Sección de etiquetas dinámicas ---
st.subheader("📝 Añadir etiquetas para clasificar los comentarios")

nueva_label = st.text_input("Escribe una etiqueta y presiona Enter o el botón para agregarla:")
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("➕ Agregar etiqueta"):
        if nueva_label.strip():
            if nueva_label.strip() not in st.session_state.labels:
                st.session_state.labels.append(nueva_label.strip())
            else:
                st.warning("⚠️ Esa etiqueta ya fue agregada.")
        else:
            st.warning("⚠️ Escribe una etiqueta antes de agregarla.")

with col2:
    if st.button("🗑️ Limpiar etiquetas"):
        st.session_state.labels = []

# Mostrar etiquetas agregadas
if st.session_state.labels:
    st.success(f"Etiquetas actuales: {', '.join(st.session_state.labels)}")
else:
    st.info("Aún no has agregado etiquetas.")

    
if archivo is not None and len(st.session_state.labels) > 0:
    try:
        def limpiar_tweet(texto):
            texto = texto.replace("\n", " ").replace("\\n", " ")
            texto = re.sub(r"^RT\s+@\w+\s+", "", texto)   
            texto = re.sub(r"@\w+", "", texto)            
            texto = re.sub(r"https..*", "", texto)
            texto = texto.strip()                      
            return texto
    
        tweets = data['Full Text'].to_list()
        df = pd.DataFrame({"tweet": tweets})
        df["tweet_limpio"] = df["tweet"].apply(limpiar_tweet)
        coments_gente = df["tweet_limpio"].to_list()

        if st.button("🚀 Ejecutar clasificación"):
            st.info(f"Categorías seleccionadas: {', '.join(st.session_state.labels)}")



        #candidate_labels = [lbl.strip() for lbl in labels_input.split(",") if lbl.strip()]

        #st.info(f"Categorías seleccionadas: {', '.join(candidate_labels)}")
    
            MODELO_ZERO_SHOT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" 
    
            clasificador = pipeline(
                "zero-shot-classification",
                model=MODELO_ZERO_SHOT,
                tokenizer=MODELO_ZERO_SHOT
            )
    
            comments = coments_gente
        
            #candidate_labels = [
            #    "salud (medicamentos)", 
            #    "deporte (fútbol, jugador, defensa, delantero, equipo, gol, partido)",
            #    "Recomendacion"
            #]
        
            results = clasificador(comments, st.session_state.labels)
        
        
            df = pd.DataFrame(results)
        
            df_labels = df['labels'].apply(pd.Series)
            df_scores = df['scores'].apply(pd.Series)
        
            df_labels = df_labels.rename(columns = lambda x : 'label_' + str(x + 1))
            df_scores = df_scores.rename(columns = lambda x : 'score_' + str(x + 1))
        
            df_final = pd.concat([df['sequence'], df_labels, df_scores], axis=1)
        
            df = df_final
        
            melted_data = []
        
            for index, row in df.iterrows():
                if pd.notna(row['label_1']):
                    melted_data.append({'sequence': row['sequence'], 'label': row['label_1'], 'score': row['score_1']})
                if pd.notna(row['label_2']):
                    melted_data.append({'sequence': row['sequence'], 'label': row['label_2'], 'score': row['score_2']})
                if pd.notna(row['label_3']):
                    melted_data.append({'sequence': row['sequence'], 'label': row['label_3'], 'score': row['score_3']})
        
            df_long = pd.DataFrame(melted_data)
        
            df_long = df_long.sort_values('sequence').reset_index(drop=True)
        
            df_long.loc[df_long['sequence'].duplicated(), 'sequence'] = ''
        
        
        
            st.subheader("Categorización de comentarios")
            st.dataframe(df_long)
        
            csv = df_long.to_csv(index=False).encode('utf-8')
            st.markdown("""
                        <style>
                        div.stDownloadButton > button {
                            background-color: #1E90FF;
                            color: white;
                            padding: 0.6em 1.2em;
                            border-radius: 10px;
                            border: none;
                            font-size: 16px;
                            font-weight: bold;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        }
                        div.stDownloadButton > button:hover {
                            background-color: #0073e6;
                            transform: scale(1.05);
                            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                        }
                        </style>
                    """, unsafe_allow_html=True)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv')
        
    
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
