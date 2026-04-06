# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd

# st.map([{"lat":40, "lon":70}])

# progress_bar = st.progress(0)
# for i in range (100):
#     progress_bar.progress(i+1)

# number = st.number_input("Enter the number")
# text_field= st.text_input("Enter the text")
# st.write("Text field is",number) 

import streamlit as st
import numpy as np

st.title("🧠 Transformer Visualizer")

# Sidebar
text = st.sidebar.text_input("Enter text", "I love AI")

# Tokenization
tokens = text.lower().split()

st.header("🔹 Tokenization")

st.write("Tokens:")
st.write(tokens)

# Create vocab
vocab = list(set(tokens))
word_to_idx = {word: i for i, word in enumerate(vocab)}

st.write("Token to Index Mapping:")
st.write(word_to_idx)

# Convert tokens to indices
indices = [word_to_idx[word] for word in tokens]
st.write("Token Indices:")
st.write(indices)

# Embeddings (random for now)
embedding_dim = 4
embedding_matrix = np.random.rand(len(vocab), embedding_dim)

embeddings = [embedding_matrix[word_to_idx[word]] for word in tokens]

st.write("Embeddings:")
st.write(embeddings)

# Heatmap style view
st.write("Embedding Visualization:")
st.dataframe(embeddings)