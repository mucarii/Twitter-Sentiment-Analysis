# %%[importar as bibliotecas]
import pandas as pd
from textblob import TextBlob
import kagglehub
import os

# %%[importar o dataset]
# Download latest version
path = kagglehub.dataset_download("kazanova/sentiment140")

# Definindo o caminho completo para o arquivo CSV
file_path = os.path.join(path, "training.1600000.processed.noemoticon.csv")

# Lendo o dataset com a codificação ISO-8859-1
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# %%[definindo os nomes das colunas]
df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# %%[verificar o dataset]
print(df.head())

# %%[analise de sentimento]
df["sentiment"] = df["text"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)

# %%[exibir o resultado]
print(df[["text", "sentiment"]].head())
