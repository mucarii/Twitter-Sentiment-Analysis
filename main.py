# %%[importar as bibliotecas]
import pandas as pd
from textblob import TextBlob
import kagglehub
import os

# %%[importar o dataset]
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None
)

# %%[definindo os nomes das colunas]
df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# %%[verificar o dataset]
print(df.head())

# %%[analise de sentimento]
df["sentiment"] = df["text"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)

# %%[exibir o resultado]
print(df[["text", "sentiment"]].head())
