# Importando as bibliotecas que vamos usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import linear_kernel

# Lendo o arquivo CSV com os filmes
df = pd.read_csv("data/disney_plus_titles.csv", encoding="utf-8")

# Tirando linhas que não têm título ou descrição
df = df.dropna(subset=["title", "description"]).drop_duplicates(subset=["title"]).reset_index(drop=True)

# Função para separar os campos que têm listas (tipo atores, diretores, etc.)
def separar_lista(x):
    if pd.isna(x) or not isinstance(x, str):
        return []
    return [i.strip() for i in x.split(",") if i.strip()]

# Aplicando a função nos campos que são listas
for col in ["director", "cast", "listed_in", "country"]:
    df[col] = df[col].apply(separar_lista)

# Transformando as descrições dos filmes em números (vetores TF-IDF)
vetorizador = TfidfVectorizer(stop_words="english", max_features=10000)
matriz_tfidf = vetorizador.fit_transform(df["description"].astype(str))

# Agrupando os filmes em 100 grupos parecidos (clusters)
kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)
df["cluster"] = kmeans.fit_predict(matriz_tfidf)

# Criando o grafo (rede) com os filmes e suas conexões
G = nx.Graph()

for idx, row in df.iterrows():
    filme = row["title"]
    G.add_node(filme, tipo="filme")

    # Ligando o filme aos atores, diretores, categorias e países
    for ator in row["cast"]:
        G.add_node(ator, tipo="ator")
        G.add_edge(filme, ator)

    for diretor in row["director"]:
        G.add_node(diretor, tipo="diretor")
        G.add_edge(filme, diretor)

    for categoria in row["listed_in"]:
        G.add_node(categoria, tipo="categoria")
        G.add_edge(filme, categoria)

    for pais in row["country"]:
        G.add_node(pais, tipo="pais")
        G.add_edge(filme, pais)

    # Ligando o filme aos 5 mais parecidos (pela descrição)
    similaridades = linear_kernel(matriz_tfidf[idx], matriz_tfidf).flatten()
    similaridades[idx] = -np.inf  # ignora ele mesmo
    top_similares = similaridades.argsort()[-5:]
    for i in top_similares:
        filme_similar = df.iloc[i]["title"]
        G.add_edge(filme, filme_similar, tipo="similar")

# Função para recomendar filmes usando Adamic-Adar
def recomendar(filme, top_n=5):
    candidatos = [n for n, dados in G.nodes(data=True) if dados.get("tipo") == "filme" and n != filme]
    pares = [(filme, c) for c in candidatos]
    pontuacoes = nx.adamic_adar_index(G, pares)
    ordenados = sorted(pontuacoes, key=lambda x: x[2], reverse=True)
    return [b for a, b, score in ordenados[:top_n]]

# Função para desenhar o subgrafo do filme
def desenhar_subgrafo(filme):
    # Adiciona os filmes recomendados como vizinhos
    recomendados = recomendar(filme)
    for r in recomendados:
        G.add_node(r, tipo="recomendado")
        G.add_edge(filme, r, tipo="recomendado")

    # Pega todos os vizinhos (inclusive os recomendados)
    vizinhos = list(G.neighbors(filme)) + [filme]
    subG = G.subgraph(vizinhos)

    # Define cores por tipo
    cores = []
    for n in subG.nodes:
        tipo = G.nodes[n].get("tipo", "filme")
        if tipo == "filme": cores.append("#76b7b2")
        elif tipo == "ator": cores.append("#f28e2c")
        elif tipo == "diretor": cores.append("#4e79a7")
        elif tipo == "categoria": cores.append("#e15759")
        elif tipo == "pais": cores.append("#59a14f")
        elif tipo == "recomendado": cores.append("#d62728")  # vermelho para recomendados
        else: cores.append("#9c9c9c")

    # Desenha o grafo
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subG, seed=42)
    nx.draw(subG, pos, with_labels=True, node_color=cores, font_size=8, node_size=700)
    plt.title(f"Subgrafo de: {filme}")
    plt.tight_layout()
    plt.show()

# Mostrando os filmes disponíveis para o usuário escolher
print("\nAlguns filmes disponíveis no catálogo:")
print(df["title"].sample(10, random_state=42).to_list())

# Pedindo para o usuário digitar o nome de um filme
nome = input("\nDigite o nome exato de um filme ou série para receber recomendações:\n> ")

# Verificando se o filme existe no grafo
if nome in G.nodes:
    print(f"\nTop-5 recomendações para: {nome}")
    for rec in recomendar(nome):
        print("-", rec)
    desenhar_subgrafo(nome)
else:
    print("\nFilme não encontrado no catálogo. Verifique se o nome está certinho.")
