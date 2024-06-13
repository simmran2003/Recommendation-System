import streamlit as st
import pandas as pd
from PIL import Image  # To load images
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv(r"https://raw.githubusercontent.com/simmran2003/Recommendation-System/main/data.csv")
music_name=list(data["name"])


numerical_data = data[['valence', 'year', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'popularity', 
       'speechiness', 'tempo']]
features = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']

# Replace missing values with 0
data[features] = data[features].fillna(0)
scaler = StandardScaler()

# Fit and transform the features
data[features] = scaler.fit_transform(data[features])
n_clusters = 10

# Initialize K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit K-means to the data
data['cluster'] = kmeans.fit_predict(data[features])

def get_kmeans_recommendations(song_name, data, top_n=10):
    # Find the cluster of the given song
    song_cluster = data.loc[data['name'] == song_name, 'cluster'].values
    
    if len(song_cluster) == 0:
        print(f"Song '{song_name}' not found in the dataset.")
        return []
    
    song_cluster = song_cluster[0]
    
    # Get the songs from the same cluster
    cluster_songs = data[data['cluster'] == song_cluster]
    
    # Exclude the input song itself
    cluster_songs = cluster_songs[cluster_songs['name'] != song_name]
    
    # Get the top N recommendations
    recommendations = cluster_songs.head(top_n)
    
    return [recommendations['name'].tolist(),recommendations['id'].tolist()]


cluster_labels = kmeans.labels_


def print_recommendation(song_name,data):
    recommended_songs = get_kmeans_recommendations(song_name, data, top_n=10)
    required=data[data["id"].isin(recommended_songs[1])]
    plt.scatter(data["popularity"],data["liveness"],c=cluster_labels)
    given=data[data["name"]==song_name]
    plt.scatter(given["popularity"],given["liveness"],c="red",marker="*",s=300)
    plt.scatter(required["popularity"],required["liveness"],c="white",marker="*",s=200)
    st.pyplot(plt)



# Define a function to display recommendations
def show_recommendations():
    st.title('Spotify Music Recommendations')
    # Add Spotify header and image
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=100)
    # Display recommendations based on artist
    st.header("Get Songs Recommended based on your Interest")
    music = st.selectbox("Search Music:",music_name)
    st.markdown("### Songs similar to your Interest")
    predicted_songs=get_kmeans_recommendations(music,data,top_n=10)[0]
    st.write(predicted_songs)

def show_clustering():
    st.title("Clustering Graph")
    st.write("Visaulaize the Actual song and predicted song Cordinates")
    music = st.selectbox("Search Music:",music_name)
    print_recommendation(music,data)
def main():
    st.sidebar.title("Navigation")
    options = ["Recommendations", "Clustering Graph"]
    choice = st.sidebar.radio("Go to", options)
    if choice == "Recommendations":
        show_recommendations()
    elif choice == "Clustering Graph":
        show_clustering()
    

# Run the main function
if __name__ == "__main__":
    main()