# -*- coding: utf-8 -*-
"""
Created on Thurs May 12 05:19:48 2022


"""

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
#print(C)
# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
#print(m)
# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape
# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
#print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Array mapping from feature integer indices to feature name.
#tfidf.get_feature_names()[1000:1010]
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()



##############################################################################################
# For API
import json
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/get_recommendations/<string:title>')
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
   

    # Return the top 10 most similar movies # returning 1 movie for testing
    
    Data =  {"titles":metadata['title'].iloc[movie_indices].iloc[1],"Desc":metadata['overview'].iloc[movie_indices].iloc[1]}
             
    ''' {"titles":metadata['title'].iloc[movie_indices].iloc[2],"Desc":metadata['overview'].iloc[movie_indices].iloc[2]},
             {"titles":metadata['title'].iloc[movie_indices].iloc[3],"Desc":metadata['overview'].iloc[movie_indices].iloc[3]},
             {"titles":metadata['title'].iloc[movie_indices].iloc[4],"Desc":metadata['overview'].iloc[movie_indices].iloc[4]},
             {"titles":metadata['title'].iloc[movie_indices].iloc[5],"Desc":metadata['overview'].iloc[movie_indices].iloc[5]},
             {"titles":metadata['title'].iloc[movie_indices].iloc[6],"Desc":metadata['overview'].iloc[movie_indices].iloc[6]},
             {"titles":metadata['title'].iloc[movie_indices].iloc[7],"Desc":metadata['overview'].iloc[movie_indices].iloc[7]},
             {"titles":metadata['title'].iloc[movie_indices].iloc[8],"Desc":metadata['overview'].iloc[movie_indices].iloc[8]},
             {"titles":metadata['title'].iloc[movie_indices].iloc[9],"Desc":metadata['overview'].iloc[movie_indices].iloc[9]},
    {"titles":metadata['title'].iloc[movie_indices].iloc[10],"Desc":metadata['overview'].iloc[movie_indices].iloc[10]}}'''
                
    
  
    
    
    return json.dumps(Data)
    



if __name__ == "__main__":
    app.run(debug=False)