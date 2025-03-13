# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
#%matplotlib inline

# Importing the Dataset
data = pd.read_csv('movies_metadata.csv', low_memory = False)

# Exploratory Data Analysis
# - Dataset Preview
data.head(3)
# - Dimensions of the Dataset
data.shape 
# - Summary of the Dataset
data.info()
# Checking for missing values
data.isnull().sum()
# Numeric Columns
data.describe()
# Character Columns
data.describe(include=['object'])

# Building the Content Based Recommender
# Calculating the average vote rate
vote_rate = data['vote_average'].mean()
print(vote_rate)
# Calculating the minimum number of votes to be in the chart
min_votes = data['vote_count'].quantile(0.90)
print(min_votes)
# Filtering out all qualified movies into a new DataFrame
new_moviesdf = data.copy().loc[data['vote_count'] >= min_votes]
new_moviesdf.shape
# Computing the score(rating) of each movie
def score(x, min_votes = min_votes, vote_rate = vote_rate):
    vote_cnt = x['vote_count']
    vote_avg = x['vote_average']
    # Calculation based on the IMDB formula
    return (vote_cnt/(vote_cnt+min_votes) * vote_avg) + (min_votes/(min_votes+vote_cnt) * vote_rate)
# Defining a new feature 'score' and calculate its value
new_moviesdf['score'] = new_moviesdf.apply(score, axis=1)
# Sorting the movies based on score calculated above
new_moviesdf = new_moviesdf.sort_values('score', ascending=False)
# Print the top 5 movies
new_moviesdf[['title', 'vote_count', 'vote_average', 'score']].head(5)

# Content - Based Recommender
# Credits, Genres, and Keywords-Based Recommender
# Load keywords and credits
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
# Remove rows with bad IDs.
data = data.drop([19730, 29503, 35587]) # Causes value error

# Convert IDs to int (Merging Purpose)
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
data['id'] = data['id'].astype('int')

# Merge keywords and credits into main 'data' dataframe
data = data.merge(credits, on='id')
data = data.merge(keywords, on='id')

# Newly merged Dataframe
data.head(2)

# Parsing the string features into their corresponding python objects
from ast import literal_eval # helps in traversing an abstract syntax tree

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    data[feature] = data[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    # Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
data['director'] = data['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    data[feature] = data[feature].apply(get_list)

# Print the new features of the first 3 films
data[['title', 'cast', 'director', 'keywords', 'genres']].head(3)

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    data[feature] = data[feature].apply(clean_data)

def merge(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

data['merge'] = data.apply(merge, axis=1)

data[['merge']].head(5)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['merge'])

count_matrix.shape

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
data = data.reset_index()
indices = pd.Series(data.index, index = data['title'])

# Function that takes in movie title as input and outputs most similar movies
def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]
