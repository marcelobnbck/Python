import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import zipfile
warnings.filterwarnings("ignore")

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Importing the Dataset
print("Importing the Movies Dataset...")
data = pd.read_csv('C:/Pocs/Python/PythonRecomendationSystem/movies_metadata.csv', low_memory=False)

# Building the Content Based Recommender
# Calculating the average vote rate
print("Calculating the average vote rate...")
vote_rate = data['vote_average'].mean()

# Calculating the minimum number of votes to be in the chart
print("Calculating the minimum number of votes...")
min_votes = data['vote_count'].quantile(0.90)

# Computing the score(rating) of each movie
def score(x, min_votes=min_votes, vote_rate=vote_rate):
    print("Computing the score of each movie...")
    vote_cnt = x['vote_count']
    vote_avg = x['vote_average']
    # Calculation based on the IMDB formula
    print("Calculation based on the IMDB formula...")
    return (vote_cnt / (vote_cnt + min_votes) * vote_avg) + (min_votes / (min_votes + vote_cnt) * vote_rate)

# Defining a new feature 'score' and calculate its value
print("Calculating Score value...")
data['score'] = data.apply(score, axis=1) #aplicando o score direto no Dataframe principal
new_moviesdf = data.copy().loc[data['vote_count'] >= min_votes]

# Sorting the movies based on score calculated above
print("Sorting the movies based on score calculated...")
new_moviesdf = new_moviesdf.sort_values('score', ascending=False)

# Load keywords and credits
#####print("Extracting Zip file...")
#####with zipfile.ZipFile('credits.zip', 'r') as zip:
#####    # Extraia todos os arquivos do arquivo ZIP para um diretório específico
#####    zip.extractall('/extracted')

arquivo_zip = zipfile.ZipFile('credits.zip')
arquivo_zip.extract('credits.csv', './extracted')
arquivo_zip.close()

print("Importing the Cretids extracted Dataset...")
credits = pd.read_csv('extracted/credits.csv')
print("Importing the Keywords Dataset...")
keywords = pd.read_csv('keywords.csv')

# Remove rows with bad IDs.
data = data.drop([19730, 29503, 35587])

# Convert IDs to int (Merging Purpose)
print("Converting IDs to int...")
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
data['id'] = data['id'].astype('int')

# Merge keywords and credits into main 'data' dataframe
print("Merging keywords and credits into main dataframe...")
data = data.merge(credits, on='id')
data = data.merge(keywords, on='id')

# Parsing the string features into their corresponding python objects
print("Parsing string features into python objects...")
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
print("Defining new director, cast, genres and keywords features...")
data['director'] = data['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    data[feature] = data[feature].apply(get_list)

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    print("Converting strings to lower case and strip names of spaces...")
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
print("Applying clean_data function...")
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    data[feature] = data[feature].apply(clean_data)

def merge(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

data['merge'] = data.apply(merge, axis=1)

# Create the count matrix
print("Creating the count matrix...")
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['merge'])

# Compute the Cosine Similarity matrix based on the count_matrix
print("Computing the Cosine Similarity matrix...")
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
print("Reseting index of your main DataFrame and construct reverse mapping...")
data = data.reset_index()
indices = pd.Series(data.index, index=data['title'])

# Function that takes in movie title as input and outputs most similar movies
def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    print("Geting the index of the movie that matches the title...")
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    print("Geting the pairwsie similarity scores of all movies with that movie...")
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    print("Sorting the movies based on the similarity scores...")
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    print("Geting the scores of the 10 most similar movies...")
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    print("Geting the movie indices...")
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies with their scores
    recommended_movies = data.iloc[movie_indices][['title', 'score']]
    return recommended_movies
    
# Get user input and call the function
movie_title = input("Enter movie title: ").strip()
print(recommend_movies(movie_title))