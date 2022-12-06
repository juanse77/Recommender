import os
import sys

import pandas as pd
import numpy as np

import typer
import pickle

from sklearn.neighbors import NearestNeighbors

import e_commerce.load_data_helper as ldh
from e_commerce.recommender import Recommender
from e_commerce.config import Configuration

def build_dirs():
    if not os.path.exists(Configuration.DATA):
        os.mkdir(Configuration.DATA)

    if not os.path.exists(Configuration.MODEL):
        os.mkdir(Configuration.MODEL)

app = typer.Typer()

@app.command()
def generate_user_purchase_matrix(matrix_path: str = typer.Argument(..., help="Path of the output file of the user purchase matrix")):
    
    build_dirs()

    try:
        test = pd.read_parquet(os.path.join(Configuration.DATA, "test.parquet"))
        train = pd.read_parquet(os.path.join(Configuration.DATA, "train.parquet"))
        val = pd.read_parquet(os.path.join(Configuration.DATA, "val.parquet"))
    except FileNotFoundError as fe:
        print(fe.strerror)
        sys.exit(1)

    test.replace('NA', np.nan, inplace=True)
    train.replace('NA', np.nan, inplace=True)
    val.replace('NA', np.nan, inplace=True)

    data = pd.concat([test, train, val], ignore_index=True)

    cat_fields = ['cat_0', 'cat_1', 'cat_2', 'cat_3']

    categories = pd.concat([
        #data['brand'],
        data['cat_0'],
        data['cat_1'],
        data['cat_2'],
        data['cat_3']
    ], ignore_index=True).dropna().unique()
    
    del(data)
    del(test)
    del(val)

    n_samples = 25

    train = train.groupby('user_id').tail(n_samples)
    train.to_parquet(os.path.join(Configuration.DATA, "filtered_train_data.parquet"))

    print(train.shape)

    user_ids = train['user_id'].unique()

    if not train.empty:

        matrix = ldh.generate_empty_matrix(reference_client_ids=user_ids, categories=categories)
        ldh.fill_matrix(matrix, train, cat_fields=cat_fields, inplace=True)

    matrix.to_pickle(os.path.join(Configuration.MODEL, matrix_path))

@app.command()
def fit_model(
    matrix_path: str = typer.Argument(..., help="Path of the user purchase matrix"),
    model_path: str = typer.Option("model.pkl", help="Output path for the fitted model"),
    n_neighbors: int = typer.Option(3, help="Number of neighbors to use by default for kneighbors queries")
):
    build_dirs()
    
    try:
        matrix = Recommender.load_user_purchase_matrix(matrix_path)
    except FileNotFoundError as fe:
        print(fe.strerror)
        sys.exit(1)
    
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )

    model.fit(matrix)

    with open(os.path.join(Configuration.MODEL, model_path), "wb") as f:
        pickle.dump(model, f)


@app.command()
def get_recommendations(
    matrix_path: str = typer.Argument(..., help="Path of the user purchase matrix"),
    model_path: str = typer.Option("model.pkl", help="Output path for the fitted model"),
    train_path: str = typer.Option(..., help="Path of the training set"),
    frac: float = typer.Option(0.1, help="Fraction of users to recommend")):

    build_dirs()

    try:
        recomender = Recommender(train_path, matrix_path, model_path, frac)
    except FileNotFoundError as fe:
        print(fe.strerror)
        sys.exit(1)

    users_train_items_matrix,  = recomender.get_user_items()
    users_recomentation_matrix = recomender.get_users_recommendations(users_train_items_matrix)

    with open(os.path.join(Configuration.DATA, "recommendations.pkl"), "wb") as f:
        pickle.dump(users_recomentation_matrix, f)

@app.command()
def get_score(
    recommendation_path: str = typer.Argument(..., help="Path of the predictions of recommendations"),
    test_path: str = typer.Option(..., help="Path of the test set")
):
    build_dirs()

    try:
        score = Recommender.get_score(recommendation_path, test_path)
        print(score)
    except FileNotFoundError as fe:
        print(fe.strerror)
        sys.exit(1)

if __name__ == "__main__":
    app()
