import os
import pandas as pd
import pickle

from e_commerce.config import Configuration

class Recommender():

    def __init__(self, train_path: str, matrix_path: str, model_path: str, frac: float = 0.1):
        self.data = pd.read_parquet(os.path.join(Configuration.DATA, train_path))
        self.model = self.__load_model(os.path.join(Configuration.MODEL, model_path))

        sub_user_profile_matrix, neighbors = self.__get_neighbors(matrix_path, frac)
        
        self.sub_user_profile_matrix = sub_user_profile_matrix
        self.neighbors = neighbors

    @staticmethod
    def __load_model(model_path: str):
        with open(os.path.join("model", model_path), "rb") as f:
            model = pickle.load(f)

        return model

    @staticmethod
    def load_user_purchase_matrix(matrix_path: str) -> pd.DataFrame:
        return pd.read_pickle(os.path.join("model", matrix_path))

    def __get_neighbors(self, matrix_path: str, frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
        user_profile_matrix = self.load_user_purchase_matrix(os.path.join(Configuration.MODEL, matrix_path))    

        sub_user_proflie_matrix = user_profile_matrix.sample(frac=frac)
        neighbors = self.model.kneighbors(sub_user_proflie_matrix, return_distance=False)

        neighbors_of_users = []
        for neigh_list in neighbors:
            neighbors_of_users.append(pd.DataFrame([user_profile_matrix.iloc[neigh_list].index]))
        
        neighbor_ids = pd.concat(neighbors_of_users, ignore_index=True)

        return sub_user_proflie_matrix, neighbor_ids

    def get_user_items(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        
        users_historic_purchase_matrix = []
        users_validation_purchase_matrix = []
        
        for u_ids, _ in self.sub_user_profile_matrix.iterrows():
            users_historic_purchase_matrix.append(self.data[self.data['user_id'] == u_ids])
            users_validation_purchase_matrix.append(self.val[self.val['user_id'] == u_ids])

        return users_historic_purchase_matrix, users_validation_purchase_matrix

    def get_users_recommendations(self, users_train_items_matrix: list[pd.DataFrame]) -> tuple[ str, list[list[str]] ]:
        
        users_recommendation_matrix = []
        
        self.neighbors.reset_index()
        for i, user_neighbors in self.neighbors.iterrows():
            
            user_recommendation_list = self.data[
                self.data['user_id'].isin(user_neighbors.values)]['product_id'].unique().tolist()
            user_recommendation_list = [item_id for item_id in user_recommendation_list if item_id not in 
                users_train_items_matrix[i]['product_id'].unique().tolist()
            ]

            users_recommendation_matrix.append((self.sub_user_profile_matrix.index[i], user_recommendation_list))

        return users_recommendation_matrix

    @staticmethod
    def get_score(recommendation_path: str, test_path: str):
        
        with open(os.path.join(Configuration.DATA, recommendation_path), "rb") as f:
            recommendations = pickle.load(f)

        validations = pd.read_parquet(os.path.join(Configuration.DATA, test_path))
        
        cont_success = 0
        for user_id, recommendation_list in recommendations:
            future_user_purchase = validations[validations['user_id'] == user_id]['product_id'].unique().tolist()

            for user_purchase in future_user_purchase:
                if user_purchase in recommendation_list:
                    cont_success += 1
                    break
        
        return cont_success / len(recommendations)

        

