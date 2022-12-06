import pandas as pd
import numpy as np


def fill_matrix(matrix: pd.DataFrame, reference_sales: pd.DataFrame, cat_fields: list[str], inplace: bool = False) -> pd.DataFrame:
    
    if inplace:
        m_filled = matrix
    else:
        m_filled = matrix.copy()

    for _, row in reference_sales.iterrows():
        for cat in cat_fields:
            if row[cat] is not np.nan:
                m_filled.loc[row['user_id'], row[cat]] += 1

    if inplace:
        return

    return m_filled

def generate_empty_matrix(reference_client_ids: np.ndarray, categories: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(data=np.zeros((len(reference_client_ids), len(categories)), dtype=np.int16), index=reference_client_ids,\
        columns=categories, dtype=np.int16)

def get_even_samples(data: pd.DataFrame, user_ids: np.ndarray, n_samples: int = 25) -> pd.DataFrame:
    
    purchase_list = []
    
    for user_id in user_ids:
        user_purchase_filter = data[data['user_id'] == user_id].iloc[-n_samples:]
        purchase_list.append(user_purchase_filter)

    return pd.concat(purchase_list)