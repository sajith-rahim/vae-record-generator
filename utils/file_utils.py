import json
import pandas as pd


def read_csv(csv_filename, has_header=True, discrete_cols=None):
    """ """
    data = pd.read_csv(csv_filename, header='infer' if has_header else None)

    if discrete_cols:
        discrete_columns = discrete_cols.split(',')
        if not has_header:
            discrete_columns = [int(i) for i in discrete_columns]

    else:
        discrete_columns = []

    return data, discrete_columns
