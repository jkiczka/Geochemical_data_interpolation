import os
import pandas as pd
from data_pipeline import preprocess_yang_dataset, adjust_yang_grid_shallow


PATH_WOA = 'data/processed/woa_grid_shallow.csv'
PATH_YANG = 'data/raw/surface_n2o_compilation.csv'
PATH_JOINED = 'data/processed/yang_woa_grid_shallow.csv'
SAVE_ALL = False # woa_grid.csv is huge! (6.07GB)
VERBOSE = True

if __name__ == "__main__":
    if not os.path.exists(PATH_WOA):
        print(f"File {PATH_WOA} not found!")
        print(f"Run src/main_grid.py first.")
    else:
        df_yang = preprocess_yang_dataset()
        df_woa_shallow = pd.read_csv(PATH_WOA)
        df_joined = adjust_yang_grid_shallow(df_yang, df_woa_shallow)
        df_joined.to_csv(PATH_JOINED, index=False)