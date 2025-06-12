from data_pipeline import process_woa_grid, get_woa_grid_shallow


PATH = 'data/processed/woa_grid.csv'
PATH_SHALLOW = 'data/processed/woa_grid_shallow.csv'
SAVE_ALL = False # woa_grid.csv is huge! (6.07GB)
VERBOSE = True

if __name__ == "__main__":
    df_woa, ds_woa = process_woa_grid(verbose=VERBOSE)
    df_woa_shallow = get_woa_grid_shallow(ds_woa)
    if SAVE_ALL:
        df_woa.to_csv(PATH, index=False)
    df_woa_shallow.to_csv(PATH, index=False)