import os
import pathlib as pl

STANDARD_DIRECTORIES = ["backend_data", "output_data", "mpnr", "knr", "rfr"]
BASE_DIR = pl.Path(os.getcwd())
STANDARD_DATA_DIR = BASE_DIR / "backend_data"
STANDARD_INPUT_DATA = "Kinetics_Input_Data.csv"
SCRAPE_AAINDEX = False
AAINDEX_DB = "streamlined_property_matrix.csv"
OUTPUT_DIR = BASE_DIR / "output_data"