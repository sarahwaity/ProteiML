import os
import pathlib as pl
from datetime import date


BASE_DIR = pl.Path(os.getcwd()) / "data"

STANDARD_DIRECTORIES = ["backend_data", "intermediate_data", "output_data", "mpnr", "knr", "rfr"]
BACKEND_DATA_DIR = BASE_DIR / "backend_data"
INTERMEDIATE_DATA_DIR = BASE_DIR / "intermediate_data"
STANDARD_INPUT_DATA = "Input_Data.csv"
SCRAPE_AAINDEX = False
AAINDEX_DB = "streamlined_property_matrix.csv"
OUTPUT_DIR = BASE_DIR / "output_data"