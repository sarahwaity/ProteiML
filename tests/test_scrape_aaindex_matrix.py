from data_processing.scrape_aaindex_matrix import *
from data_processing.constants import *
import numpy as np

def test_aaindex_scrape_or_load():

    if SCRAPE_AAINDEX == True:
        clean_names = scrape_aaindex_indicies()
        property_dataset = find_accession_numbers(clean_names)
        property_dataset.to_csv(BACKEND_DATA_DIR / "AAINDEX_Property_Dataset.csv")


    else:
        property_dataset = pd.read_csv(BACKEND_DATA_DIR / AAINDEX_DB, index_col=0)
    
    assert np.shape(property_dataset) == (20, 6)