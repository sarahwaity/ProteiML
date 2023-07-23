import os
import pathlib as pl
import pandas as pd
import logging
from typing import Union, List, Dict
from data_processing import format_mutation_dataset as fmr
from data_processing import construct_new_library as cnl
from data_processing import scrape_aaindex_matrix as sam
from data_processing.knr_execute import *

STANDARD_DIRECTORIES = ["backend_data", "mpnr", "knr", "rfr"]
STANDARD_DATA_DIR = pl.Path(os.getcwd()) / "backend_data"
STANDARD_INPUT_DATA = "Kinetics_Input_Data.csv"
SCRAPE_AAINDEX = False
AAINDEX_DB = "full_property_matrix.csv"


def create_project_directories(
    directories: Union[List[str], str] = STANDARD_DIRECTORIES
) -> None:
    """_summary_

    Args:
        directories (Union[List[str], str], optional): _description_. Defaults to STANDARD_DIRECTORIES.

    Raises:
        ValueError: _description_
    """
    if type(directories) is list:
        for dir_name in directories:
            logging.info(f"Checking/Creating {pl.Path(os.getcwd()) / dir_name}")
            os.makedirs(f"{pl.Path(os.getcwd()) / dir_name}", exist_ok=True)
    elif type(directories) is str:
        logging.info(f"Checking/Creating {pl.Path(os.getcwd()) / directories}")
        os.makedirs(f"{pl.Path(os.getcwd()) / dir_name}", exist_ok=True)
    else:
        raise ValueError("Input must be of type list or str.")


def load_input_csv_into_metadata_list(input_filepath: pl.Path) -> List[Dict[str, str]]:
    raw_input = pd.read_csv(input_filepath)

    return raw_input.to_dict(orient="records")


def step_1_format_mutation_dataset(
    metadata_dictionary_list: List[Dict[str, str]]
) -> pd.DataFrame:
    seq_df_list = list()
    cols_list = list()
    for metadata in metadata_dictionary_list:
        seq_entry, cols_entry = fmr.read_seq_data(metadata=metadata)
        seq_df_list.append(seq_entry)
        cols_list.append(cols_entry)

    combined_seq_data = fmr.combine_seq_datasets(seq_df_list)

    it = iter(cols_list)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError("not all mutation datasets have same length!")
    else:
        column_names = cols_list[0]
        base_variant_sequence_length = len(
            metadata_dictionary_list[0]["Sequence of Base Variant"]
        )
        biophysical_property_final = metadata_dictionary_list[0][
            "Desired Name of Dependent Variable"
        ]

    deduped_sequence_df = fmr.check_for_duplicates(
        combined_df=combined_seq_data,
        cols=column_names,
        base_variant_sequence_length=base_variant_sequence_length,
        biophysical_property_final=biophysical_property_final,
    )
    # Data Saving
    return deduped_sequence_df


def step_2_clean_variants(metadata_list):
    reference_variant_for_base_mutation = str(
        metadata_list[0]["Reference Index in Mutation Library"]
    )
    variant_name = metadata_list[0]["Variant For Optimization Name"]

    variant_df, testing_df, base_info = cnl.create_variant_dataframe(
        variant_library_filepath=STANDARD_DATA_DIR / "combined_dataset.csv",
        reference_variant_for_base_mutation=reference_variant_for_base_mutation,
        variant_name=variant_name,
    )

    cleaned_variants = cnl.clean_variant_dataframe(
        variant_df,
        testing_df,
        base_info=base_info,
    )

    return cleaned_variants


def step_3_scrape_aaindex_for_properties():
    if SCRAPE_AAINDEX == True:
        clean_names = sam.scrape_aaindex_indicies()
        property_dataset = sam.find_accession_numbers(clean_names)
        property_dataset.to_csv(STANDARD_DATA_DIR / "AAINDEX_Property_Dataset.csv")
        return property_dataset

    else:
        return pd.read_csv(STANDARD_DATA_DIR / AAINDEX_DB, index_col=0)


def prepare_and_train_knr_model(
    encoding_data, combined_dataset, len_sequence_of_base_variant, dependent_var
):
    x_train, x_test, y_train, y_test = prepare_encoding_data(
        combined_dataset, len_sequence_of_base_variant, dependent_var
    )
    knr_df = train_knr_model(encoding_data, x_train, x_test, y_train, y_test)

    return knr_df


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    ## Initialize Input/Output Directories
    create_project_directories()

    ## Load Input File
    metadata_list = load_input_csv_into_metadata_list(
        input_filepath=STANDARD_DATA_DIR / STANDARD_INPUT_DATA
    )

    ## Format and Combine Input Datasets
    combined_df = step_1_format_mutation_dataset(metadata_list)
    combined_df.to_csv(STANDARD_DATA_DIR / "combined_dataset.csv")

    ## Clean and save variant data
    cleaned_variants = step_2_clean_variants(metadata_list)
    cleaned_variants.to_csv(STANDARD_DATA_DIR / "novel_variant_library.csv")

    ## Scrape properties data.
    property_dataset = step_3_scrape_aaindex_for_properties()

    knr_df = prepare_and_train_knr_model(
        encoding_data=property_dataset,
        combined_dataset=combined_df,
        len_sequence_of_base_variant=len(metadata_list[0]["Sequence of Base Variant"]),
        dependent_var=metadata_list[0]["Desired Name of Dependent Variable"],
    )
    knr_df.to_csv(STANDARD_DATA_DIR / "KNR/KNR_Encoding_Datasets.csv")
