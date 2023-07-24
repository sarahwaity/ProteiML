import os
import pathlib as pl
import pandas as pd
import numpy as np
import logging
from typing import Union, List, Dict, Tuple
from datetime import date

from data_processing.constants import *
from data_processing import format_mutation_dataset as fmr
from data_processing import construct_new_library as cnl
from data_processing import scrape_aaindex_matrix as sam
from data_processing import model_training as mt
from data_processing import model_prediction as mp
from data_processing import ensemble_and_cross_validate as ecv

def create_project_directories(
    directories: Union[List[str], str] = STANDARD_DIRECTORIES
) -> None:
    """creates subdirectories consistant with the STANDARD_DIRECTORIES constant

    Args:
        directories (Union[List[str], str], optional): list of directories to be made. Defaults to STANDARD_DIRECTORIES.

    Raises:
        ValueError: incorrect input for directory generation
    """
    if type(directories) is list:
        for dir_name in directories:
            logging.info(f"Checking/Creating {BASE_DIR / dir_name}")
            os.makedirs(BASE_DIR / dir_name, exist_ok=True)
    elif type(directories) is str:
        logging.info(f"Checking/Creating {BASE_DIR / directories}")
        os.makedirs(BASE_DIR /  dir_name, exist_ok=True)
    else:
        raise ValueError("Input must be of type list or str.")


def load_input_csv_into_metadata_list(input_filepath: pl.Path) -> List[Dict[str, str]]:
    """imports the user provided .csv and loads it to a dict

    Args:
        input_filepath (pl.Path): path to run information

    Returns:
        List[Dict[str, str]]: dict of information found in input file
    """
    raw_input = pd.read_csv(input_filepath)

    return raw_input.to_dict(orient="records")


def ingest_and_format_mutation_dataset(
    metadata_dictionary_list: List[Dict[str, str]]
) -> pd.DataFrame:
    """Formats the supplied mutation dataframe into a format that can be understood by machine learning models. 

    Args:
        metadata_dictionary_list (List[Dict[str, str]]): output of load_input_csv_into_metadata_list

    Raises:
        ValueError: mutation datasets could not be combined because they contain difference length sequences!

    Returns:
        pd.DataFrame: formatted muation df, one column for each residue position, one column for the target variable, one column that contains the mutant name/ID
    """
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


def clean_variants(metadata_list: List[Dict[str, str]]) -> pd.DataFrame:
    """Generates a novel point mutation library for based off the known mutant df and chosen base variant. 

    Args:
        metadata_list (List[Dict[str, str]]): output of load_input_csv_into_metadata_list

    Returns:
        pd.DataFrame: dataframe that contains a point mutation library, with no redundant sequences both internally and with the known variant library. 
    """
    reference_variant_for_base_mutation = str(
        metadata_list[0]["Reference Index in Mutation Library"]
    )
    variant_name = metadata_list[0]["Variant For Optimization Name"]

    variant_df, testing_df, base_info = cnl.create_variant_dataframe(
        variant_library_filepath=BACKEND_DATA_DIR / "combined_dataset.csv",
        reference_variant_for_base_mutation=reference_variant_for_base_mutation,
        variant_name=variant_name,
    )

    cleaned_variants = cnl.clean_variant_dataframe(
        variant_df,
        testing_df,
        base_info=base_info,
    )

    return cleaned_variants


def scrape_aaindex_for_properties() -> pd.DataFrame:
    """if the property matrix does not exist, this will scrape AAINDEX for all of the property matricies.

    Returns:
        pd.DataFrame: dataframe with property matricies from AAINDEX
    """
    if SCRAPE_AAINDEX == True:
        clean_names = sam.scrape_aaindex_indicies()
        property_dataset = sam.find_accession_numbers(clean_names)
        property_dataset.to_csv(BACKEND_DATA_DIR / AAINDEX_DB)
        return property_dataset

    else:
        return pd.read_csv(BACKEND_DATA_DIR / AAINDEX_DB, index_col=0)


def prepare_and_train_models(
    encoding_data: pd.DataFrame,
    combined_dataset: pd.DataFrame,
    len_sequence_of_base_variant: int,
    dependent_var: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trains each model (RFR,MPNR,KNR) on the sequence-to-function library on each complete property matrix in the AAINDEX property matricies list

    Args:
        encoding_data (pd.DataFrame): amino acid property matrcies from AAINDEX
        combined_dataset (pd.DataFrame): output of format_mutation_dataset function, full sequence-to-function library
        len_sequence_of_base_variant (int): length of base variant sequence, used to exract sequence information
        dependent_var (str): name of the variable for optimization 

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: one dataframe per regressor type, that contain information regarding encoding dataset and R squared achieved. 
    """
    x_train, x_test, y_train, y_test, encoded_df = mt.prepare_encoding_data(
        combined_dataset, len_sequence_of_base_variant, dependent_var
    )
    logging.info("Training KNR Model...")
    knr_df = mt.train_knr_model(encoding_data, x_train, x_test, y_train, y_test)

    logging.info("Training MPNR Model...")
    mpnr_df = mt.train_mpnr_model(
        encoding_data, encoded_df, x_train, x_test, y_train, y_test
    )

    logging.info("Training RFR Model...")
    rfr_df = mt.train_rfr_model(encoding_data, x_train, x_test, y_train, y_test)

    return knr_df, mpnr_df, rfr_df


def retrain_and_get_novel_predictions(
    encoding_data: pd.DataFrame,
    combined_dataset: pd.DataFrame,
    len_sequence_of_base_variant: int,
    novel_library_path: pl.Path,
    top_encoding_dataset_paths: Dict[str, str],
    dependent_variable: str,
) -> Tuple[np.array, np.array, np.array, np.array, pd.DataFrame]:
    """Retrains each regressor using the top five performing datsets and generates predictions on the novel mutant library and withheld test datasets. 

    Args:
        encoding_data (pd.DataFrame): amino acid property matrcies from AAINDEX
        combined_dataset (pd.DataFrame): output of format_mutation_dataset function, full sequence-to-function library 
        len_sequence_of_base_variant (int): length of base variant sequence, used to exract sequence information
        novel_library_path (pl.Path): path to the novel point mutation library 
        top_encoding_dataset_paths (Dict[str,str]): path to output of prepare_and_train_models function, one dataframe per regressor type, that contain information regarding encoding dataset and R squared achieved.
        dependent_variable (str): name of the variable for optimization
    """
    x_train, x_test, y_train, y_test, encoded_df = mt.prepare_encoding_data(
        combined_dataset, len_sequence_of_base_variant, dependent_var=dependent_variable
    )

    data_pred, _ = mp.generate_prediction_data(novel_library_path)
    knr_names = mp.create_encoded_names(top_encoding_dataset_paths["knr"])
    mpnr_names = mp.create_encoded_names(top_encoding_dataset_paths["mpnr"])
    rfr_names = mp.create_encoded_names(top_encoding_dataset_paths["rfr"])

    mp.retrain_predict_knr(
        knr_names,
        encoding_data,
        data_pred,
        x_train,
        x_test,
        y_train,
        y_test,
        dependent_variable,
    )
    mp.retrain_predict_mpnr(
        mpnr_names,
        encoding_data,
        encoded_df,
        data_pred,
        x_train,
        x_test,
        y_train,
        y_test,
        dependent_variable,
    )

    mp.retrain_predict_rfr(
        rfr_names,
        encoding_data,
        data_pred,
        x_train,
        x_test,
        y_train,
        y_test,
        dependent_variable,
    )


def cross_validate_and_ensemble(
    top_encoding_dataset_paths: Dict[str, str],
    len_base_variant_sequence: int,
    dependent_variable: str,
    comparison_variant: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads in the results from the model predictions and generates final ensemble predictions, as well as cross-validation R2 values for every model

    Args:
        top_encoding_dataset_paths (Dict[str,str]): path to output of prepare_and_train_models function, one dataframe per regressor type, that contain information regarding encoding dataset and R squared achieved.
        len_base_variant_sequence (int): length of base variant sequence, used to exract sequence information
        dependent_variable (str): name of the variable for optimization
        comparison_variant (str): variant used to form the novel library, now used as a metric of prediction outliers

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: dataframe containing the predictions from each contributor model, as well as the ensemble predictions, variant ID, and P-value compared to predictions made on the base construct. Cross validation gives R2 values for each contributor model and the ensemble results. 
    """
    name_dict = ecv.isolate_encoding_datasets(
        rfr_encoding_dataset_path=top_encoding_dataset_paths["rfr"],
        mpnr_encoding_dataset_path=top_encoding_dataset_paths["mpnr"],
        knr_encoding_dataset_path=top_encoding_dataset_paths["knr"],
    )

    ensemble_df = ecv.create_ensemble_dataset(
        dict_of_names=name_dict,
        len_base_variant_sequence=len_base_variant_sequence,
        dependent_variable=dependent_variable,
        comparison_variant=comparison_variant,
    )

    r_squared_df = ecv.create_r2_validations_dataframe(dict_of_names=name_dict)

    return ensemble_df, r_squared_df


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    ## Initialize Input/Output Directories
    create_project_directories()

    ## Load Input File
    metadata_list = load_input_csv_into_metadata_list(
        input_filepath=BACKEND_DATA_DIR / STANDARD_INPUT_DATA
    )

    ## Format and Combine Input Datasets
    combined_df = ingest_and_format_mutation_dataset(metadata_list)
    combined_df.to_csv(INTERMEDIATE_DATA_DIR / "combined_dataset.csv")

    ## Clean and save variant data
    cleaned_variants = clean_variants(metadata_list)
    cleaned_variants.to_csv(INTERMEDIATE_DATA_DIR / "novel_variant_library.csv")

    ## Scrape properties data.
    property_dataset = scrape_aaindex_for_properties()

    knr_df, mpnr_df, rfr_df = prepare_and_train_models(
        encoding_data=property_dataset,
        combined_dataset=combined_df,
        len_sequence_of_base_variant=len(metadata_list[0]["Sequence of Base Variant"]),
        dependent_var=metadata_list[0]["Desired Name of Dependent Variable"],
    )

    top_encoding_dataset_paths = {
        "knr": BASE_DIR / "knr/KNR_Encoding_Datasets.csv",
        "mpnr": BASE_DIR / "mpnr/MPNR_Encoding_Datasets.csv",
        "rfr": BASE_DIR / "rfr/RFR_Encoding_Datasets.csv",
    }

    knr_df.to_csv(top_encoding_dataset_paths["knr"])
    mpnr_df.to_csv(top_encoding_dataset_paths["mpnr"])
    rfr_df.to_csv(top_encoding_dataset_paths["rfr"])

    retrain_and_get_novel_predictions(
        encoding_data=property_dataset,
        combined_dataset=combined_df,
        len_sequence_of_base_variant=len(metadata_list[0]["Sequence of Base Variant"]),
        dependent_variable=metadata_list[0]["Desired Name of Dependent Variable"],
        novel_library_path=INTERMEDIATE_DATA_DIR / "novel_variant_library.csv",
        top_encoding_dataset_paths=top_encoding_dataset_paths,
    )

    ensemble_df, r_squared_dict = cross_validate_and_ensemble(
        top_encoding_dataset_paths=top_encoding_dataset_paths,
        len_base_variant_sequence=len(metadata_list[0]["Sequence of Base Variant"]),
        dependent_variable=metadata_list[0]["Desired Name of Dependent Variable"],
        comparison_variant=metadata_list[0]["Variant For Optimization Name"],
    )

    ensemble_df.to_csv(OUTPUT_DIR / "Ensemble_insilico_predictions_.csv")
    r_squared_dict.to_csv(OUTPUT_DIR / "Cross_Validation_Scores.csv")
