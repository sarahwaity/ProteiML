# MAIN PACKAGES
import pathlib as pl
import pandas as pd
import logging
from typing import Tuple


def create_variant_dataframe(
    variant_library_filepath: pl.Path,
    reference_variant_for_base_mutation: str,
    variant_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Constructs a point mutation library based off of a chosen variant. 

    Args:
        variant_library_filepath (pl.Path): string that contains path to known variant library 
        reference_variant_for_base_mutation (str): name of the base construct variant as it appears in variant library file
        variant_name (str): name descriptor of the base construct variant (reference_variant_for_base_mutation and variant_name may be identical)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: point mutation library, known sequences, sequence and ID of base construct
    """
    # data input and sorting
    data = pd.read_csv(variant_library_filepath, index_col=0)
    data = data.sort_values(by="Variant ID", ascending=False)

    # isolates all columns that are sequence
    cols = data.columns[:-2]
    # isolates sequence desired for base construct
    ref_var_index = data.loc[reference_variant_for_base_mutation][cols]
    base_sequence = list(ref_var_index)

    base_info = list(data.loc[reference_variant_for_base_mutation][cols].values) + [
        variant_name
    ]

    # isolate the columns with sequence data
    testing_df = data[cols]

    # define amino acids for point saturation
    amino_acid_keys = [
        "G",
        "A",
        "V",
        "L",
        "I",
        "M",
        "F",
        "W",
        "P",
        "S",
        "T",
        "C",
        "Y",
        "N",
        "D",
        "Q",
        "E",
        "K",
        "R",
        "H",
    ]

    # external datasaving
    variant_df = pd.DataFrame()
    name_list = []

    for column in cols:
        # initialize base sequence for each column
        base_sequence = list(ref_var_index)

        # define the starting amino acid
        reference_aa = base_sequence[int(column)]

        # find residues with mutation data and saturate each position
        if len(testing_df[column].unique()) > 1:
            for aa in amino_acid_keys:
                base_sequence[int(column)] = aa
                name_list.append(variant_name + " " + reference_aa + column + aa)
                # save each novel sequence to the outside dataframe
                variant_df_ = pd.DataFrame(base_sequence)
                variant_df = pd.concat([variant_df, variant_df_], axis=1)
    variant_df = variant_df.T
    variant_df.reset_index(inplace=True, drop=True)
    variant_df = variant_df.set_axis(cols, axis=1)
    variant_df["mutant_name"] = name_list  # adds identity of each mutation

    return variant_df, testing_df, base_info


def clean_variant_dataframe(
    variant_df: pd.DataFrame, testing_df: pd.DataFrame, base_info: pd.DataFrame
) -> pd.DataFrame:
    """takes the point mutation library (variant dataframe) and compares the sequence to the known variant library, removes any duplicates if they exist. 

    Args:
        variant_df (pd.DataFrame): point mutation library based on base construct
        testing_df (pd.DataFrame): all the sequences that are contained in the known variant library
        base_info (pd.DataFrame): contains sequence and identity of base construct

    Returns:
        pd.DataFrame: point mutation library of unknown sequences
    """
    ##Duplicate Values Check
    # takes all of the sequences from the known library
    comparison_massive_list_set = [
        list(testing_df.iloc[e]) for e in range(len(testing_df))
    ]
    # all of the sequences from the point-mutation library
    cols_int = variant_df.columns[:-1]

    variant_df_copy = variant_df.copy().T

    for row in range(len(variant_df)):
        # Input List Initialization
        input = comparison_massive_list_set

        # List to be searched
        list_search = list(variant_df[cols_int].iloc[row])
        index_ = variant_df.index[row]

        if list_search in input:
            _ = variant_df_copy.pop(index_)

    variant_df_cleaned = variant_df_copy.T
    logging.info(
        str((len(variant_df) - len(variant_df_cleaned)))
        + " Duplicate Variants Found and Removed"
    )

    base_var_df = pd.DataFrame([base_info], columns=variant_df_cleaned.columns)

    variant_df_cleaned = pd.concat([base_var_df, variant_df_cleaned])

    logging.info(f"Variant DF Cleaned. Shape: {variant_df_cleaned.shape}")
    return variant_df_cleaned
