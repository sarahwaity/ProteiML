from data_processing.format_mutation_dataset import (
    get_col_count,
    read_seq_data,
    combine_seq_datasets,
    check_for_duplicates,
)
import pytest
import pandas as pd
import numpy as np
from main import STANDARD_DATA_DIR


def test_read_seq_data(metadata, base_variant_sequence):
    df_seq = read_seq_data(metadata)
    assert len(df_seq.columns) == len(base_variant_sequence) + 1
    assert len(np.unique(df_seq.index.values)) == len(df_seq)


def test_combine_seq_datasets(df_list):
    combined_df = combine_seq_datasets(df_list)
    assert len(combined_df) == sum([len(df) for df in df_list])


def test_check_for_duplicates(
    combined_df, cols, base_variant_sequence, biophysical_property_final
):
    df_seq = check_for_duplicates(
        combined_df, cols, len(base_variant_sequence), biophysical_property_final
    )
    assert len(df_seq[df_seq.duplicated(cols, keep=False)]) == 0
    assert len(df_seq.columns) == len(base_variant_sequence) + 2
