from data_processing.format_mutation_dataset import (
    read_seq_data,
    combine_seq_datasets,
    check_for_duplicates,
)

import pytest
import numpy as np
import pandas as pd
from data_processing.constants import *


@pytest.fixture
def metadata_fixture():
    raw_input = pd.read_csv(BACKEND_DATA_DIR / "Input_Data.csv")

    return raw_input.to_dict(orient="records")


@pytest.fixture
def base_variant_sequence_fixture(metadata_fixture):
    return metadata_fixture[0]["Sequence of Base Variant"]


def test_format_mutation_dataset(metadata_fixture, base_variant_sequence_fixture):
    seq_df_list = list()
    cols_list = list()
    for metadata in metadata_fixture:
        seq_entry, cols_entry = read_seq_data(metadata=metadata)
        seq_df_list.append(seq_entry)
        cols_list.append(cols_entry)

    combined_seq_data = combine_seq_datasets(seq_df_list)

    it = iter(cols_list)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError("not all mutation datasets have same length!")
    else:
        column_names = cols_list[0]
        base_variant_sequence_length = len(
            metadata_fixture[0]["Sequence of Base Variant"]
        )
        biophysical_property_final = metadata_fixture[0][
            "Desired Name of Dependent Variable"
        ]

    deduped_sequence_df = check_for_duplicates(
        combined_df=combined_seq_data,
        cols=column_names,
        base_variant_sequence_length=base_variant_sequence_length,
        biophysical_property_final=biophysical_property_final,
    )
    # Data Saving
    assert len(seq_entry.columns) == len(base_variant_sequence_fixture) + 1
    assert len(np.unique(seq_entry.index.values)) == len(seq_entry)
    assert len(combined_seq_data) == sum([len(df) for df in seq_df_list])
    assert len(deduped_sequence_df.columns) == len(base_variant_sequence_fixture) + 2
    # assert (
    #     len(deduped_sequence_df[deduped_sequence_df.duplicated(cols_entry, keep=False)])
    #     == 0
    # )
