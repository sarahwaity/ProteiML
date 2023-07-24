import data_processing.model_training as mt
import logging
import pytest
import pandas as pd
from data_processing.constants import BACKEND_DATA_DIR, INTERMEDIATE_DATA_DIR



@pytest.fixture
def len_sequence_of_base_variant_fixture():
    return 451

@pytest.fixture
def dependent_var_fixture():
    return "Decay half time (10 AP)"

@pytest.fixture
def var_id_fixture():
    return "Variant ID"

@pytest.fixture
def combined_df_fixture(dependent_var_fixture, var_id_fixture):
    df = pd.read_csv(INTERMEDIATE_DATA_DIR / "combined_dataset.csv", index_col=0)
    new_cols = [int(col) for col in df.columns[:-2]] + [dependent_var_fixture,var_id_fixture]
    df.columns = new_cols
    return df

@pytest.fixture
def generate_training_data_fixture(
    combined_df_fixture, len_sequence_of_base_variant_fixture, dependent_var_fixture
):
    
    
    x_train, x_test, y_train, y_test, encoded_df = mt.prepare_encoding_data(
        combined_df_fixture, len_sequence_of_base_variant_fixture, dependent_var_fixture
    )
    return [x_train, x_test, y_train, y_test, encoded_df]


@pytest.fixture
def encoding_fixture():
    return pd.read_csv(BACKEND_DATA_DIR / "streamlined_property_matrix.csv", index_col=0)

def test_knr_training(encoding_fixture, generate_training_data_fixture):
    logging.info("Training KNR Model...")
    knr_df = mt.train_knr_model(
        encoding_fixture,
        generate_training_data_fixture[0],
        generate_training_data_fixture[1],
        generate_training_data_fixture[2],
        generate_training_data_fixture[3],
    )
    assert len(knr_df) == len(encoding_fixture.columns) - 1
    assert knr_df["Test Set R Squared"].iloc[0] >= knr_df["Test Set R Squared"].iloc[1]

def test_mpnr_training(encoding_fixture, generate_training_data_fixture):
    logging.info("Training MPNR Model...")
    mpnr_df = mt.train_mpnr_model(
        encoding_fixture,
        generate_training_data_fixture[4],
        generate_training_data_fixture[0],
        generate_training_data_fixture[1],
        generate_training_data_fixture[2],
        generate_training_data_fixture[3],
    )
    assert len(mpnr_df) == len(encoding_fixture.columns) - 1
    assert mpnr_df["Test Set R Squared"].iloc[0] >= mpnr_df["Test Set R Squared"].iloc[1]


def test_rfr_training(encoding_fixture, generate_training_data_fixture):
    logging.info("Training RFR Model...")
    rfr_df = mt.train_rfr_model(
        encoding_fixture,
        generate_training_data_fixture[0],
        generate_training_data_fixture[1],
        generate_training_data_fixture[2],
        generate_training_data_fixture[3],
    )
    assert len(rfr_df) == len(encoding_fixture.columns) - 1
    assert (
        rfr_df["Test Set R Squared"].iloc[0] >= rfr_df["Test Set R Squared"].iloc[1]
    )
