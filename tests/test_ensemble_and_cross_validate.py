import data_processing.ensemble_and_cross_validate as ecv
from data_processing.constants import BASE_DIR
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def top_encoding_dataset_paths_fixture():
    return {
        "knr": BASE_DIR / "knr/KNR_Encoding_Datasets.csv",
        "mpnr": BASE_DIR / "mpnr/MPNR_Encoding_Datasets.csv",
        "rfr": BASE_DIR / "rfr/RFR_Encoding_Datasets.csv",
    }


@pytest.fixture
def len_sequence_of_base_variant_fixture():
    return 451


@pytest.fixture
def dependent_var_fixture():
    return "Decay half time (10 AP)"


@pytest.fixture
def comparison_variant_fixture():
    return "jGCaMP7s"

@pytest.fixture
def knr_encoding_data_fixture(top_encoding_dataset_paths_fixture):
    knr_top_encoding_data = pd.read_csv(top_encoding_dataset_paths_fixture["knr"], index_col=0)
    knr_top_encoding_data = knr_top_encoding_data.sort_values(
        by="Test Set R Squared", ascending=False
    )
    return knr_top_encoding_data

def test_ensemble_and_r_squared_fit(
    top_encoding_dataset_paths_fixture,
    len_sequence_of_base_variant_fixture,
    dependent_var_fixture,
    comparison_variant_fixture,
    knr_encoding_data_fixture,
):
    name_dict = ecv.isolate_encoding_datasets(
        rfr_encoding_dataset_path=top_encoding_dataset_paths_fixture["rfr"],
        mpnr_encoding_dataset_path=top_encoding_dataset_paths_fixture["mpnr"],
        knr_encoding_dataset_path=top_encoding_dataset_paths_fixture["knr"],
    )

    ensemble_df = ecv.create_ensemble_dataset(
        dict_of_names=name_dict,
        len_base_variant_sequence=len_sequence_of_base_variant_fixture,
        dependent_variable=dependent_var_fixture,
        comparison_variant=comparison_variant_fixture,
    )

    r_squared_df = ecv.create_r2_validations_dataframe(dict_of_names=name_dict)

    assert list(knr_encoding_data_fixture.columns) == ['Encoding Dataset', 'Test Set R Squared']

    assert len(name_dict.keys()) == 3
    assert knr_encoding_data_fixture['Test Set R Squared'].iloc[0]>=knr_encoding_data_fixture['Test Set R Squared'].iloc[1]

    assert len(ensemble_df.columns) == 18
    assert ensemble_df[ensemble_df.columns[0:15]].iloc[0].mean() == ensemble_df['Average Prediction'].iloc[0]

    assert len(r_squared_df) == 19
    assert type(r_squared_df['Regressor + Encoding Dataset'].iloc[0]) == str
    assert type(r_squared_df['R2 Performance Score'].iloc[0]) == np.float64
