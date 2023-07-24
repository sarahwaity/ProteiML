from data_processing.construct_new_library import *
from data_processing.constants import INTERMEDIATE_DATA_DIR, BACKEND_DATA_DIR
import pytest 

@pytest.fixture
def metadata_fixture():
    raw_input = pd.read_csv(BACKEND_DATA_DIR / "Input_Data.csv")

    return raw_input.to_dict(orient="records")


def test_create_and_clean_variant_df_fixture(metadata_fixture):
    reference_variant_for_base_mutation = str(
        metadata_fixture[0]["Reference Index in Mutation Library"]
    )
    variant_name = metadata_fixture[1]["Variant For Optimization Name"]

    variant_df, testing_df, base_info = create_variant_dataframe(
        variant_library_filepath=INTERMEDIATE_DATA_DIR / "combined_dataset.csv",
        reference_variant_for_base_mutation=reference_variant_for_base_mutation,
        variant_name=variant_name,
    )

    cleaned_variants = clean_variant_dataframe(
        variant_df,
        testing_df,
        base_info=base_info,
    )

    assert len(variant_df.columns) == len(metadata_fixture[0]["Sequence of Base Variant"])+1
    assert len(variant_df) > len(cleaned_variants)
    assert len(cleaned_variants[cleaned_variants.duplicated(variant_df.columns[:-1], keep = False)]) == 0
