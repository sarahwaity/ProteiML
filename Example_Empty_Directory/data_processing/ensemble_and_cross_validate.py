import pandas as pd
import numpy as np
import pathlib as pl
from typing import List, Dict

from scipy import stats

import sklearn

from data_processing.constants import BASE_DIR, OUTPUT_DIR


def isolate_encoding_datasets(
    rfr_encoding_dataset_path: pl.Path,
    mpnr_encoding_dataset_path: pl.Path,
    knr_encoding_dataset_path: pl.Path,
) -> Dict[str, List[str]]:
    """Generates a dict with the names of the top encoding datasets for each regressor type.

    Args:
        rfr_encoding_dataset_path (pl.Path): path to rfr_encoding_dataset, (output of model training)
        mpnr_encoding_dataset_path (pl.Path): path to mpnr_encoding_dataset, (output of model training)
        knr_encoding_dataset_path (pl.Path): path to knr_encoding_dataset, (output of model training)

    Returns:
        Dict[str, List[str]]: dict that contains the names of the top encoding datsets for rfr, mpnr, and knr
    """
    # isolate top RFR encoding datasets
    rfr_top_encoding_data = pd.read_csv(rfr_encoding_dataset_path, index_col=0)
    rfr_top_encoding_data = rfr_top_encoding_data.sort_values(
        by="Test Set R Squared", ascending=False
    )
    rfr_names = [e[-10:] for e in rfr_top_encoding_data["Encoding Dataset"][0:5]]

    # isolate top MPNR encoding datasets
    mpnr_top_encoding_data = pd.read_csv(mpnr_encoding_dataset_path, index_col=0)
    mpnr_top_encoding_data = mpnr_top_encoding_data.sort_values(
        by="Test Set R Squared", ascending=False
    )
    mpnr_names = [e[-10:] for e in mpnr_top_encoding_data["Encoding Dataset"][0:5]]

    # isolate top KNR encoding datasets
    knr_top_encoding_data = pd.read_csv(knr_encoding_dataset_path, index_col=0)
    knr_top_encoding_data = knr_top_encoding_data.sort_values(
        by="Test Set R Squared", ascending=False
    )
    knr_names = [e[-10:] for e in knr_top_encoding_data["Encoding Dataset"][0:5]]

    dict_of_names = {"rfr": rfr_names, "mpnr": mpnr_names, "knr": knr_names}

    return dict_of_names


def create_ensemble_dataset(
    dict_of_names: Dict[str, List[str]],
    len_base_variant_sequence: int,
    dependent_variable: str,
    comparison_variant: str,
) -> pd.DataFrame:
    """reads in the results from each regressors predictions of the novel variant library and averages them to form final ensemble predictions.

    Args:
        dict_of_names (Dict[str, List[str]]): output of isolate_encoding_datasets function
        len_base_variant_sequence (int): length of base variant sequence, used to form column lists
        dependent_variable (str): name of the predicted dependent variable
        comparison_variant (str): name of the base construct, should match base variant used to construct novel library

    Returns:
        pd.DataFrame: Dataframe that contains one column for each contributor model, a column for their averaged predictions, a column of variant IDs, and P-value of the predictions vs those made on the comparison variant. 
    """
    ensemble_df = pd.DataFrame()

    for key in dict_of_names.keys():
        # initialize list of predictions
        predictions_list = []
        for predictions_file in dict_of_names[key]:
            # read in and sort data
            data = pd.read_csv(
                BASE_DIR / key / f"{predictions_file}_novel_library_predictions.csv",
                index_col=0,
            )
            data = data.sort_values(by="mutant_name")

            # one last duplicate values check
            index_cols = [str(e) for e in np.arange(0, len_base_variant_sequence)]
            short_data = data[index_cols]
            cleaned_data = data[short_data.duplicated() == False]
            cleaned_data = cleaned_data.sort_values(by="mutant_name", ascending=False)

            predictions_list.append(
                list(cleaned_data[dependent_variable + " Predicted"])
            )

        # match the predictions_notebook_with_prediction_outputs &
        # save to dataframe
        keys_df = pd.DataFrame(
            {
                key + "_" + dict_of_names[key][0]: predictions_list[0],
                key + "_" + dict_of_names[key][1]: predictions_list[1],
                key + "_" + dict_of_names[key][2]: predictions_list[2],
                key + "_" + dict_of_names[key][3]: predictions_list[3],
                key + "_" + dict_of_names[key][4]: predictions_list[4],
            }
        )

        # save to exterior dataframe for full ensemble
        ensemble_df = pd.concat([ensemble_df, keys_df], axis=1)

        # derive model specific metrics for export
        data_columns = list(keys_df.columns)
        keys_df["Average Prediction"] = [
            np.mean(keys_df.iloc[e]) for e in range(len(keys_df))
        ]
        keys_df["Mutation"] = list(cleaned_data["mutant_name"])
        comparison_list = list(
            keys_df[keys_df["Mutation"] == comparison_variant][data_columns].values[0]
        )
        keys_df["P-Values"] = [
            stats.ttest_ind(
                comparison_list, list(keys_df[data_columns].iloc[e].values)
            ).pvalue
            for e in range(len(keys_df))
        ]

        # save dataframe for each model type
        keys_df = keys_df.sort_values(by="Average Prediction")
        keys_df.to_csv(OUTPUT_DIR / f"{key}_in_silico_predictions_.csv")

    # Derive ensemble metrics such as mean prediction and get pvalues
    # compared to predictions for base construct
    data_columns = list(ensemble_df.columns)
    ensemble_df["Average Prediction"] = [
        np.mean(ensemble_df.iloc[e]) for e in range(len(ensemble_df))
    ]
    ensemble_df["Mutation"] = list(cleaned_data["mutant_name"])
    comparison_list = list(
        ensemble_df[ensemble_df["Mutation"] == comparison_variant][data_columns].values[
            0
        ]
    )
    ensemble_df["P-Values"] = [
        stats.ttest_ind(
            comparison_list, list(ensemble_df[data_columns].iloc[e].values)
        ).pvalue
        for e in range(len(ensemble_df))
    ]

    # export data to csv for downstream usage
    ensemble_df = ensemble_df.sort_values(by="Average Prediction")

    return ensemble_df


def create_r2_validations_dataframe(
    dict_of_names: Dict[str, List[str]]
) -> pd.DataFrame:
    """Generates R squared value for each contributor model, as well as the ensemble. Based off predictions made on the withheld test dataset. 

    Args:
        dict_of_names (Dict[str,List[str]]): output of isolate_encoding_datasets function

    Returns:
        pd.DataFrame: Dataframe with R squared values for all contributor models, each regressor as a whole, and the ensemble. 
    """
    ensemble_df = pd.DataFrame()
    r2_list = []
    column_list = []

    for key in dict_of_names.keys():
        # initialize list of predictions
        predictions_list = []
        for predictions_file in dict_of_names[key]:
            # read in and sort data
            data = pd.read_csv(
                BASE_DIR / key / f"{predictions_file}_test_set_predictions.csv",
                index_col=0,
            )
            predictions_list.append(list(data["Predicted"]))

        # match the predictions_notebook_with_prediction_outputs &
        # save to dataframe
        keys_df = pd.DataFrame(
            {
                key + "_" + dict_of_names[key][0]: predictions_list[0],
                key + "_" + dict_of_names[key][1]: predictions_list[1],
                key + "_" + dict_of_names[key][2]: predictions_list[2],
                key + "_" + dict_of_names[key][3]: predictions_list[3],
                key + "_" + dict_of_names[key][4]: predictions_list[4],
            }
        )

        # save to exterior dataframe for full ensemble
        ensemble_df = pd.concat([ensemble_df, keys_df], axis=1)

        # derive model specific metrics for export
        keys_df[key + " Average Prediction"] = [
            np.mean(keys_df.iloc[e]) for e in range(len(keys_df))
        ]
        keys_df["True"] = list(data["True"])

        for column in keys_df.columns[:-1]:
            x = keys_df[column]
            y = keys_df["True"]
            column_list.append(column)
            r2_list.append(sklearn.metrics.r2_score(x, y))

    # derive model specific metrics for export
    x = [np.mean(ensemble_df.iloc[e]) for e in range(len(ensemble_df))]
    y = list(data["True"])
    column_list.append("Ensemble")
    r2_list.append(sklearn.metrics.r2_score(x, y))

    r2_df = pd.DataFrame(
        {"Regressor + Encoding Dataset": column_list, "R2 Performance Score": r2_list}
    )

    return r2_df
