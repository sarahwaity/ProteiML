# MAIN PACKAGES
import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm
import pathlib as pl
import logging

from typing import List, Tuple, Union

# MODEL VALIDATION
from sklearn.metrics import mean_squared_error

# MODEL SPECIFIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor

from data_processing.constants import BASE_DIR


def generate_prediction_data(
    novel_library_path: pl.Path,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """loads in the novel library and checks it for duplicate sequences

    Args:
        novel_library_path (pl.Path): path to the location of novel library

    Returns:
        Tuple[pd.DataFrame, Union[pd.DataFrame,None]]: novel library and df of duplicate sequences 
    """
    data_pred = pd.read_csv(novel_library_path, index_col=0)

    # Remove Any Duplicate Sequences (redundancies with 7s)
    data_pred = data_pred.drop_duplicates(
        subset=data_pred.columns[2:-2], keep="first", inplace=False, ignore_index=False
    )
    dropped = data_pred[data_pred.duplicated(subset=data_pred.columns[2:-2])]

    return data_pred, dropped


def create_encoded_names(top_encoding_dataset_path: pl.Path) -> List[str]:
    """generates a list of top encoding datasets based off results from model training. 

    Args:
        top_encoding_dataset_path (pl.Path): path to top encoding dataset, seperate one for each regressor

    Returns:
        List[str]: list of the top five encoding datasets
    """
    # read in the encoding datasets:
    top_encoding_data = pd.read_csv(top_encoding_dataset_path, index_col=0)
    top_encoding_data = top_encoding_data.sort_values(
        by="Test Set R Squared", ascending=False
    )
    names = [e for e in top_encoding_data["Encoding Dataset"][0:5]]

    return names


def retrain_predict_knr(
    names: List[str],
    encoding_data: pd.DataFrame,
    data_pred: pd.DataFrame,
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
    dependent_variable: str,
):
    """retrains the KNR model with each of the top encoding datasets and uses them to generate predictions on the novel mutant library and the withheld test set. 

    Args:
        names (List[str]): list of the top performing datasets
        encoding_data (pd.DataFrame): dataframe that contains all amino acid property matricies
        data_pred (pd.DataFrame): sequence columns of the novel mutant library
        x_train (np.array): sequence columns from the train set 
        x_test (np.array): sequence columns from the test set 
        y_train (np.array): dependent variable of the train set
        y_test (np.array): dependent variable of the test set
        dependent_variable (str): name of the biophyscial property (dependent variable) being trained on 
    """
    save_path = BASE_DIR / "knr"
    # initialize output dataframe
    df_KNR = pd.DataFrame()

    logging.info("Retraining and getting predictions from KNR...")
    # Iterate through every encoding dataset + train model +
    # record performance
    for AA_property_dataset in tqdm(names, desc="Property Datasets Encoded:"):
        # Make the train/test be on a copy to ensure there
        # is no data overwriting within loop
        x_train_copy = x_train.copy()
        x_test_copy = x_test.copy()
        x_pred_copy = data_pred.copy()

        # extract encoding data for specific iteration
        volume_dict = {
            "Amino Acid Code": encoding_data[encoding_data.columns[0]],
            AA_property_dataset[-10:]: encoding_data['AAindex: '+ AA_property_dataset],
        }
        volume_data = pd.DataFrame(volume_dict)

        # Some encoding datasets contain NaNs, I skip these
        # datasets since theyre incomplete
        # Next three lines check the encoding data for NaNs
        df = list(volume_data.iloc[:, 1].values)
        T = np.isnan(df)
        TF = True in T

        # If there is no NaN, perform model training
        if TF == False:
            # initialize list to append to throughout training
            interlist = []

            # Use volume_data as a codex to translate sequence data...
            # amino acids will translate to float type data
            col_title = volume_data.columns[1]
            for row, sample in enumerate(volume_data["Amino Acid Code"]):
                amino = sample
                replacement_value = float(volume_data[col_title].iloc[row])
                x_train_copy = x_train_copy.replace(amino, replacement_value)
                x_test_copy = x_test_copy.replace(amino, replacement_value)
                x_pred_copy = x_pred_copy.replace(amino, replacement_value)

            # apply SelectKBest class to extract best features
            bestfeatures = SelectKBest(score_func=mutual_info_regression, k="all")
            fit = bestfeatures.fit(x_train_copy, y_train)
            dfscores = pd.DataFrame(fit.scores_)
            dfcolumns = pd.DataFrame(x_train_copy.columns)

            # concat two dataframes for better visualization
            featureScores = pd.concat([dfcolumns, dfscores], axis=1)
            featureScores.columns = ["Specs", "Score"]  # naming the dataframe columns

            ##Hyper Parameter Tuning
            # Grid Search approach, I test every iteration to find the best
            # combination of neighbors and features

            # Initialize the output lists to append to
            n_neigh = []
            test_r2s = []

            # For 21 possible best features
            for l in range(21):
                if l > 0:
                    cols = featureScores.nlargest(l, "Score")

                    # extract l features from X_train/X_test
                    x_train_copy_ = x_train_copy[list(cols["Specs"].values)]
                    x_test_copy_ = x_test_copy[list(cols["Specs"].values)]

                    # Initialize the output lists to append to
                    n_mse_list = []
                    n_r2_list = []

                    # inialize list of neighbors
                    n_neighbors = [1, 2, 3, 4, 5, 6, 7]

                    for neighbors in n_neighbors:
                        # initialize model with n number of neighbors
                        np.random.seed(42)
                        clf_RF = KNeighborsRegressor(n_neighbors=neighbors)
                        # Fit the Train data
                        clf_RF.fit(x_train_copy_, y_train)
                        # Predict the Test set and generate metrics of fit
                        y_RF = clf_RF.predict(x_test_copy_)
                        n_mse_list.append(mean_squared_error(y_test, y_RF))
                        n_r2_list.append(sklearn.metrics.r2_score(y_test, y_RF))

                    # find which number of neighbors led to the best R2 for
                    # the test set
                    best_neigh = n_neighbors[n_mse_list.index(min(n_mse_list))]
                    n_neigh.append(best_neigh)
                    test_r2s.append(np.mean(n_r2_list))

            # find which iteration led to the greatest R2
            # neigh_best is the number of neighbors when the
            # max R2 occured
            best_feats = test_r2s.index(max(test_r2s))
            neigh_best = n_neigh[best_feats]

            # similarly, extract the optimal number of features
            # by extracting the number of features that led to the
            # greatest R2
            cols = featureScores.nlargest(best_feats + 1, "Score")
            x_pred_copy_cols = [str(x) for x in list(cols["Specs"].values)]
            x_train_copy_ = x_train_copy[list(cols["Specs"].values)]
            x_test_copy_ = x_test_copy[list(cols["Specs"].values)]
            x_pred_copy_ = x_pred_copy[x_pred_copy_cols]

            # initialize new model + fit data
            np.random.seed(42)
            clf = KNeighborsRegressor(n_neighbors=neigh_best)
            clf.fit(x_train_copy_, y_train)

            # Create cross validation prediction
            y_pred = clf.predict(x_test_copy_)

            # Save the overall R2 for the tuned model
            r2 = sklearn.metrics.r2_score(y_test, y_pred)

            # append the data to a dataframe for export
            inter_df = pd.DataFrame(
                {
                    "Encoding Dataset": [AA_property_dataset[-10:]],
                    "Test Set R Squared": [r2],
                }
            )

            df_KNR = pd.concat([df_KNR, inter_df], ignore_index=True)

            # Novel Library Predictions
            y_pred_new = clf.predict(x_pred_copy_)
            data_pred[dependent_variable + " Predicted"] = y_pred_new
            data_pred = data_pred.sort_values(
                by=dependent_variable + " Predicted", ascending=False
            )
            newmut = pd.DataFrame(data_pred)

            # Test Set Predictions
            test_set_df = pd.DataFrame()
            test_set_df["True"] = y_test
            test_set_df["Predicted"] = y_pred

            # file saving
            newmut.to_csv(
                save_path / f"{AA_property_dataset[-10:]}_novel_library_predictions.csv"
            )
            test_set_df.to_csv(
                save_path / f"{AA_property_dataset[-10:]}_test_set_predictions.csv"
            )


def retrain_predict_mpnr(
    names: List[str],
    encoding_data: pd.DataFrame,
    encoded_df: pd.DataFrame,
    data_pred: pd.DataFrame,
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
    dependent_variable: str,
):
    """retrains the MPNR model with each of the top encoding datasets and uses them to generate predictions on the novel mutant library and the withheld test set.

    Args:
        names (List[str]): list of the top performing datasets
        encoding_data (pd.DataFrame): dataframe that contains all amino acid property matricies
        encoded_df (pd.DataFrame): sequences of full known mutant library (not train/test split)
        data_pred (pd.DataFrame): sequence columns of the novel mutant library
        x_train (np.array): sequence columns from the train set 
        x_test (np.array): sequence columns from the test set 
        y_train (np.array): dependent variable of the train set
        y_test (np.array): dependent variable of the test set
        dependent_variable (str): name of the biophyscial property (dependent variable) being trained on 
    """
    save_path = BASE_DIR / "mpnr"
    # initialize output dataframe
    df_MPNR = pd.DataFrame()

    logging.info("Retraining and getting predictions from MPNR...")
    # Iterate through every encoding dataset + train model +
    # record performance
    for AA_property_dataset in tqdm(names, desc="Property Datasets Encoded:"):
        # Make the train/test be on a copy to ensure there
        # is no data overwriting within loop
        x_train_copy = x_train.copy()
        x_test_copy = x_test.copy()
        x_pred_copy = data_pred.copy()
        x_full = encoded_df.copy()

        # extract encoding data for specific iteration
        volume_dict = {
            "Amino Acid Code": encoding_data[encoding_data.columns[0]],
            AA_property_dataset[-10:]: encoding_data['AAindex: '+ AA_property_dataset],
        }
        volume_data = pd.DataFrame(volume_dict)

        # Some encoding datasets contain NaNs, I skip these
        # datasets since theyre incomplete
        # Next three lines check the encoding data for NaNs
        df = list(volume_data.iloc[:, 1].values)
        T = np.isnan(df)
        TF = True in T

        # If there is no NaN, perform model training
        if TF == False:
            # initialize list to append to throughout training
            interlist = []

            # Use volume_data as a codex to translate sequence data...
            # amino acids will translate to float type data
            col_title = volume_data.columns[1]
            for row, sample in enumerate(volume_data["Amino Acid Code"]):
                amino = sample
                replacement_value = float(volume_data[col_title].iloc[row])
                x_train_copy = x_train_copy.replace(amino, replacement_value)
                x_test_copy = x_test_copy.replace(amino, replacement_value)
                x_pred_copy = x_pred_copy.replace(amino, replacement_value)
                x_full = x_full.replace(amino, replacement_value)

            # Data Scaling
            scaler = RobustScaler()
            scaler.fit(x_full)
            x_train_copy = scaler.transform(x_train_copy)
            x_test_copy = scaler.transform(x_test_copy)

            x_train_copy = pd.DataFrame(x_train_copy, columns=x_full.columns)
            x_test_copy = pd.DataFrame(x_test_copy, columns=x_full.columns)

            # apply SelectKBest class to extract best features
            bestfeatures = SelectKBest(score_func=mutual_info_regression, k="all")
            fit = bestfeatures.fit(x_train_copy, y_train)
            dfscores = pd.DataFrame(fit.scores_)
            dfcolumns = pd.DataFrame(x_train_copy.columns)

            # concat two dataframes for better visualization
            featureScores = pd.concat([dfcolumns, dfscores], axis=1)
            featureScores.columns = ["Specs", "Score"]  # naming the dataframe columns

            ##Hyper Parameter Tuning
            # Grid Search approach, I test every iteration to find the best
            # number of features

            # Initialize the output lists to append to
            test_r2s = []

            # For 25 possible best features
            for l in range(25):
                if l > 0:
                    cols = featureScores.nlargest(l, "Score")

                    # extract l features from X_train/X_test
                    x_train_copy_ = x_train_copy[list(cols["Specs"].values)]
                    x_test_copy_ = x_test_copy[list(cols["Specs"].values)]

                    # initialize model
                    clf_RF = MLPRegressor(random_state=42)
                    # Fit the Train data
                    clf_RF.fit(x_train_copy_, y_train)
                    # Predict the Test set and generate metrics of fit
                    y_RF = clf_RF.predict(x_test_copy_)
                    test_r2s.append(sklearn.metrics.r2_score(y_test, y_RF))

            # find which number of feature led to the
            # max R2, skip the zeroith index
            best_feats = test_r2s.index(max(test_r2s))

            # Extract the features that led to the best performance
            cols = featureScores.nlargest(best_feats + 1, "Score")
            x_pred_copy_cols = [str(x) for x in list(cols["Specs"].values)]
            x_train_copy_ = x_train_copy[list(cols["Specs"].values)]
            x_test_copy_ = x_test_copy[list(cols["Specs"].values)]
            x_pred_copy_ = x_pred_copy[x_pred_copy_cols]

            # initialize new model + fit data
            clf = MLPRegressor(random_state=42)
            clf.fit(x_train_copy_, y_train)

            # Create cross validation prediction
            y_pred = clf.predict(x_test_copy_)

            # Save the overall R2 for the tuned model
            r2 = sklearn.metrics.r2_score(y_test, y_pred)

            # append the data to a dataframe for export
            inter_df = pd.DataFrame(
                {
                    "Encoding Dataset": [AA_property_dataset[-10:]],
                    "Test Set R Squared": [r2],
                }
            )

            df_MPNR = pd.concat([df_MPNR, inter_df], ignore_index=True)

            # Novel Library Predictions
            y_pred_new = clf.predict(x_pred_copy_)
            data_pred[dependent_variable + " Predicted"] = y_pred_new
            data_pred = data_pred.sort_values(
                by=dependent_variable + " Predicted", ascending=False
            )
            newmut = pd.DataFrame(data_pred)

            # Test Set Predictions
            test_set_df = pd.DataFrame()
            test_set_df["True"] = y_test
            test_set_df["Predicted"] = y_pred

            # file saving
            newmut.to_csv(
                save_path / f"{AA_property_dataset[-10:]}_novel_library_predictions.csv"
            )
            test_set_df.to_csv(
                save_path / f"{AA_property_dataset[-10:]}_test_set_predictions.csv"
            )


def retrain_predict_rfr(
    names: List[str],
    encoding_data: pd.DataFrame,
    data_pred: pd.DataFrame,
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
    dependent_variable: str,
):
    """Retrains the RFR model with each of the top encoding datasets and uses them to generate predictions on the novel mutant library and the withheld test set.

    Args:
        names (List[str]): list of the top performing datasets
        encoding_data (pd.DataFrame): dataframe that contains all amino acid property matricies
        data_pred (pd.DataFrame): sequence columns of the novel mutant library
        x_train (np.array): sequence columns from the train set 
        x_test (np.array): sequence columns from the test set 
        y_train (np.array): dependent variable of the train set
        y_test (np.array): dependent variable of the test set
        dependent_variable (str): name of the biophyscial property (dependent variable) being trained on 
    """
    # initialize output dataframe
    df_RFR = pd.DataFrame()
    save_path = BASE_DIR / "rfr/"

    logging.info("Retraining and getting predictions from RFR...")
    # Iterate through every encoding dataset + train model +
    # record performance
    for AA_property_dataset in tqdm(names, desc="Property Datasets Encoded:"):
        # Make the train/test be on a copy to ensure there
        # is no data overwriting within loop
        x_train_copy = x_train.copy()
        x_test_copy = x_test.copy()
        x_pred_copy = data_pred.copy()

        # extract encoding data for specific iteration
        volume_dict = {
            "Amino Acid Code": encoding_data[encoding_data.columns[0]],
            AA_property_dataset[-10:]: encoding_data['AAindex: '+ AA_property_dataset],
        }
        volume_data = pd.DataFrame(volume_dict)

        # Some encoding datasets contain NaNs, I skip these
        # datasets since theyre incomplete
        # Next three lines check the encoding data for NaNs
        df = list(volume_data.iloc[:, 1].values)
        T = np.isnan(df)
        TF = True in T

        # If there is no NaN, perform model training
        if TF == False:
            # initialize list to append to throughout training
            interlist = []

            # Use volume_data as a codex to translate sequence data...
            # amino acids will translate to float type data
            col_title = volume_data.columns[1]
            for row, sample in enumerate(volume_data["Amino Acid Code"]):
                amino = sample
                replacement_value = float(volume_data[col_title].iloc[row])
                x_train_copy = x_train_copy.replace(amino, replacement_value)
                x_test_copy = x_test_copy.replace(amino, replacement_value)
                x_pred_copy = x_pred_copy.replace(amino, replacement_value)

            # Initialize Model and fit data to extract feature importances
            model = RandomForestRegressor()
            model.fit(x_train_copy, y_train)
            feat_importances = pd.Series(
                model.feature_importances_, index=x_train_copy.columns
            )

            ##Hyper Parameter Tuning
            # Grid Search approach, I test every iteration to find the best
            # combination of neighbors and features

            # Initialize the output lists to append to
            n_est = []
            test_r2s = []

            # For 21 possible best features
            for l in range(21):
                if l > 0:
                    cols = list(feat_importances.nlargest(l).index)

                    # extract l features from X_train/X_test
                    x_train_copy_ = x_train_copy[cols]
                    x_test_copy_ = x_test_copy[cols]

                    # Initialize list of estimators to test
                    n_estimators = [
                        10,
                        15,
                        20,
                        25,
                        30,
                        35,
                        40,
                        45,
                        50,
                        55,
                        65,
                        75,
                        85,
                        100,
                    ]

                    # Initialize the output lists to append to
                    n_mse_list = []
                    n_r2_list = []

                    for estimators in n_estimators:
                        # initialize model with n number of estimators
                        clf_RF = RandomForestRegressor(
                            n_estimators=estimators, random_state=42
                        )
                        # Fit the Train data
                        clf_RF.fit(x_train_copy_, y_train)
                        # Predict the Test set and generate metrics of fit
                        y_RF = clf_RF.predict(x_test_copy_)
                        n_mse_list.append(mean_squared_error(y_test, y_RF))
                        n_r2_list.append(sklearn.metrics.r2_score(y_test, y_RF))

                    # find which number of estimators led to the best R2 for
                    # the test set
                    best_est = n_estimators[n_mse_list.index(min(n_mse_list))]
                    n_est.append(best_est)
                    test_r2s.append(np.mean(n_r2_list))

            # find which iteration led to the greatest R2
            # est_best is the number of estimators when the
            # max R2 occured
            best_feats = test_r2s.index(max(test_r2s))
            est_best = n_est[best_feats]

            # similarly, extract the optimal number of features
            # by extracting the number of features that led to the
            # greatest R2
            cols = list(feat_importances.nlargest(best_feats + 1).index)
            x_train_copy_ = x_train_copy[cols]
            x_test_copy_ = x_test_copy[cols]
            x_pred_copy_ = x_pred_copy[[str(x) for x in cols]]

            # initialize new model + fit data
            clf = RandomForestRegressor(n_estimators=est_best, random_state=42)
            clf.fit(x_train_copy_, y_train)

            # Create cross validation prediction
            y_pred = clf.predict(x_test_copy_)

            # Save the overall R2 for the tuned model
            r2 = sklearn.metrics.r2_score(y_test, y_pred)

            # append the data to a dataframe for export
            inter_df = pd.DataFrame(
                {
                    "Encoding Dataset": [AA_property_dataset[-10:]],
                    "Test Set R Squared": [r2],
                }
            )

            df_RFR = pd.concat([df_RFR, inter_df], ignore_index=True)

            # Novel Library Predictions
            y_pred_new = clf.predict(x_pred_copy_)
            data_pred[dependent_variable + " Predicted"] = y_pred_new
            data_pred = data_pred.sort_values(
                by=dependent_variable + " Predicted", ascending=False
            )
            newmut = pd.DataFrame(data_pred)

            # Test Set Predictions
            test_set_df = pd.DataFrame()
            test_set_df["True"] = y_test
            test_set_df["Predicted"] = y_pred

            newmut.to_csv(
                save_path / f"{AA_property_dataset[-10:]}_novel_library_predictions.csv"
            )
            test_set_df.to_csv(
                save_path / f"{AA_property_dataset[-10:]}_test_set_predictions.csv"
            )
