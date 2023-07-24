# MAIN PACKAGES
import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm
from typing import Tuple

# MODEL VALIDATION
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# KNN SPECIFIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor


def prepare_encoding_data(
    combined_dataset: pd.DataFrame,
    len_sequence_of_base_variant: int,
    dependent_var: str,
) -> Tuple[np.array, np.array, np.array, np.array, pd.DataFrame]:
    """ generates a train + test sets for ML algorithm training

    Args:
        combined_dataset (pd.DataFrame): full known mutant library (formatted by format_mutation_dataset.py)
        len_sequence_of_base_variant (int): length of base variant sequence, used to form column lists
        dependent_var (str): chosen biophysical property for optimization

    Returns:
        Tuple[np.array, np.array, np.array, np.array, pd.DataFrame]: exports of the x/y train + test sets as well as the full X data. 
    """
    position_cols = np.arange(0, len_sequence_of_base_variant)

    encoded_df = combined_dataset[position_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        encoded_df,
        combined_dataset[dependent_var],
        test_size=0.20,
        random_state=42,
    )

    return x_train, x_test, y_train, y_test, encoded_df


def train_knr_model(
    encoding_data: pd.DataFrame,
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
) -> pd.DataFrame:
    """trains the KNR model over every property matrix and marks the property matrix and R Squared achieved. 

    Args:
        encoding_data (pd.DataFrame): df that contains of the amino acid property matricies from AAINDEX 
        x_train (np.array): sequence data (independent variable) for the train set 
        x_test (np.array): sequence data (independent variable) for the test set 
        y_train (np.array): functional variabilites (dependent variable) for the train set
        y_test (np.array): functional variabilites (dependent variable) for the test set

    Returns:
        pd.DataFrame: dataframe that contains each property matrix used for encoding linked to the R2 value they achieved. 
    """
    # initialize output dataframe
    df_KNR = pd.DataFrame()

    # Iterate through every encoding dataset + train model +
    # record performance
    for AA_property_dataset in tqdm(
        encoding_data.columns[1:], desc="Property Datasets Encoded:"
    ):
        # Make the train/test be on a copy to ensure there
        # is no data overwriting within loop
        x_train_copy = x_train.copy()
        x_test_copy = x_test.copy()

        # extract encoding data for specific iteration
        volume_dict = {
            "Amino Acid Code": encoding_data[encoding_data.columns[0]],
            AA_property_dataset: encoding_data[AA_property_dataset],
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
            x_train_copy_ = x_train_copy[list(cols["Specs"].values)]
            x_test_copy_ = x_test_copy[list(cols["Specs"].values)]

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
                    "Encoding Dataset": [AA_property_dataset[-11:]],
                    "Test Set R Squared": [r2],
                }
            )

            df_KNR = pd.concat([df_KNR, inter_df], ignore_index=True)

    df_KNR = df_KNR.sort_values(by="Test Set R Squared", ascending=False)

    return df_KNR


def train_mpnr_model(
    encoding_data: pd.DataFrame,
    encoded_df: pd.DataFrame,
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
) -> pd.DataFrame:
    """trains the KNR model over every property matrix and marks the property matrix and R Squared achieved. 

    Args:
        encoding_data (pd.DataFrame): df that contains of the amino acid property matricies from AAINDEX
        encoded_df (pd.DataFrame): full sequence information for train+test sets (independent variable unsplit)
        x_train (np.array): sequence data (independent variable) for the train set 
        x_test (np.array): sequence data (independent variable) for the test set 
        y_train (np.array): functional variabilites (dependent variable) for the train set
        y_test (np.array): functional variabilites (dependent variable) for the test set

    Returns:
        pd.DataFrame: dataframe that contains each property matrix used for encoding linked to the R2 value they achieved. 
    """
    # initialize output dataframe
    df_MPNR = pd.DataFrame()

    # Iterate through every encoding dataset + train model +
    # record performance
    for AA_property_dataset in tqdm(
        encoding_data.columns[1:], desc="Property Datasets Encoded:"
    ):
        # Make the train/test be on a copy to ensure there
        # is no data overwriting within loop
        x_train_copy = x_train.copy()
        x_test_copy = x_test.copy()
        x_full = encoded_df.copy()

        # extract encoding data for specific iteration
        volume_dict = {
            "Amino Acid Code": encoding_data[encoding_data.columns[0]],
            AA_property_dataset: encoding_data[AA_property_dataset],
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
            x_train_copy_ = x_train_copy[list(cols["Specs"].values)]
            x_test_copy_ = x_test_copy[list(cols["Specs"].values)]

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
                    "Encoding Dataset": [AA_property_dataset[-11:]],
                    "Test Set R Squared": [r2],
                }
            )

            df_MPNR = pd.concat([df_MPNR, inter_df], ignore_index=True)

    df_MPNR = df_MPNR.sort_values(by="Test Set R Squared", ascending=False)
    return df_MPNR


def train_rfr_model(
    encoding_data: pd.DataFrame,
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
) -> pd.DataFrame:
    """trains the RFR model over every property matrix and marks the property matrix and R Squared achieved. 

    Args:
        encoding_data (pd.DataFrame): df that contains of the amino acid property matricies from AAINDEX 
        x_train (np.array): sequence data (independent variable) for the train set 
        x_test (np.array): sequence data (independent variable) for the test set 
        y_train (np.array): functional variabilites (dependent variable) for the train set
        y_test (np.array): functional variabilites (dependent variable) for the test set

    Returns:
        pd.DataFrame: dataframe that contains each property matrix used for encoding linked to the R2 value they achieved. 
    """
    # initialize output dataframe
    df_RFR = pd.DataFrame()

    # Iterate through every encoding dataset + train model +
    # record performance
    for AA_property_dataset in tqdm(
        encoding_data.columns[1:], desc="Property Datasets Encoded:"
    ):
        # Make the train/test be on a copy to ensure there
        # is no data overwriting within loop
        x_train_copy = x_train.copy()
        x_test_copy = x_test.copy()

        # extract encoding data for specific iteration
        volume_dict = {
            "Amino Acid Code": encoding_data[encoding_data.columns[0]],
            AA_property_dataset: encoding_data[AA_property_dataset],
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
                    "Encoding Dataset": [AA_property_dataset[-11:]],
                    "Test Set R Squared": [r2],
                }
            )

            df_RFR = pd.concat([df_RFR, inter_df], ignore_index=True)

    df_RFR = df_RFR.sort_values(by="Test Set R Squared", ascending=False)

    return df_RFR
