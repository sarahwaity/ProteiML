# MAIN PACKAGES
import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm
import logging

# MODEL VALIDATION
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# KNN SPECIFIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

def prepare_encoding_data(combined_dataset, len_sequence_of_base_variant, dependent_var):
    position_cols = np.arange(0, len_sequence_of_base_variant)
    # position_cols = [str(i) for i in position_cols]

    encoded_df = combined_dataset[position_cols]

    # FOR SAM: '1 AP ∆F/F0' is Desired Name of Dependent Variable
    x_train, x_test, y_train, y_test = train_test_split(
        encoded_df,
        combined_dataset[dependent_var],
        test_size=0.20,
        random_state=42,
    )

    return x_train, x_test, y_train, y_test


def train_knr_model(encoding_data, x_train, x_test, y_train, y_test):
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

    df_KNR.sort_values(by="Test Set R Squared", ascending=False)

    return df_KNR
