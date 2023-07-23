# Running this shell will install/import necessary libraries to format dataseet.

import pandas as pd
import numpy as np
import logging
from typing import List
from main import STANDARD_DATA_DIR


def get_col_count(base_variant_seq, biophysical_property_final):
    cols = list(np.arange(0, len(base_variant_seq)))
    cols.append("Variant ID")
    cols.append(biophysical_property_final)
    return cols


def read_seq_data(metadata):
    # read in mutation library for GCaMP6
    seq_data = pd.read_csv(STANDARD_DATA_DIR / metadata["Mutant Data File Path"])

    # isolate the column that contains the mutations added to each variant
    mutation_data = seq_data[metadata["Mutations Column"]]

    # splits the string into individual mutations
    for i in range(len(mutation_data)):
        txt_mut = str(mutation_data[i])
        if txt_mut == "base":
            mutation_data.at[i] = [""]
        else:
            x = txt_mut.split()
            mutation_data.at[i] = x

    # saves the mutation information with the variants in which they belong...
    # in addition to the biophyscial characteristic
    mutation_df = pd.DataFrame(
        {
            "Mutations": mutation_data,
            "Variant ID": seq_data[metadata["Name of Index Column"]],
            metadata["Desired Name of Dependent Variable"]: seq_data[
                metadata["Biophysical Property Column Name"]
            ],
        }
    )

    # columns for output dataframe
    cols = get_col_count(
        metadata["Sequence of Base Variant"], metadata["Desired Name of Dependent Variable"]
    )

    ##Formatting Mutation Strings To Residue/Amino Acid Format

    list_of_mutation_location = []  # used to save position information
    list_of_mutation_aa = []  # used to save amino acid information
    for row in range(len(mutation_df)):  # takes each variant found in the mutation df
        # pulls the mutations found in each variant (type: list of strings)
        mutation_location = mutation_df["Mutations"].iloc[row]

        # interloop datasaving locations
        residue_list = []
        aa_change_list = []

        # find the residue locations and to which amino acid the mutation was made
        for iterator in range(
            len(mutation_location)
        ):  # takes each mutation found in single variant
            # isolate mutation at iterator
            position_mutation_location = mutation_location[iterator]
            # finds all of the digit values in the mutation and joins them to ...
            # isolate residue location
            numeric_string = "".join(filter(str.isdigit, position_mutation_location))

            if (
                numeric_string == ""
            ):  # this case would only happen if the construct is base
                residue_list.append("")  # no residue locations to mutate
                aa_change_list.append("")  # no amino acids to mutate to
            else:
                # returns int type of residue location
                residue_list.append(int(numeric_string))
                # returns str type of final amino acid mutation
                aa_change_list.append(mutation_location[iterator][-1])

        # Save interloop list to exterior datasaving list
        # expected: len(residue_list) = len(mutation_location)
        # expected: len(list_of_mutation_location) = len(mutation_df)
        # expected: len(list_of_mutation_aa) = len(mutation_df)
        list_of_mutation_location.append(residue_list)
        list_of_mutation_aa.append(aa_change_list)

    # write exterior saving locations to Pandas Series in maintain index information
    appending_list_mutation_location = pd.Series(
        list_of_mutation_location, index=mutation_df.index
    )
    appending_list_mutation_aa = pd.Series(list_of_mutation_aa, index=mutation_df.index)

    df_seq = pd.DataFrame()

    for row in range(len(mutation_df)):
        # initialize the base sequence for each loop
        x = [e for e in metadata["Sequence of Base Variant"]]

        # isolates the mutated residues/amino acids for each row
        mutation_loc = appending_list_mutation_location[row]
        mutation_aa = appending_list_mutation_aa[row]

        if type(mutation_loc[0]) is int:  # tests to see if row is the base construct
            for mut in range(len(mutation_loc)):
                x[mutation_loc[mut]] = mutation_aa[
                    mut
                ]  # inplace mutation onto base sequence
            x.append(
                mutation_df["Variant ID"].loc[row]
            )  # append the variants primary key
            x.append(
                mutation_df[metadata["Desired Name of Dependent Variable"]].loc[row]
            )  # append variant's dependent information
            concat_df = pd.DataFrame([x], columns=cols)
            df_seq = pd.concat(
                [df_seq, concat_df]
            )  # append row's dataframe with external dataframe

        else:
            x.append(
                mutation_df["Variant ID"].loc[row]
            )  # append the variants primary key
            x.append(
                mutation_df[metadata["Desired Name of Dependent Variable"]].loc[row]
            )  # append variant's dependent information
            concat_df = pd.DataFrame([x], columns=cols)
            df_seq = pd.concat(
                [df_seq, concat_df]
            )  # append row's dataframe with external dataframe

    # renormalize chen dataset to GCaMP6s == 1.0 for 1 AP
    # find the value for GCaMP6s
    g6s_data = df_seq[df_seq["Variant ID"] == metadata["Variant For Normalization"]][
        metadata["Desired Name of Dependent Variable"]
    ].values[0]
    # divide the biophysical property column by the GCaMP6s value
    df_seq[metadata["Desired Name of Dependent Variable"]] = (
        df_seq[metadata["Desired Name of Dependent Variable"]] / g6s_data
    )

    df_seq.set_index("Variant ID", append=False, inplace=True)
    return df_seq, cols


def combine_seq_datasets(df_list: List[pd.DataFrame]):
    ## Combining the Two Datasets:
    # concatenate the two sequence libraries
    combined_df = pd.DataFrame()
    for df in df_list:
        combined_df = pd.concat([combined_df, df])
    combined_df["Variant ID"] = combined_df.index
    logging.info("Combined Length:" + str(len(combined_df)))
    return combined_df


def check_for_duplicates(
    combined_df, cols, base_variant_sequence_length, biophysical_property_final
):
    ## Duplicate Values Check

    cols_1 = list(np.arange(0, base_variant_sequence_length))

    # isolate rows that contain duplicated values
    duplicated_seq_df = combined_df[combined_df.duplicated(cols_1, keep=False)]

    if (len(duplicated_seq_df)) > 0:
        logging.info(
            "Found "
            + str(len(duplicated_seq_df))
            + " duplicated rows! Cleaning up data now!"
        )

        # isolate just the full sequence
        duplicated_seq_df["full seq"] = [
            "".join(list(duplicated_seq_df.copy().loc[:, cols_1].iloc[e].values))
            for e in range(len(duplicated_seq_df))
        ]

        # aggregate the data based on the full sequence & group data by mean of group
        x = duplicated_seq_df.groupby("full seq")
        y = duplicated_seq_df.groupby("full seq").mean(numeric_only=True)

        # create new dataframe with aggregated samples
        new_df = pd.DataFrame(columns=cols)
        # isolate groups of duplicated data
        for index in y.index:
            duplicated_sequences = x.get_group(index)

            # give them a new variant ID
            new_variant_id = list(duplicated_sequences["Variant ID"].values)
            # find the average performance from all the duplicated variants
            averaged_prop = np.mean(
                list(duplicated_sequences[biophysical_property_final].values)
            )
            # isolate the sequence + append information
            sequence_list = list(duplicated_sequences[cols_1].iloc[0].values)
            sequence_list.append(new_variant_id)
            sequence_list.append(averaged_prop)
            # append Data to external save dataframe
            new_df = pd.concat(
                [new_df, pd.DataFrame([sequence_list], columns=cols)], ignore_index=True
            )

        df_seq = pd.concat([combined_df, new_df])
        df_seq = df_seq.drop_duplicates(cols_1, keep="last")

        logging.info(f"De-duplication successful. Final dimensions = {df_seq.shape}.")
        return df_seq
    else:
        logging.info("No duplicate rows found!")
        return combined_df
