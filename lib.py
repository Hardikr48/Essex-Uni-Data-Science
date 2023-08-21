
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import glob


from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score

import pickle

warnings.filterwarnings("ignore")


def preprocess_data(raw_data):
    # Exclude the first column
    processed_data = raw_data.iloc[:, 1:]

    # Columns to remove if they are not relevant for Empathy
    cols_to_exclude = ['Mouse position X', 'Mouse position Y', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
                       'Event', 'Event value', 'Computer timestamp', 'Export date', 'Recording date',
                       'Recording date UTC', 'Recording start time', 'Timeline name', 'Recording Fixation filter name',
                       'Recording software version', 'Recording resolution height', 'Recording resolution width',
                       'Recording monitor latency', 'Presented Media width', 'Presented Media height',
                       'Presented Media position X (DACSpx)', 'Presented Media position Y (DACSpx)', 'Original Media width',
                       'Recording start time UTC', 'Original Media height', 'Sensor']

    # Fill forward for columns related to pupil diameter and fixation point
    columns_to_fill_forward = [
        'Pupil diameter left', 'Pupil diameter right', 'Fixation point X', 'Fixation point Y']
    processed_data[columns_to_fill_forward] = processed_data[columns_to_fill_forward].ffill()

    # Columns to be converted to numeric values
    numeric_columns = ['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z',
                       'Gaze direction right X', 'Gaze direction right Y', 'Gaze direction right Z',
                       'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
                       'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)',
                       'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
                       'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)',
                       'Gaze point left X (MCSnorm)', 'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)',
                       'Gaze point right Y (MCSnorm)', 'Pupil diameter left', 'Pupil diameter right']

    # Convert string values to numeric
    for col in numeric_columns:
        processed_data[col] = pd.to_numeric(
            processed_data[col].str.replace(',', '.'), errors='coerce')

    return processed_data

def generate_eye_tracking_summary(data, group):
    # Filter out valid data based on eye validity
    valid_eye_data = data[(data['Validity left'] == 'Valid') &
                          (data['Validity right'] == 'Valid')]

    # Calculate the total count of fixations
    total_fixations_count = data[data['Eye movement type']
                                 == 'Fixation'].shape[0]

    # Compute the average duration of fixations
    avg_fixation_duration = data[data['Eye movement type']
                                 == 'Fixation']['Gaze event duration'].mean()

    # Compute statistics for pupil diameter, Gaze point X, Gaze point Y, Fixation point X, and Fixation point Y
    pupil_diameter_stats = data[['Pupil diameter left', 'Pupil diameter right']].mean(
        axis=1).agg(['mean', 'median', 'std']).rename(lambda x: f'Pupil Diameter {x.capitalize()}')
    gaze_point_x_stats = data['Gaze point X'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Gaze Point X {x.capitalize()}')
    gaze_point_y_stats = data['Gaze point Y'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Gaze Point Y {x.capitalize()}')
    fixation_point_x_stats = data['Fixation point X'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Fixation Point X {x.capitalize()}')
    fixation_point_y_stats = data['Fixation point Y'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Fixation Point Y {x.capitalize()}')

    # Prepare a summary row containing relevant data
    summarized_data = {
        'Participant': data['Participant name'].iloc[0],
        'Project': group,
        'Recording': data['Recording name'].iloc[0],
        'Total Fixations': total_fixations_count,
        'Avg. Fixation Duration': avg_fixation_duration
    }
    summarized_data.update(pupil_diameter_stats)
    summarized_data.update(gaze_point_x_stats)
    summarized_data.update(gaze_point_y_stats)
    summarized_data.update(fixation_point_x_stats)
    summarized_data.update(fixation_point_y_stats)

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summarized_data, index=[0])

    return summary_df

def visualize_pupil_diameter_uncertainty(data_frame):
    # Extract unique participant names
    unique_participants = data_frame['Participant Name'].unique()

    # Iterate over each unique participant
    for participant in unique_participants:
        # Select data specific to the current participant
        participant_data = data_frame[data_frame['Participant Name']
                                      == participant]

        # Prepare a subset of the data for analysis
        subset_data = participant_data.reset_index().rename(
            columns={'index': 'occurrence'}).head(6)

        # Group the subset data by occurrence
        grouped_data = subset_data.groupby('occurrence').agg({
            'Pupil Diameter Mean': 'mean',
            'Pupil Diameter Median': 'mean',
            'Pupil Diameter Std': 'mean'
        }).reset_index()

        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Plot the mean with error bars and the median
        ax.errorbar(grouped_data['occurrence'], grouped_data['Pupil Diameter Mean'],
                    grouped_data['Pupil Diameter Std'], linestyle='-', marker='o', capsize=5,
                    ecolor="green", elinewidth=0.5, label='Mean')
        ax.plot(grouped_data['occurrence'], grouped_data['Pupil Diameter Median'],
                linestyle='-', marker='s', label='Median')

        # Set labels and title
        ax.set_xlabel('Occurrence')
        ax.set_ylabel('Avg Pupil Diameter (mm)')
        ax.set_title(
            f'Mean and Median Pupil Diameter Uncertainty for {participant}')

        # Add legend and show the plot
        ax.legend()
        plt.show()

def visualize_actual_vs_predicted(dataframe):
    # Extract actual and predicted empathy scores from the dataframe
    actual_scores = dataframe['Original Empathy Score'].tolist()
    predicted_scores = dataframe['Predicted Empathy Score'].tolist()

    # Create a scatter plot of actual vs. predicted empathy scores
    plt.scatter(actual_scores, predicted_scores,
                color='blue', label='Predicted')
    plt.xlabel('Original Empathy Scores')
    plt.ylabel('Predicted Empathy Scores')
    plt.title('Comparison of Original and Predicted Empathy Scores')

    # Add a diagonal line for perfect predictions
    min_score = min(min(actual_scores), min(predicted_scores))
    max_score = max(max(actual_scores), max(predicted_scores))
    plt.plot([min_score, max_score], [min_score, max_score],
             color='red', label='Perfect Prediction')

    # Display legend and the plot
    plt.legend()
    plt.show()

    return

def train_and_evaluate(data_frame, study_group):
    # Prepare the feature matrix X and target vector y
    X = data_frame.drop(
        columns=['Total Score extended', 'Project Name', 'Recording Name'])
    y = data_frame['Total Score extended']

    # Initialize a DataFrame to store evaluation results
    results_dataframe = pd.DataFrame(
        columns=['Participant Name', 'Original Empathy Score', 'Predicted Empathy Score'])

    # Encode the 'Participant Name' column using LabelEncoder
    encoder = LabelEncoder()
    X['Participant Name'] = encoder.fit_transform(X['Participant Name'])
    participant_groups = data_frame['Participant Name']

    # Set the number of splits for GroupKFold
    num_splits = 30  # Number of participants
    group_kfold = GroupKFold(n_splits=num_splits)

    # Initialize lists to store evaluation metrics
    mean_squared_errors = []
    r2_scores = []
    root_mean_squared_errors = []
    median_absolute_errors = []
    all_test_scores = []  # Initialize list to store all test scores
    all_predicted_scores = []  # Initialize list to store all predicted scores

    # Loop over each fold in GroupKFold
    for fold, (train_indices, test_indices) in enumerate(group_kfold.split(X, y, groups=participant_groups)):

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        ###################################
        # Initialize and train the model (RandomForestRegressor)
        model = RandomForestRegressor()

        model.fit(X_train, y_train)

        # Make predictions using the trained model
        y_pred = model.predict(X_test)
        ######################################

        print(f"Fold {fold + 1}:")

        for idx, (original, predicted) in enumerate(zip(y_test, y_pred)):
            participant_name = data_frame.iloc[test_indices[idx]
                                               ]['Participant Name']
            print(
                f"  Participant: {participant_name}, Original Empathy Score: {original}, Predicted Empathy Score: {predicted:.2f}")
            results_dataframe = results_dataframe.append(
                {'Participant Name': participant_name, 'Original Empathy Score': original, 'Predicted Empathy Score': predicted}, ignore_index=True)

        mse = mean_squared_error(y_test, y_pred)
        root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        median_absolute_err = median_absolute_error(y_test, y_pred)
        explained_var_score = explained_variance_score(y_test, y_pred)

        mean_squared_errors.append(mse)
        r2_scores.append(r2)
        root_mean_squared_errors.append(root_mse)
        median_absolute_errors.append(median_absolute_err)
        all_test_scores.extend(y_test)
        all_predicted_scores.extend(y_pred)

    # Calculate average evaluation metrics
    average_r2_score = np.mean(r2_scores)
    average_root_mse = np.mean(root_mean_squared_errors)
    average_median_absolute_err = np.mean(median_absolute_errors)
    average_mean_squared_err = np.mean(mean_squared_errors)

    print(f"Average Root Mean Squared Error: {average_root_mse}")
    print(f"Average Median Absolute Error: {average_median_absolute_err}")
    print(f"Average Mean Squared Error: {average_mean_squared_err}")

    return results_dataframe

def visualize_correlation_heatmap(data_frame, target_column, top_n=15):
    """
    Visualizes a heatmap showing the correlations between features in the DataFrame.

    Args:
        data_frame (pd.DataFrame): The input DataFrame containing features.
        target_column (str): The column used for sorting correlations.
        top_n (int, optional): Number of top correlated features to display. Defaults to 15.
    """
    # Calculate the correlation matrix
    correlation_matrix = data_frame.corr()

    # Select the top_n columns with the highest correlation to the target column
    top_correlated_cols = correlation_matrix.nlargest(top_n, target_column)[
        target_column].index

    # Calculate the correlation matrix for the selected columns
    selected_cols_corr = data_frame[top_correlated_cols].corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 10))
    heatmap_ax = sns.heatmap(selected_cols_corr, annot=True, cmap='coolwarm')

    # Set the title of the heatmap
    heatmap_ax.set_title(f'Correlation Heatmap for {target_column}')

    # Display the heatmap
    plt.show()

    return

def visualize_empathy_scores(df):
    # Calculate the mean of original and predicted empathy scores grouped by participant
    results_test_mean = df.groupby('Participant Name').agg(
        {'Original Empathy Score': 'first', 'Predicted Empathy Score': 'mean'})

    # Set options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Print the grouped DataFrame
    print("Grouped DataFrame:")
    print(results_test_mean)

    # Reshape the data for visualization
    melted_df = results_test_mean.reset_index().melt(id_vars=['Participant Name'], value_vars=[
        'Original Empathy Score', 'Predicted Empathy Score'], var_name='Score Type', value_name='Score')

    # Display the scores for the first 7 participants
    first_7_participants = melted_df['Participant Name'].unique()[:7]
    filtered_df = melted_df[melted_df['Participant Name'].isin(
        first_7_participants)]

    # Create a bar plot for visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(data=filtered_df, x='Participant Name',
                y='Score', hue='Score Type')

    plt.title(
        'Bar Plot of Actual and Predicted Empathy Scores for the First few Participants')
    plt.xlabel('Participant Name')
    plt.ylabel('Empathy Score')

    plt.show()

    return
