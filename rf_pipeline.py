#!/usr/bin/env python3
# coding: utf-8
# Application samplify.py (Segmentation and Classification tool for images of Arabidopsis Seeds)
# Author: Ronja Lea Jennifer Müller, Potsdam 2025

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import logging
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import json
import platform
import getpass
import datetime


"""
    Generates a comprehensive documentation file for the Random Forest training process.
    
    The documentation includes:
    - Date and time of training
    - System and user information
    - Training and testing dataset details
    - Features used in the model
    - Outlier removal status
    - Label mapping applied
    - Scaling status
    - Images (list) used for training
    - Train-test split ratio
    - Model parameters
    - Sklearn and Python version
    - Model testing results (if available)
    - Feature importances
    - Original author and affiliation
"""

global rf_training_doc 

rf_training_doc = {
    "Training Dataset Information": {}
}

def save_rf_documentation(output_dir):
    """Saves the updated documentation dictionary to a JSON file."""
    rf_training_doc["Processing Details"] = {
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "User": getpass.getuser(),
            "System": platform.system(),
            "Machine": platform.node(),
            "OS Version": platform.version(),
            "Python Version": platform.python_version(),
            "Sklearn Version Used": sklearn.__version__,
            "Command to run the script": ' '.join(sys.argv)
        }
    rf_training_doc["Original Author"] = {
        "Script": "rf_pipeline.py",
        "Author": "Ronja Lea Jennifer Müller",
        "Email": "ronja_mueller@gmx.net",
        "Affiliation": "Max Planck Institute for Molecular Plant Physiology",
        "Head of Research": "Prof. Dirk Walther",
        "Year": "2025"
    }
    rf_path = rf_training_doc["Model Information"]["Model Path"]
    rf_name = os.path.basename(rf_path).split(".")[0]
    doc_path = os.path.join(output_dir, f"{rf_name}_documentation.json")

    try:
        with open(doc_path, "w") as f:
            json.dump(rf_training_doc, f, indent=4)
        print(f"Saved documentation in {output_dir}")
    except Exception as e:
            logging.error(f"Error saving documentation to {output_dir}: {e}")
            print("Documentation could not be saved. Check error_log.txt file")


def read_coordinates_from_txt(file_path, x_col='X', y_col='Y', label_col='Counter', delimiter='\t'):
    """
    Read coordinates and labels from a TXT file.
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        if x_col not in df.columns or y_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"File is missing required columns: {x_col}, {y_col}, or {label_col}.")
        return list(zip(df[x_col], df[y_col], df[label_col]))
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []

def map_labels_to_features(features_group, labeled_points):
    """
    Maps labels from labeled points to features based on the closest label to the centroid.
    """
    used_points = set()
    
    for idx, feature in features_group.iterrows():  # Iterate through DataFrame rows
        bbox = feature["Bounding Box"]
        if isinstance(bbox, tuple):  # If bbox is already a tuple, unpack directly
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
        else:
            bbox = bbox.replace("(", "").replace(")", "")
            bbox_x, bbox_y, bbox_w, bbox_h = map(int, bbox.split(","))

        centroid = feature["Centroid"]
        if isinstance(centroid, tuple):  # If centroid is already a tuple, unpack directly
            centroid_x, centroid_y = centroid
        else:
            centroid = centroid.replace("(", "").replace(")","")
            centroid_x, centroid_y = map(int, centroid.split(","))  # Fix split method

        closest_label = None
        min_distance = float('inf')

        for point_idx, (point_x, point_y, label) in enumerate(labeled_points):
            if point_idx in used_points:
                continue

            if bbox_x <= point_x <= bbox_x + bbox_w and bbox_y <= point_y <= bbox_y + bbox_h:
                dist_sq = (centroid_x - point_x)**2 + (centroid_y - point_y)**2
                if dist_sq < min_distance:
                    min_distance = dist_sq
                    closest_label = (label, point_idx)

        if closest_label:
            label, point_idx = closest_label
            features_group.at[idx, "Label"] = ["Normal", "Partially Collapsed", "Fully Collapsed"][label]  
            used_points.add(point_idx)
        else:
            features_group.at[idx, "Label"] = None

    return features_group
    
def labeling_the_data(df, label_dir):
    labeled_data = []
    non_labeled_img = 0
        
    for image_name, group in df.groupby('Image Name'):
        label_path = os.path.join(label_dir, f"{image_name}.txt")
        if not os.path.exists(label_path):
            label_path = os.path.join(label_dir, f"{image_name}_counted.txt") # txt file name should match image name
            
        if os.path.exists(label_path):
            labeled_points = read_coordinates_from_txt(label_path)
            if not labeled_points:
                non_labeled_img += 1
        else:
            error_msg = f"No txt file for {image_name} at {label_path}. Skipping this image."
            logging.error(error_msg)  # Log the error
            non_labeled_img += 1
            continue
                
        updated_features_df = map_labels_to_features(group, labeled_points)  # Ensure this returns a DataFrame
        labeled_data.append(updated_features_df)

    if non_labeled_img >= 1:
        print(f"\033[31mThere was no correct corresponding text file to label {non_labeled_img} images. \nPlease check error_log.txt for exact image names.\033[0m")
    
    return pd.concat(labeled_data, ignore_index=True), non_labeled_img

                                                       
def filter_outliers(df, columns=['Area', 'Perimeter'], group_by_col='Image Name'):
    """ Removes outliers using 3 * IQR filtering per image. Only applied to training set."""

    final_mask = pd.Series(False, index=df.index)
    
    for image_name, group in df.groupby(group_by_col):
        temp_mask = pd.Series(True, index=group.index)
        
        for col in columns:
            Q1, Q3 = group[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 3 * IQR, Q3 + 3 * IQR
            
            temp_mask &= (group[col] >= lower_bound) & (group[col] <= upper_bound)
        
        final_mask.loc[group.index] |= temp_mask  # Keep valid rows
    
    return df[final_mask]

def preprocess_data(data, label_col="Label", drop_cols=None, remove_outliers=False):
    """ Reads dataset, removes outliers (if specified), and drops unnecessary columns. """
    df = data
    outlier_drops = None
    
    if drop_cols is None:
        drop_cols = ['Contour', 'Hull Area', 'Hull Perimeter', 'Perimeter', 'Centroid', 'Bounding Box', 'Area', 'Predicted Label', 'Predicted Probability']
    
    if remove_outliers:
        df1 = filter_outliers(df)
        outlier_drops = len(df) -len(df1)
        df = df1
    df = df.drop(columns=drop_cols, errors='ignore')
    df_processed = df.dropna(subset=[label_col])
    na_drops = len(df) - len(df_processed)
    
    return df_processed, na_drops, outlier_drops

def save_rf_model(rf_model, output_dir):
    """
    Saves the trained Random Forest model with a user-defined name.
    
    Parameters:
    - rf_model: Trained Random Forest model object
    - output_dir (str): Directory to save the model
    
    Returns:
    - model_path (str): Path where the model was saved
    """
    
    while True:
        rf_model_name = input(
            "Please enter a unique name for your trained RF model: ").strip()

        # Ensure the file has the correct .pkl extension
        if not rf_model_name.endswith(".pkl"):
            rf_model_name += ".pkl"
        
        model_path = os.path.join(output_dir, rf_model_name)

        # Check if the file already exists
        if os.path.exists(model_path):
            print(f"\n\033[1;31;40mWarning: A model file named '{rf_model_name}' already exists!\033[0m")
            confirm = input("Do you want to overwrite it? (y/n): ").strip().lower()
            
            if confirm == "n":
                continue
            else:
                print("Overwriting the existing model...")
        
        # Try saving the model
        try:
            joblib.dump(rf_model, model_path)
            return rf_model_name, model_path
        except Exception as e:
            logging.error(f"Error saving model to {model_path}: {e}")
            return None, None
        break

def train_rf(train_file, output_dir, label_dir=None):
    """ Trains a Random Forest classifier and saves the model and feature importance. """

    df = pd.read_csv(train_file)
    data_labeled = "Label" in df.columns
    non_labeled_img = 0

    if not data_labeled:
        try:
            df, non_labeled_img = labeling_the_data(df, label_dir)
        except Exception as e:
            logging.error(f"Labeling for {train_file} could not be performed: {e}")
            print(f"""Your File '{train_file}' does not contain Labels and could not be labeled. 
            \nYou provided the following labeling directory: {label_dir}
            \nPlease check if the Image Names are matching and if the path is correct.
            \nProcess stopped.""")
            exit()
            
    if df.empty:
        logging.error(f"No valid data found for {train_file}. Exiting training.")
        print(f"""No valid data could be read from you {train_file}.
        \nProcess stopped.""")
        exit()
    
    df, na_drops, outlier_drops = preprocess_data(df, remove_outliers=True)

    # Save dataset info
    rf_training_doc["Training Dataset Information"].update({
        "Training Data File": train_file,
        "Train Dataset Labeled already": data_labeled,
        "Number of images impossible to label (excluded)": non_labeled_img,
        "Images Used": list(df["Image Name"].unique()) if "Image Name" in df.columns else [],
        "Outlier Removal": True,
        "Number of Outliers removed": outlier_drops,
        "Number of NaN in 'Label' removed": na_drops,
        "Scaling Applied": False,
    })
    
    X = df.drop(columns=["Label", "Image Name"], errors='ignore')
    y = df["Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    rf_model = RandomForestClassifier(random_state=42, bootstrap=True, oob_score=True,
                                      n_estimators=100, min_samples_split=5, max_depth=7)
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
            
    # Save model
    rf_model_name, model_path = save_rf_model(rf_model, output_dir)

    result_file = os.path.join(output_dir, f"train_results_{os.path.basename(train_file).split(".")[0]}.txt")
    try:
        with open(result_file, "w") as f:
            f.write(f"Training performed by: {getpass.getuser()}\n")
            f.write(f"RF Model Path: {model_path}\n")
            f.write(f"Train File: {train_file}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Report: {report}")
    except Exception as e:
        logging.error(f"Error writing results to {result_file}: {e}")

    importances = rf_model.feature_importances_
    feature_names = rf_model.feature_names_in_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances[indices], align="center",color="black")
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance from Model "{rf_model_name}"')
    plt.savefig(os.path.join(output_dir, "feature_importance.jpg"), bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

    feature_importances = {name: imp for name, imp in zip(X.columns, rf_model.feature_importances_)}

    rf_training_doc["Model Information"] = {
            "Model Path": model_path,
            "Features Used": list(X.columns),
            "Train-Test Split": {"Train": 70, "Test": 30},
            "Random Forest Parameters": {
                "n_estimators": 100,
                "min_samples_split": 5,
                "max_depth": 7,
                "random_state": 42,
                "bootstrap": True,
                "oob_score": True },
        "Feature Importances": feature_importances,
        "Accuracy": accuracy,
        "Classification Report": report
    }
    
    return model_path

def ask_data_details(data_file):
    """
    Asks for user input about data details.
    
    Parameters:
    - data_file (str): The dataset file path.
    
    Returns:
    - test_annotator (str): Name of the annotator.
    - captured_by (str): Name of the person who captured the data.
    - used_standard_protocol (str): Whether the standard protocol was used (yes/no).
    """
    annotator = input(f"\nWho labeled the data in \033[1;36;40m{data_file}?\033[0m ").strip()
    captured_by = input("Who took the image data? ").strip()
    used_standard_protocol = input(
        "Were the images taken under the standard acquisition protocol from Dr. Heinrich Bente? (yes/no): ").strip().lower()

    return annotator, captured_by, used_standard_protocol
    
def test_rf(model_path, test_files, output_dir, label_dir):
    """
    Tests a trained Random Forest model on one or multiple test datasets and updates the documentation.
    
    Parameters:
    - model_path (str): Path to the trained model.
    - test_files (list of str): List of test dataset file paths.
    - output_dir (str): Directory to save test results and documentation.
    - label_dir (str): Directory where the labeling information is stored.
    
    Returns:
    - None
    """
    
    # Load the trained model
    if not os.path.exists(model_path):
        logging.error(f"Error: Model file '{model_path}' not found!")
        return

    rf_model = joblib.load(model_path)
    
    test_results_summary = {}

    # Iterate over all test files
    for test_file in test_files:
        test_file_name = os.path.basename(test_file).split(".")[0]
        df = pd.read_csv(test_file)
        data_labeled = "Label" in df.columns
        non_labeled_img = 0
        
        if not data_labeled:
            try:
                df, non_labeled_img = labeling_the_data(df, label_dir)
            except Exception as e:
                logging.error(f"Labeling for {test_file} could not be performed: {e}")
                print(f"""Your File '{test_file}' does not contain Labels and could not be labeled. 
                \nYou provided the following labeling directory: {label_dir}
                \nPlease check if the Image Names are matching and if the path is correct.
                \nProcess stopped.""")
                save_rf_documentation(args.output_dir)
                exit()
            
        if df.empty:
            logging.error(f"No valid data found for {test_file}. Exiting training.")
            print(f"""No valid data could be read from you {test_file}.
            \nProcess stopped.""")
            save_rf_documentation(args.output_dir)
            exit()
        
        df, na_drops, outlier_drops = preprocess_data(df)

        if test_file == test_files[-1]:
            test_annotator = rf_training_doc["Training Dataset Information"]["Training Data Annotator"]
            captured_by = rf_training_doc["Training Dataset Information"]["Training Data Capture"]
            used_standard_protocol = rf_training_doc["Training Dataset Information"]["Standard Image Acquisition Protocol"] 
        
        else:
            test_annotator = input(f"\nWho labeled the test data in \033[1;36;40m{test_file}?\033[0m ").strip()
            captured_by = input("Who took the testing image data? ").strip()
            used_standard_protocol = input(
                "Were the training images taken under the standard acquisition protocol from Dr. Heinrich Bente? (yes/no): ").strip().lower()
        

        # Prepare test dataset
        X_test = df.drop(columns=["Label", "Image Name"], errors="ignore")
        y_test = df["Label"]

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test,cmap="Blues")
        plt.title(f"Confusion Matrix {test_file_name}")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{test_file_name}.jpg"), bbox_inches='tight', pad_inches=0.1, dpi=300)

        # Save results
        test_results_summary[test_file] = {
            "Test Dataset Labeled already": data_labeled,
            "Test Annotator": test_annotator,
            "Test Data Capture": captured_by,
            "Standard Image Acquisition Protocol": used_standard_protocol,
            "Number of images impossible to label (excluded)": non_labeled_img,
            "Images Used": list(df["Image Name"].unique()) if "Image Name" in df.columns else [],
            "Number of NaN in 'Label' removed": na_drops,
            "Test Accuracy": accuracy,
            "Classification Report": report
        }

        # Save results to a file
        result_file = os.path.join(output_dir, f"test_results_{test_file_name}.txt")
        try:
            with open(result_file, "w") as f:
                f.write(f"Testing performed by: {getpass.getuser()}\n")
                f.write(f"RF Model Path: {model_path}\n")
                f.write(f"Test File: {test_file}\n")
                f.write(f"Annotator: {test_annotator}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write("Classification Report:\n")
                f.write(report)
            print(f"Test results saved to {result_file}")
        except Exception as e:
            logging.error(f"Error writing results to {result_file}: {e}")

    # Update documentation with test results
    rf_training_doc["Testing Information"] = test_results_summary
    
def gather_test_files(test_dir):
    """ Retrieves all CSV files from the specified directory for testing. """
    return [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

def setup_logging(output_dir):
    """Setup logging configuration."""
    filename = os.path.join(output_dir, "error_log.txt")
    logging.basicConfig(
        filename=filename,  # Log file location
        level=logging.ERROR,  # Set logging level to ERROR
        format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log entries
        filemode='w'  # Overwrite the log file each time the script runs
    )
    return filename

def check_file_exists(file_path):
    """Checks if a file exists at the given path."""
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

def check_directory_exists(directory_path):
    """Checks if a directory exists at the given path."""
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        sys.exit(1)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a Random Forest classifier with mapped labels.")
    parser.add_argument("train_file", help="Path to the training CSV dataset")
    parser.add_argument("--label_dir", help="Directory containing label TXT files for all train and test images (Check README to obtain the TXT files correctly)")
    parser.add_argument("--test_dir", help="Directory containing test datasets (optional)")
    parser.add_argument("--output_dir", default="RF_training_output", help="Output directory for model and results")
    args = parser.parse_args()

    # Validate paths
    check_file_exists(args.train_file)  # Check if training file exists
    if args.label_dir:
        check_directory_exists(args.label_dir)  # Check if label directory exists
    if args.test_dir:
        check_directory_exists(args.test_dir)  # Check if test directory exists
        
    test_files = []
    if args.test_dir:
        test_files = gather_test_files(args.test_dir)

    annotator, captured_by, used_standard_protocol = ask_data_details(args.train_file)
    rf_training_doc = {"Training Dataset Information": 
                       {"Training Data Annotator": annotator, "Training Data Capture": captured_by, "Standard Image Acquisition Protocol": used_standard_protocol}}
    
    print("\n--- Input Summary ---")
    print(f"Training file: {args.train_file}, annotated by {annotator}, images captured by {captured_by}")
    print(f"Label files directory: {args.label_dir}")
    if args.test_dir:
        print(f"Testing directory: {args.test_dir} (found {len(test_files)} files)")
    else:
        print("No test directory provided. Only training will be performed.")
    print(f"Output directory: {args.output_dir}")
    print(f"Model will be saved at: {args.output_dir}")

    confirm = input("\nProceed with training and testing? (y/n): ")
    if confirm.lower() != 'y':
        print("Process aborted.")
        exit()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = setup_logging(args.output_dir)
    rf_training_doc["Processing Details"] = {"Error Log File Path": log_file}
    
    model_path = train_rf(args.train_file, args.output_dir, args.label_dir)

    print(f"\nTraining complete. Results saved in: {args.output_dir} \n")
    
    if test_files:
        test_files.append(args.train_file)  # Also test on training set
        test_rf(model_path, test_files, args.output_dir, args.label_dir)
        print(f"\nTesting complete. Results saved in: {args.output_dir} \n")
    else:
        print("\nNo test data provided. Training only.")

    save_rf_documentation(args.output_dir)
        
        
