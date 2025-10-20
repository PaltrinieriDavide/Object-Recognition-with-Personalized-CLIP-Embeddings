# Object Recognition with Personalized CLIP Embeddings

This project implements a binary classifier for specific object recognition, addressing the limitations of standard CLIP in preserving an object's unique identity. The core methodology leverages HiPer to learn an object's distinctive features from a target image, encoding this identity into specialized textual embeddings that share CLIP's vector space. This process transforms the classification challenge into a tabular data problem by creating a feature set based on the similarity between an image and the specialized HiPer prompt.
A final classical machine learning model is then trained on these engineered features. The proposed solution demonstrates high reliability and efficiency, achieving an average accuracy of over 87.5%. This result significantly outperforms standalone CLIP, which averaged only 57.5% accuracy on the same multi-prompt classification tasks, proving the effectiveness of this combined approach.

## Prerequisites and Setup

### 1. Installation and Environment
First, clone the repository, create and activate the environment and install all the required dependecies. They are listed in requirements.txt

### 2. Dataset Preparation
The scripts require a specific directory structure for the image dataset. The root dataset folder must be split into `train` and `test` subdirectories. Inside each, create folders named `0` for the negative class (images not containing the target object) and `1` for the positive class (images containing the target object). This structure is crucial for the `dataset_generator.py` script to correctly assign labels.

**Required Folder Structure:**
```
/path/to/dataset/
├── train/
│   ├── 0/
│   │   ├── image_001.png
│   │   └── ...
│   └── 1/
│       ├── target_image_001.png
│       └── ...
└── test/
    ├── 0/
    │   ├── image_101.png
    │   └── ...
    └── 1/
        ├── target_image_101.png
        └── ...
```

## How to Run the Pipeline

The end-to-end workflow consists of two main stages, executed by separate scripts.

### Step 1: Generate Feature Datasets
This initial step processes the raw images and constructs the tabular datasets (`.csv` files) that will be used for training the machine learning models. The script performs three key operations:
1.  Generates standard CLIP embeddings for every image in the training and testing sets.
2.  Fine-tunes a specialized set of HiPer text embeddings using a few sample images from the positive training class (`train/1/`).
3.  Computes a similarity vector for each image by comparing its CLIP embedding to the fine-tuned HiPer prompt, creating the final feature set.

**Execution:**
Before running, you can configure key parameters within the `main()` function of `dataset_generator.py`:
-   `text`: The descriptive text prompt for the target object (e.g., `"A photo of a blue backpack"`). This is the base for HiPer fine-tuning.
-   `num_images`: The number of positive examples to use for generating the HiPer embeddings.
-   `train_steps`: The number of optimization steps for the HiPer training process.

Run the script from the command line, providing paths to your train and test directories:```bash
python dataset_generator.py /path/to/dataset/train /path/to/dataset/test
```
This process will create `similarities_datset_train.csv` and `similarities_datset_test.csv` in the `data/splitted/` directory.

### Step 2: Train and Evaluate the Classifier
With the feature datasets generated, this second step automates the process of selecting, training, and evaluating the best-performing machine learning model. The script uses `GridSearchCV` to perform an exhaustive search over a predefined set of models (e.g., Logistic Regression, SVM, Random Forest) and their hyperparameters, using stratified cross-validation to ensure robust evaluation.

**Execution:**
Run the `final_train.py` script, passing the paths to the CSV files generated in the previous step:
```bash
python final_train.py data/splitted/similarities_datset_train.csv data/splitted/similarities_datset_test.csv
```

## Interpreting the Output

The `final_train.py` script generates a `results/` directory containing a comprehensive summary of the experiment.

-   **`results/data.log`**: This log file is the primary text output. It contains detailed metrics from the cross-validation phase for every model tested, the optimal hyperparameters found for the best model, and a final classification report and confusion matrix from the evaluation on the held-out test set.

-   **`results/final_model/`**: This folder contains the final, trained model object, serialized as a `.pkl` file (e.g., `SVC.pkl`). This file can be loaded in other applications for inference on new data without needing to retrain.

-   **`results/graphs/`**: This directory provides key performance visualizations that are crucial for interpreting the model's behavior on the test set:
    -   `confusion_matrix.png`: Visualizes true vs. predicted labels.
    -   `roc_curve.png`: Shows the trade-off between true positive rate and false positive rate.
    -   `precision_recall_curve.png`: Illustrates the trade-off between precision and recall.
    -   `classification_report.txt`: A text file containing precision, recall, and F1-score for each class.
