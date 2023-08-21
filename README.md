# Project: Predicting Empathy through Eye-Tracking Data

The objective of this Python endeavor is to forecast individuals' empathy levels using eye-tracking data. This project employs machine learning methods, such as **RandomForestRegressor**, in combination with **GroupKFold validation**, to construct a predictive model.

### Getting Started

This guide will provide you with step-by-step instructions to establish the project on your local machine, facilitating development and testing endeavors.

### Prerequisites

For running this project, Python 3.x and the subsequent libraries need to be installed:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pickle

###### You can acquire these libraries by executing the subsequent command:

pip install pandas numpy matplotlib seaborn scikit-learn pickle

### Project Structure

The project is structured in the subsequent manner:

- lib.py: Houses utility functions for data preprocessing, feature extraction, and model evaluation.
- empathy-Dataset-analysis.ipynb: Contains the machine learning pipeline encompassing exploration and illustrative examples.

* Usage
  To initiate the project, execute the empathy.ipynb file. This action will carry out data preprocessing, extract pertinent features, train the RandomForestRegressor model, and assess its performance via cross-validation.

Ensure to incorporate all essential information to facilitate users in comprehending the project and its operational aspects.

### Dataset Acknowledgements

The dataset employed in this investigation comprises the following data. To make use of the dataset, just download it and modify the file paths to align with your configuration.

I extend my appreciation to the creators and collaborators behind the EyeT4Empathy dataset for its open accessibility. You can access the dataset at the provided link below:

[EyeT4Empathy Dataset](https://doi.org/10.1038/s41597-022-01862-w)

Kindly acknowledge the dataset using the subsequent reference:

P. Lencastre, S. Bhurtel, A. Yazidi, S. Denysov, P. G. Lind, et al. EyeT4Empathy: Dataset of foraging for visual information, gaze typing and empathy assessment. Scientific Data, 9(1):1–8, 2022

<pre>
```bibtex
@article{Lencastre2022,
  author = {Lencastre, Pedro and Bhurtel, Sanchita and Yazidi, Anis and et al.},
  title = {EyeT4Empathy: Dataset of foraging for visual information, gaze typing and empathy assessment},
  journal = {Sci Data},
  volume = {9},
  pages = {752},
  year = {2022},
  doi = {10.1038/s41597-022-01862-w}
}
```
</pre>
