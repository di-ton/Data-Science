import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Data preparation and manipulation script"""

def plot_binary_feature(dataset, feature):

    """ Function plotting binary features with possible states of 'Yes' and 'No' """
    
    feature_mapping = {0: 'No', 1: 'Yes'}
    feature_counts = dataset[f"{feature}"].map(feature_mapping).value_counts()

    feature_counts.plot(kind = 'bar')
    plt.xlabel(f"{feature}")
    plt.ylabel('Count')
    plt.title(f"Distribution of {feature}")
    plt.xticks(rotation = 0)
    plt.show()

def plot_categorical_feature(dataset, feature, mapping_categories):

    """ 
    Function plotting categorical features with different number of categories.
    The parameter mapping_categories represents a list of the categories.
    """
    
    feature_mapping = {}
    for n in range(len(mapping_categories)):
        feature_mapping[n] = mapping_categories[n]

    feature_counts = dataset[f"{feature}"].map(feature_mapping).value_counts()

    feature_counts.plot(kind = 'bar')
    plt.xlabel(f"{feature}")
    plt.ylabel('Count')
    plt.title(f"Distribution of {feature}")
    plt.xticks(rotation = 0)
    plt.show()

def plot_histogram_feature(dataset, feature, bins = 'fd'):

    """ 
    Function plotting continous features.
    """
    
    plt.hist(dataset[f"{feature}"], bins = bins)
    plt.xlabel(f"{feature}")
    plt.ylabel('Count')
    plt.title(f"Distribution of {feature}")
    plt.xticks(rotation = 0)
    plt.show()

