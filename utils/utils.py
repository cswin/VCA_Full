"""
Utility functions for VCA model training and testing.
Extracted and organized from the main utils.py file.
"""

import torch
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from models.VCA import VCA
from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def initialize_model(args):
    """
    Initialize the VCA model.
    
    Args:
        args: Argument parser object with model configuration
        
    Returns:
        model: Initialized model
        is_gist: Boolean flag for gist features
        is_saliency: Boolean flag for saliency features
    """
    # Validate if args has is_gist and is_saliency parameters
    if not hasattr(args, 'is_gist'):
        args.is_gist = False
    if not hasattr(args, 'is_saliency'):
        args.is_saliency = False

    is_gist = args.is_gist
    is_saliency = args.is_saliency

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_class = getattr(args, 'num_classes', None)

    if args.model_to_run == 71:
        print('model 71 to run: VCA ')
        model = VCA(
            args.clip_model_name,
            is_predict_two=args.is_predict_two,
            num_class=num_class,
            device=device,
        )
    else:
        raise ValueError(f"Model {args.model_to_run} is not supported. Use 71 (VCA).")

    return model, is_gist, is_saliency


def save_checkpoint(best_R, best_loss, best_epoch, model, best_state_dict, optimizer, model_name):
    """
    Save model checkpoint.
    
    Args:
        best_R: Best Pearson correlation
        best_loss: Best loss value
        best_epoch: Best epoch number
        model: Model object
        best_state_dict: Best model state dictionary
        optimizer: Optimizer object
        model_name: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': best_epoch,
        'best_per': best_R,
        'best_loss': best_loss,
        'state_dict': best_state_dict,
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_name)


def save_checkpoint_V2(best_R, best_R2, best_rmse, best_epoch, best_state_dict, optimizer, model_dir, model_name):
    """
    Save model checkpoint (Version 2 with R2 and RMSE).
    
    Args:
        best_R: Best Pearson correlation or list of correlations
        best_R2: Best R2 score or list of R2 scores
        best_rmse: Best RMSE or list of RMSEs
        best_epoch: Best epoch number
        best_state_dict: Best model state dictionary
        optimizer: Optimizer object
        model_dir: Directory to save the checkpoint
        model_name: Name of the checkpoint file
    """
    checkpoint = {
        'epoch': best_epoch,
        'best_R': best_R,
        'best_R2': best_R2,
        'best_rmse': best_rmse,
        'state_dict': best_state_dict,
        # 'optimizer': optimizer.state_dict()
    }

    # Determine file path
    file_path = os.path.join(model_dir, model_name)
    # Save checkpoint
    torch.save(checkpoint, file_path)


def perform_quantile_analysis(actual, predicted, n_quantiles=25):
    """
    Performs quantile analysis on the given actual and predicted values.

    Parameters:
    - actual: Array-like, the actual values.
    - predicted: Array-like, the predicted values.
    - n_quantiles: int, the number of quantiles to split the data into.

    Returns:
    - A dictionary containing the Pearson correlation, R² score, and RMSE for the quantile averages.
    """
    # Create a DataFrame from the actual and predicted values
    data = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted
    })

    # Split the data into quantiles based on 'Actual' values
    data['Quantile'] = pd.qcut(data['Actual'], q=n_quantiles, labels=False)

    # Calculate the average of 'Actual' and 'Predicted' for each quantile
    quantile_averages = data.groupby('Quantile').mean()

    # Calculate the metrics for the quantile averages
    pearson_corr = pearsonr(quantile_averages['Actual'], quantile_averages['Predicted'])[0]
    r2 = r2_score(quantile_averages['Actual'], quantile_averages['Predicted'])
    rmse = np.sqrt(mean_squared_error(quantile_averages['Actual'], quantile_averages['Predicted']))

    # Return the results in a dictionary
    return {
        'Pearson Correlation': pearson_corr,
        'R² Score': r2,
        'RMSE': rmse
    }

