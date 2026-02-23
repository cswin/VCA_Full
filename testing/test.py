import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import RegressionDataset, reg_dataloader_testdata
from utils.utils import initialize_model, perform_quantile_analysis
import matplotlib.pyplot as plt

# Initialize and parse command-line arguments
import argparse

# Get the directory of this script and VCA folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VCA_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODEL_PATH = os.path.join(VCA_DIR, 'trained_models', 'KNVCombined_M71_ViTL4_Amy_TransformerV4_lr1e4_bs64_ep50_May152024_PredictBoth_repeat5.pth')

parser = argparse.ArgumentParser(description='Test a trained model on a dataset')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the trained model')
parser.add_argument('--csv_path', type=str, default='', help='Path to the test dataset CSV')
parser.add_argument('--img_dir', type=str, help='Directory with test images')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
parser.add_argument('--image_size', type=int, default=224, help='Image size (assuming square images)')

parser.add_argument('--model_to_run', default=71, type=int,
                    help='71: VCA (VCA)')

parser.add_argument('--is_gist', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply gist as input to low-road instead')

parser.add_argument('--is_saliency', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply saliency as input to low-road instead')

parser.add_argument('--isarousal', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use arousal as the label instead of valence')

#result folder
parser.add_argument('--resultfolder', default='results', type=str, help='the folder to save the results')

# is_predict_two
parser.add_argument('--is_predict_two', default=True,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to predict two labels at the same time')

#CLIP model name
parser.add_argument('--clip_model_name', default='ViT-L/14', type=str, help='the name of the CLIP model')


#dropout rate
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for the model')

args = parser.parse_args()



def plot_scatter_plot(actual, predicted,  args, variable_name='', data_type='test'):

    plt.figure(figsize=(6, 6))

    plt.scatter(actual, predicted, color='blue', label='Predicted Values', alpha=0.5)
    y = actual
    f = predicted

    # Calculate coefficients for the fitted line
    coef = np.polyfit(f, y, 1)
    poly1d_fn = np.poly1d(coef)

    # Plot the fitted line
    plt.plot(f, poly1d_fn(f), color='red', label='Fitted Line')


    r, p = pearsonr(y, f)
    print('Pearson Correlation:', r)
    # put this r value on the plot
    plt.text(1, 8.5, 'Pearson Correlation r = %0.4f' % r, fontsize=12)

    # Calculate R-Squared btw line and actual values
    r2_line = r2_score(y, poly1d_fn(f))
    print('R-Squared (line vs actual):', r2_line)
    # put this r2_line value on the plot
    plt.text(1, 8, 'R-Squared = %0.4f' % r2_line, fontsize=12)

    # Calculate RMSE btw line and actual values
    rmse = np.sqrt(mean_squared_error(y, poly1d_fn(f)))
    print('RMSE (line vs actual):', rmse)
    # put this rmse value on the plot
    plt.text(1, 7.5, 'RMSE = %0.4f' % rmse, fontsize=12)

    # set x and y axis limits
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    # range of x and y axis
    plt.xticks(np.arange(0, 9, 1))
    plt.yticks(np.arange(0, 9, 1))

    if variable_name == '':
        plt.ylabel('Actual Value')
        plt.xlabel('Predicted Value')
    else:
        plt.ylabel(f'Actual {variable_name}')
        plt.xlabel(f'Predicted {variable_name}')

    plt.grid(True)

    figure_save_path = os.path.join(args.resultfolder, f'{data_type}_{variable_name}_scatter_plot.png')
    plt.savefig(figure_save_path)  # Save plot to file
    plt.close()  # Close the plot to free memory

    # Save fold actual predicted values into a csv file
    fold_actual_predicted = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    fold_actual_predicted.to_csv(os.path.join(args.resultfolder, f'{data_type}_{variable_name}_actual_predicted.csv'), index=False)



# Function to load the model
def load_model(model_path, model, device):
    # Load the checkpoint from the file (robust across PyTorch 2.6+ defaults)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only arg
        checkpoint = torch.load(model_path, map_location=device)
    except Exception:
        # Fallback: allowlist numpy scalar used by some older checkpoints
        try:
            from torch.serialization import add_safe_globals
            import numpy as np
            add_safe_globals([np.core.multiarray.scalar])
        except Exception:
            pass
        checkpoint = torch.load(model_path, map_location=device)

    # Extract the state_dict from the checkpoint.
    # This assumes your checkpoint is a dictionary with a 'state_dict' key.
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # If there's no 'state_dict' key, assume the whole checkpoint is the model state dict.
        state_dict = checkpoint

    # Load the state dict into the model
    model.load_state_dict(state_dict)

    # Move the model to the specified device
    model = model.to(device)

    return model


# Function to prepare test data
def prepare_test_data(csv_path, img_dir, batch_size, is_gist, is_saliency, isarousal, image_size, is_predict_two):
    dataset = RegressionDataset(csv_file=csv_path, root_dir=img_dir, transform=None, is_gist=is_gist,
                                is_saliency=is_saliency, isarousal=isarousal, is_predict_two=is_predict_two)
    dataloader, dataset_size = reg_dataloader_testdata(dataset, batch_size, is_gist, is_saliency, isarousal, image_size)
    return dataloader, dataset_size


# Function to test the model
def test_model(dataloader, model, criterion, device, is_predict_two=False):
    model.eval()  # Set model to evaluation mode
    all_targets = []
    all_predictions = []

    labels_list = []
    preds_list = []
    labels_list1, preds_list1 = [], []
    labels_list2, preds_list2 = [], []

    with torch.no_grad():
        for _, inputs, labels in dataloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if is_predict_two:
                # Assuming the model returns outputs in the format [output1, output2]
                outputs1, outputs2 = model(inputs)  # Assume model returns two outputs
                labels_list1.extend(labels[:, 0].cpu().numpy())
                preds_list1.extend(outputs1.detach().cpu().numpy())
                labels_list2.extend(labels[:, 1].cpu().numpy())
                preds_list2.extend(outputs2.detach().cpu().numpy())
            else:
                outputs = model(inputs)
                labels_list.extend(labels.cpu().numpy())
                preds = outputs.reshape(labels.shape).cpu().detach().numpy()
                preds_list.extend(preds)

    if is_predict_two:
        metrics = {}
        # Calculate Pearson r for both variables
        metrics['r_Arousal'], _ = pearsonr(labels_list1, preds_list1)
        metrics['r_Valence'], _ = pearsonr(labels_list2, preds_list2)

        # Compute fitted-line R² and RMSE for Arousal
        coef1 = np.polyfit(preds_list1, labels_list1, 1)
        poly1d_fn1 = np.poly1d(coef1)
        fitted1 = poly1d_fn1(preds_list1)
        metrics['R2_Arousal'] = r2_score(labels_list1, fitted1)
        metrics['RMSE_Arousal'] = np.sqrt(mean_squared_error(labels_list1, fitted1))

        # Compute fitted-line R² and RMSE for Valence
        coef2 = np.polyfit(preds_list2, labels_list2, 1)
        poly1d_fn2 = np.poly1d(coef2)
        fitted2 = poly1d_fn2(preds_list2)
        metrics['R2_Valence'] = r2_score(labels_list2, fitted2)
        metrics['RMSE_Valence'] = np.sqrt(mean_squared_error(labels_list2, fitted2))


        # Save the   predictions and targets for both variables
        all_predictions = [preds_list1, preds_list2]
        all_targets = [labels_list1, labels_list2]

        best_R = [metrics['r_Arousal'], metrics['r_Valence']]
        best_R2 = [metrics['R2_Arousal'], metrics['R2_Valence']]
        best_RMSE = [metrics['RMSE_Arousal'], metrics['RMSE_Valence']]

        # Convert lists to numpy arrays for processing
        all_targets = [np.array(targets) for targets in all_targets]
        all_predictions = [np.array(predictions) for predictions in all_predictions]

    else:
        # Pearson r
        best_R, _ = pearsonr(labels_list, preds_list)

        # Fitted-line R² and RMSE
        coef = np.polyfit(preds_list, labels_list, 1)
        poly1d_fn = np.poly1d(coef)
        fitted = poly1d_fn(preds_list)
        best_R2 = r2_score(labels_list, fitted)
        best_RMSE = np.sqrt(mean_squared_error(labels_list, fitted))

        all_targets = labels_list

        # After the loop, convert lists to numpy arrays for processing
        all_targets = np.array(all_targets)
        all_predictions = np.array(preds_list)

        # Return targets and predictions along with other metrics
    return best_R, best_R2, best_RMSE, all_targets, all_predictions


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #if the result folder does not exist, create it
    if not os.path.exists(args.resultfolder):
        os.makedirs(args.resultfolder)

    # Initialize model
    print('Initializing model')
    model, _, _ = initialize_model(args)
    # Load model
    print('Loading model')
    model = load_model(args.model_path, model, device)
    criterion = torch.nn.MSELoss()  # Assuming MSE loss for regression problems

    # Prepare test data
    print('Preparing test data')
    test_dataloader, _ = prepare_test_data(args.csv_path, args.img_dir, args.batch_size, args.is_gist, args.is_saliency,
                                           args.isarousal, args.image_size, args.is_predict_two)

    # Test model
    test_performances = []
    test_performances_arousal = []
    test_performances_valence = []

    all_test_targets = []
    all_test_predictions = []

    all_test_targets_arousal = []
    all_test_targets_valence = []
    all_test_predictions_arousal = []
    all_test_predictions_valence = []


    print('Testing model')
    test_R, test_R2, test_RMSE, test_targets, test_predictions  = test_model(
        test_dataloader, model, criterion, device,
        args.is_predict_two)

    # Save the results
    print('Saving results')
    if args.is_predict_two:
        print('-' * 10)
        print(f" Test - r_Arousal: {test_R[0]:.4f},"
              f" R2_Arousal: {test_R2[0]:.4f}, RMSE_Arousal: {test_RMSE[0]:.4f},"
              f" r_Valence: {test_R[1]:.4f}, R2_Valence: {test_R2[1]:.4f}, RMSE_Valence: {test_RMSE[1]:.4f}")

        test_performances_arousal.append({ 'r': test_R[0], 'R2': test_R2[0],
                                          'RMSE': test_RMSE[0]})
        test_performances_valence.append({ 'r': test_R[1], 'R2': test_R2[1],
                                          'RMSE': test_RMSE[1]})

        # save the scatter plot
        plot_scatter_plot(test_targets[0], test_predictions[0], args, variable_name='Arousal', data_type='test')
        plot_scatter_plot(test_targets[1], test_predictions[1],  args, variable_name='Valence', data_type='test')

        #quantile analysis
        quantile_analysis_25_arousal = perform_quantile_analysis(test_targets[0], test_predictions[0])
        print('Arousal- Quantile analysis for 25th percentile:', quantile_analysis_25_arousal)
        quantile_analysis_25_valence = perform_quantile_analysis(test_targets[1], test_predictions[1])
        print('Valence- Quantile analysis for 25th percentile:', quantile_analysis_25_valence)

    else:
        print('-' * 10)
        print(f"Test - r: {test_R:.4f},"
              f"R2: {test_R2:.4f}, RMSE: {test_RMSE:.4f}")

        test_performances.append({ 'r': test_R, 'R2': test_R2, 'RMSE': test_RMSE})

        variable_name = 'Valence' if not args.isarousal else 'Arousal'
        # save the scatter plot
        plot_scatter_plot(test_targets, test_predictions, args, variable_name=variable_name, data_type='test')

        #quantile analysis
        quantile_analysis_25 = perform_quantile_analysis(test_targets, test_predictions)
        print('Quantile analysis for 25th percentile:', quantile_analysis_25)
