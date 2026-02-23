from __future__ import print_function, division
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from warmup_scheduler import GradualWarmupScheduler
import matplotlib
matplotlib.use('Agg')
import time

import copy
from utils.utils import save_checkpoint, initialize_model, save_checkpoint_V2
from utils.dataloader import reg_dataloader_KFold, RegressionDataset, reg_dataloader_testdata
import argparse
# from logger import Logger
from scipy.stats import pearsonr
import numpy as np
from torchsummary import summary
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ===== Custom LR schedules (epoch-wise targets) =====
def _get_step_lr(epoch: int, base_lr: float, step_size: int, gamma: float = 0.1) -> float:
    num_steps = epoch // step_size
    return (gamma ** num_steps) * base_lr


def _get_cyclic_lr(epoch: int, base_lr: float, total_epochs: int, lr_peak_epoch: int) -> float:
    xs = [0, lr_peak_epoch, total_epochs]
    ys = [1e-4 * base_lr, base_lr, 0.0]
    return float(np.interp([epoch], xs, ys)[0])


def _get_cyclic_lr_with_plateau(epoch: int, base_lr: float, lr_min: float,
                                lr_peak_epoch: int, lr_cycle_end_epoch: int,
                                total_epochs: int) -> float:
    xs = [0, lr_peak_epoch, lr_cycle_end_epoch, total_epochs]
    ys = [1e-4 * base_lr, base_lr, lr_min, lr_min]
    return float(np.interp([epoch], xs, ys)[0])

#
# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='/DATA/pliu/EmotionPerceptionDNN_data/', type=str, help='the data root folder')

#training data
parser.add_argument('--train_data', default='KNV_combine', type=str, help='the folder of training data')
parser.add_argument('--train_csv_data', default='KNV_combine.csv', type=str, help='the path of training data csv file')

#add test data
parser.add_argument('--test_data', default='IAPS', type=str, help='the folder of test data')
parser.add_argument('--test_csv_data', default='IAPS.csv', type=str, help='the path of test data csv file')

# image size
parser.add_argument('--image_size', default=224, type=int, help='input image size')

parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=2, type=int, help='number of train epoches')
#smaller lr is important or the output will be nan
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for SGD')
parser.add_argument('--step_size', default=40, type=int, help='learning rate decay step size')
parser.add_argument('--lr_schedule_type', default='step', type=str,
                    choices=['step', 'cyclic', 'cyclic_plateau'],
                    help='learning rate schedule strategy')
parser.add_argument('--lr_min', default=2e-5, type=float, help='minimum lr for cyclic schedules')
parser.add_argument('--lr_peak_epoch', default=2, type=int, help='epoch at which LR peaks for cyclic schedules')
parser.add_argument('--lr_cycle_end_epoch', default=88, type=int,
                    help='epoch at which LR cycle ends (before plateau) for cyclic_plateau')

parser.add_argument('--early_stop_patience', default=0, type=int,
                    help='epochs to wait for val improvement before stopping (0 to disable)')


parser.add_argument('--model_dir', default='/DATA/pliu/EmotionPerceptionDNN_model', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='test.pth',
                    type=str, help='name of the trained model')
parser.add_argument('--logfolder', default='test',
                    type=str, help='name of the log')
#add the folder for result
parser.add_argument('--resultfolder', default='/DATA/pliu/EmotionPerceptionDNN_result/',  type=str, help='name of the log')
#add validation_performances.csv
parser.add_argument('--validation_performances', default='test.csv', type=str, help='name of the log')


parser.add_argument('--is_log', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to record log')

parser.add_argument('--resume', default=None, type=str, help='the path to checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--model_to_run', default=71, type=int,
                    help='71: VCA (VCA)')

parser.add_argument('--n_layers', default=1, type=int,
                    help='the number of layers in Amygdala; this setting is only for model_to_run of 1')
# num of neurons in the fc layer of amygdala
parser.add_argument('--n_neurons_fist_layer', default=1000, type=int,
                    help='the number of neurons in the first fc layer of amygdala; this setting is only for model_to_run of 1')
#n_hidden
parser.add_argument('--n_hidden', default=1000, type=int,
                    help='the number of neurons in the hidden layer; this setting is only for model_to_run of 1')

parser.add_argument('--lowfea_VGGlayer', default=3, type=int,
                    help='which layer of vgg to extract features as the low-road input. '
                         'This setting is only available for model_to_run 2 and 4')
parser.add_argument('--highfea_VGGlayer', default=36, type=int,
                    help='which layer of vgg to extract features as the high-road input to amygdala. '
                         'This setting is only available for model_to_run 2 and 4')

parser.add_argument('--is_gist', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply gist as input to low-road instead')

parser.add_argument('--is_saliency', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply saliency as input to low-road instead')

parser.add_argument('--isarousal', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use arousal as the label instead of valence')

parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

# is_predict_two
parser.add_argument('--is_predict_two', default=True,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to predict two labels at the same time')

#CLIP model name
parser.add_argument('--clip_model_name', default='ViT-L/14', type=str, help='the name of the CLIP model')

parser.add_argument('--pretrained_data_name', default='', type=str, help='the name of the pretrained data for OpenCLIP')

#split data into training and testing with ratio: 9:1
parser.add_argument('--split_ratio', default=0.9, type=float, help='the ratio of training data')

#repeat times
parser.add_argument('--repeat', default=1, type=int, help='the number of repeat times')

# whether save model after each epoch
parser.add_argument('--is_saveModel_epoch', default=False,type=lambda x: (str(x).lower() == 'true'),
                    help='whether to predict two labels at the same time')

#dropout rate
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for the model')

# Define the transformation type you want to test  transformation_type = 'color'  # Options: 'color', 'noise', 'lighting', '
parser.add_argument('--transformation_type', default=None, type=str, help='the type of transformation to apply to the test images')
#noise variance
parser.add_argument('--variance', default=0, type=int, help='the variance of the noise to apply to the test images')

def plot_scatter_plot(actual, predicted, repeat, args, variable_name='', data_type='val'):
    from scipy.stats import pearsonr
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

    plt.title(f'Repeat {repeat + 1}')
    plt.grid(True)

    figure_save_path = os.path.join(args.resultfolder, args.logfolder, f'{data_type}_{variable_name}_scatter_plot_repeat_{repeat + 1}.png')
    plt.savefig(figure_save_path)  # Save plot to file
    plt.close()  # Close the plot to free memory

    # Save fold actual predicted values into a csv file
    fold_actual_predicted = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    fold_actual_predicted.to_csv(os.path.join(args.resultfolder, args.logfolder, f'{data_type}_{variable_name}_repeat_{repeat + 1}_actual_predicted.csv'), index=False)


def train_model(dataloaders, dataset_sizes, TRAIN, VAL, model, criterion, optimizer, scheduler,
                num_epochs, device, model_dir, model_name,
                start_epoch=0, is_gist=False,
                is_saliency=False, is_predict_two=False, is_saveModel_epoch=False,
                args=None):


    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_R = 0.0
    best_R2 = 0.0
    best_loss = 10
    best_rmse = 10
    best_avg_rmse = 10
    best_avg_r2 = -1.0
    no_improve_epochs = 0
    patience = getattr(args, 'early_stop_patience', 0)
    best_avg_R = 0.0

    if is_saveModel_epoch:
        # save the model before training starts (epoch 0)
        epoch0_model_name = model_name[:-4] + '_epoch0.pth'
        # Save the model after each epoch
        save_checkpoint_V2(best_R=best_R,
                           best_R2=best_R2,
                           best_rmse=best_rmse,
                           best_epoch=0,
                           best_state_dict=model.state_dict(),
                           optimizer=optimizer,
                           model_dir=model_dir,
                           model_name=epoch0_model_name)
        print(f'Saved model before training starts as {epoch0_model_name}')

    if num_epochs == 0:
        print('Training complete in 0m 0s')
        return best_model_wts, best_loss, best_R, best_R2, best_rmse, 0, [], []
    # Initialize lists to store targets and predictions for quantile metrics calculation
    all_targets = []
    best_predictions = []
    # Per-epoch training/validation history
    history_rows = []

    # Initialize best metrics for both variables
    best_metrics = {'R1': 0.0, 'R2_1': 0.0, 'RMSE1': float('inf'),
                    'R2': 0.0, 'R2_2': 0.0, 'RMSE2': float('inf')}
    # Define criteria for two-head regression when predicting both
    arousal_criterion = nn.SmoothL1Loss(beta=0.5)
    valence_criterion = nn.MSELoss()
    # Loss weights schedule to avoid hurting valence:
    # early: emphasize arousal (0.7/0.3), middle: soften (0.6/0.4), late: equal (0.5/0.5)
    def get_loss_weights(current_epoch: int, total_epochs: int):
        progress = (current_epoch - start_epoch) / max(1, total_epochs)
        if progress < 0.3:
            aw = 0.7
        elif progress < 0.7:
            aw = 0.6
        else:
            aw = 0.5
        return aw, 1.0 - aw

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Track whether validation improved this epoch for early stopping
        val_improved_this_epoch = False

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            labels_list = []
            preds_list = []
            running_loss = 0.0
            # Initialize lists to store targets and predictions for both outputs
            labels_list1, preds_list1 = [], []
            labels_list2, preds_list2 = [], []
            # Prepare per-iteration LR schedule if training and using cyclic strategies
            if phase == TRAIN and getattr(args, 'lr_schedule_type', 'step') != 'step':
                total_epochs = num_epochs
                # Target LR at current and next epoch boundaries
                if args.lr_schedule_type == 'cyclic':
                    lr_start = _get_cyclic_lr(epoch, args.lr, total_epochs, args.lr_peak_epoch)
                    lr_end = _get_cyclic_lr(epoch + 1, args.lr, total_epochs, args.lr_peak_epoch)
                elif args.lr_schedule_type == 'cyclic_plateau':
                    lr_start = _get_cyclic_lr_with_plateau(epoch, args.lr, args.lr_min, args.lr_peak_epoch,
                                                           args.lr_cycle_end_epoch, total_epochs)
                    lr_end = _get_cyclic_lr_with_plateau(epoch + 1, args.lr, args.lr_min, args.lr_peak_epoch,
                                                         args.lr_cycle_end_epoch, total_epochs)
                else:
                    lr_start = _get_step_lr(epoch, args.lr, args.step_size)
                    lr_end = _get_step_lr(epoch + 1, args.lr, args.step_size)

                iters = max(1, len(dataloaders[TRAIN]))
                per_iter_lrs = np.interp(np.arange(iters), [0, iters - 1], [lr_start, lr_end])
                iter_ix = 0

            # Iterate over data.
            for imgName, inputs, labels in dataloaders[phase]:

                # Handling for different input types
                model_dtype = next(model.parameters()).dtype
                if not is_gist and not is_saliency:
                    inputs = inputs.to(device=device, dtype=model_dtype)
                elif is_gist and is_saliency:
                    inputs = [inp.to(device=device, dtype=model_dtype) for inp in inputs]
                elif is_gist or is_saliency:
                    inputs = [inputs[0].to(device=device, dtype=model_dtype), inputs[1].to(device=device, dtype=model_dtype)]

                labels = labels.to(device=device, dtype=torch.float32)

                # zero the parameter gradients and set per-iteration LR if applicable
                optimizer.zero_grad()
                if phase == TRAIN and getattr(args, 'lr_schedule_type', 'step') != 'step':
                    current_lr = float(per_iter_lrs[iter_ix])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

                # forward
                with torch.set_grad_enabled(phase == TRAIN):

                    if is_predict_two:
                        outputs1, outputs2 = model(inputs)  # Assume model returns two outputs
                        # Calculate loss for each output (arousal=head1, valence=head2)
                        loss1 = arousal_criterion(outputs1.view(-1), labels[:, 0].float())
                        loss2 = valence_criterion(outputs2.view(-1), labels[:, 1].float())
                        # Reweight to emphasize arousal
                        arousal_weight, valence_weight = get_loss_weights(epoch, num_epochs)
                        loss = arousal_weight * loss1 + valence_weight * loss2
                    else:
                        outputs = model(inputs)
                        # Some models may return a tuple even for single-output; take the first tensor
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                        # Compute loss in native output scale
                        loss = criterion(outputs.view(labels.size()), labels.float())


                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()
                        if getattr(args, 'lr_schedule_type', 'step') != 'step':
                            iter_ix += 1

                    running_loss += loss.item() * inputs.size(0)
                    if is_predict_two:
                        labels_list1.extend(labels[:, 0].cpu().numpy())
                        preds_list1.extend(outputs1.detach().cpu().numpy())
                        labels_list2.extend(labels[:, 1].cpu().numpy())
                        preds_list2.extend(outputs2.detach().cpu().numpy())
                    else:
                        labels_list.extend(labels.cpu().numpy())
                        preds = outputs.reshape(labels.shape).cpu().detach().numpy()
                        preds_list.extend(preds)

                # clean the cache
                if is_predict_two:
                    del inputs, labels, outputs1, outputs2, loss1, loss2, loss
                else:
                    del inputs, labels, outputs, loss, preds
                torch.cuda.empty_cache()

            if phase == TRAIN and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if is_predict_two:
                epoch_metrics = {}
                # Calculate metrics for the first variable
                epoch_metrics['R1'], _ = pearsonr(labels_list1, preds_list1)
                # Fitted-line R²/RMSE for Arousal
                coef1 = np.polyfit(preds_list1, labels_list1, 1)
                poly1d_fn1 = np.poly1d(coef1)
                fitted1 = poly1d_fn1(preds_list1)
                epoch_metrics['R2_1'] = r2_score(labels_list1, fitted1)
                epoch_metrics['RMSE1'] = np.sqrt(mean_squared_error(labels_list1, fitted1))
                # Calculate metrics for the second variable
                epoch_metrics['R2'], _ = pearsonr(labels_list2, preds_list2)
                # Fitted-line R²/RMSE for Valence
                coef2 = np.polyfit(preds_list2, labels_list2, 1)
                poly1d_fn2 = np.poly1d(coef2)
                fitted2 = poly1d_fn2(preds_list2)
                epoch_metrics['R2_2'] = r2_score(labels_list2, fitted2)
                epoch_metrics['RMSE2'] = np.sqrt(mean_squared_error(labels_list2, fitted2))

                print(
                    f"{phase} Loss: {epoch_loss:.4f} - R1: {epoch_metrics['R1']:.4f}"
                    f" R2_1: {epoch_metrics['R2_1']:.4f} RMSE1: {epoch_metrics['RMSE1']:.4f}"
                    f" - R2: {epoch_metrics['R2']:.4f} R2_2: {epoch_metrics['R2_2']:.4f}"
                    f" RMSE2: {epoch_metrics['RMSE2']:.4f}")


                # Calculate the average RMSE for both outputs
                avg_rmse = (epoch_metrics['RMSE1'] + epoch_metrics['RMSE2']) / 2

                #average the R and R2
                # avg_R = (epoch_metrics['R1'] + epoch_metrics['R2']) / 2
                # avg_R2 = (epoch_metrics['R2_1'] + epoch_metrics['R2_2']) / 2

                # record history row
                history_rows.append({
                    'epoch': epoch + 1,
                    'phase': phase,
                    'loss': float(epoch_loss),
                    'R1': float(epoch_metrics['R1']),
                    'R2_1': float(epoch_metrics['R2_1']),
                    'RMSE1': float(epoch_metrics['RMSE1']),
                    'R2': float(epoch_metrics['R2']),
                    'R2_2': float(epoch_metrics['R2_2']),
                    'RMSE2': float(epoch_metrics['RMSE2']),
                    'AVG_RMSE': float(avg_rmse),
                })

                # Choose best by average R2 (higher is better)
                avg_r2 = (epoch_metrics['R2_1'] + epoch_metrics['R2_2']) / 2.0
                improved = False
                if phase == VAL and avg_r2 > best_avg_r2:
                    improved = True
                    val_improved_this_epoch = True
                    best_avg_r2 = avg_r2
                    best_avg_rmse = avg_rmse
                    best_rmse = [epoch_metrics['RMSE1'], epoch_metrics['RMSE2']]
                    best_loss = epoch_loss
                    best_metrics['RMSE1'] = epoch_metrics['RMSE1']
                    best_metrics['R1'] = epoch_metrics['R1']
                    best_metrics['R2_1'] = epoch_metrics['R2_1']
                    best_metrics['RMSE2'] = epoch_metrics['RMSE2']
                    best_metrics['R2'] = epoch_metrics['R2']
                    best_metrics['R2_2'] = epoch_metrics['R2_2']
                    best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(
                        f'Best epoch: {best_epoch} Best loss: {best_loss:.4f} - avgR2: {best_avg_r2:.4f} '
                        f'(R2_1: {best_metrics["R2_1"]:.4f}, R2_2: {best_metrics["R2_2"]:.4f})')
                    # Save the best predictions and targets for both variables
                    best_predictions = [preds_list1, preds_list2]
                    all_targets = [labels_list1, labels_list2]

                    best_R = [epoch_metrics['R1'], epoch_metrics['R2']]
                    best_R2 = [epoch_metrics['R2_1'], epoch_metrics['R2_2']]
                    # Save the best checkpoint (overwrite same file name)
                    save_checkpoint_V2(
                        best_R=best_R,
                        best_R2=best_R2,
                        best_rmse=best_rmse,
                        best_epoch=best_epoch,
                        best_state_dict=best_model_wts,
                        optimizer=optimizer,
                        model_dir=model_dir,
                        model_name=model_name,
                    )
            else:
                epoch_R, _ = pearsonr(labels_list, preds_list)
                # Fitted-line R²/RMSE for single-output
                coef = np.polyfit(preds_list, labels_list, 1)
                poly1d_fn = np.poly1d(coef)
                fitted = poly1d_fn(preds_list)
                epoch_R2 = r2_score(labels_list, fitted)
                epoch_rmse = np.sqrt(mean_squared_error(labels_list, fitted))

                print('{} Loss: {:.4f} R: {:.4f} R2: {:.4f} RMSE: {:.4f}'.format(
                    phase, epoch_loss, epoch_R, epoch_R2, epoch_rmse))

                # record history row for single-output case
                history_rows.append({
                    'epoch': epoch + 1,
                    'phase': phase,
                    'loss': float(epoch_loss),
                    'R': float(epoch_R),
                    'R2': float(epoch_R2),
                    'RMSE': float(epoch_rmse),
                })

                improved = False
                if phase == VAL and epoch_R2 > best_R2:
                    improved = True
                    val_improved_this_epoch = True
                    best_R = epoch_R
                    best_R2 = epoch_R2
                    best_loss = epoch_loss
                    best_rmse = epoch_rmse
                    best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(model.state_dict())

                    print('Best epoch: {:} Best loss: {:.4f} Best R: {:.4f} Best R^2: {:.4f} RMSE: {:.4f}'.format(best_epoch, best_loss, best_R, best_R2, best_rmse))

                    all_targets = labels_list
                    best_predictions = preds_list
                    # Save the best checkpoint (overwrite same file name)
                    save_checkpoint_V2(
                        best_R=best_R,
                        best_R2=best_R2,
                        best_rmse=best_rmse,
                        best_epoch=best_epoch,
                        best_state_dict=best_model_wts,
                        optimizer=optimizer,
                        model_dir=model_dir,
                        model_name=model_name,
                    )


        if is_saveModel_epoch:
            # Assuming `model_name` is defined and includes the '.pth' extension.
            # Add epoch info into model name
            current_model_name = model_name[:-4] + '_epoch' + str(epoch + 1) + '.pth'
            # Save the model after each epoch
            save_checkpoint_V2(best_R=best_R,
                            best_R2=best_R2,
                            best_rmse=best_rmse,
                            best_epoch=epoch + 1,
                            best_state_dict=model.state_dict(),
                            optimizer=optimizer,
                            model_dir=model_dir,
                            model_name=current_model_name)

        print()

        # Early stopping check after completing both TRAIN and VAL phases
        if patience:
            if val_improved_this_epoch:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f'No val improvement for {no_improve_epochs}/{patience} epoch(s)')
                if no_improve_epochs >= patience:
                    print('Early stopping triggered.')
                    break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if is_predict_two:
        # Convert lists to numpy arrays for processing
        all_targets = [np.array(targets) for targets in all_targets]
        best_predictions = [np.array(predictions) for predictions in best_predictions]

        #print
        print('Best val R1: {:4f}'.format(best_metrics['R1']))
        print('Best val R2_1: {:4f}'.format(best_metrics['R2_1']))
        print('Best val RMSE1: {:4f}'.format(best_metrics['RMSE1']))
        print('Best val R2: {:4f}'.format(best_metrics['R2']))
        print('Best val R2_2: {:4f}'.format(best_metrics['R2_2']))
        print('Best val RMSE2: {:4f}'.format(best_metrics['RMSE2']))
    else:
        print('Best val R: {:4f}'.format(best_R))
        print('Best epoch: {:}'.format(best_epoch))
        print('Best loss: {:4f}'.format(best_loss))
        print('Best R^2: {:4f}'.format(best_R2))
        print('Best RMSE: {:4f}'.format(best_rmse))

        # After the loop, convert lists to numpy arrays for processing
        all_targets = np.array(all_targets)
        best_predictions = np.array(best_predictions)
    print()
    # Return targets and predictions along with other metrics and history
    return best_model_wts, best_loss, best_R, best_R2, best_rmse, best_epoch, all_targets, best_predictions, history_rows


# fundtion to test the model
def test_model(dataloader, dataset_size, model, criterion, device, is_predict_two=False):
    model.eval()  # Set the model to evaluation mode

    all_targets = []
    all_predictions = []
    labels_list = []
    preds_list = []
    labels_list1, preds_list1 = [], []
    labels_list2, preds_list2 = [], []
    # No gradient calculation needed
    with torch.no_grad():
        for _, inputs, labels in dataloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_predict_two:
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
        # Calculate Pearson r for the first variable
        metrics['r_Arousal'], _ = pearsonr(labels_list1, preds_list1)
        # Fitted-line R²/RMSE for Arousal
        coef1 = np.polyfit(preds_list1, labels_list1, 1)
        poly1d_fn1 = np.poly1d(coef1)
        fitted1 = poly1d_fn1(preds_list1)
        metrics['R2_Arousal'] = r2_score(labels_list1, fitted1)
        metrics['RMSE_Arousal'] = np.sqrt(mean_squared_error(labels_list1, fitted1))
        # Calculate Pearson r for the second variable
        metrics['r_Valence'], _ = pearsonr(labels_list2, preds_list2)
        # Fitted-line R²/RMSE for Valence
        coef2 = np.polyfit(preds_list2, labels_list2, 1)
        poly1d_fn2 = np.poly1d(coef2)
        fitted2 = poly1d_fn2(preds_list2)
        metrics['R2_Valence'] = r2_score(labels_list2, fitted2)
        metrics['RMSE_Valence'] = np.sqrt(mean_squared_error(labels_list2, fitted2))

        # print(
        #     f"r_Arousal: {metrics['r_Arousal']:.4f}"
        #     f" R2_Arousal: {metrics['R2_Arousal']:.4f} RMSE_Arousal: {metrics['RMSE_Arousal']:.4f}"
        #     f" - r_Valence: {metrics['r_Valence']:.4f} R2_Valence: {metrics['R2_Valence']:.4f}"
        #     f" RMSE_Valence: {metrics['RMSE_Valence']:.4f}")

        #save the best predictions and targets
        # Save the best predictions and targets for both variables
        best_predictions = [preds_list1, preds_list2]
        all_targets = [labels_list1, labels_list2]

        best_R = [metrics['r_Arousal'], metrics['r_Valence']]
        best_R2 = [metrics['R2_Arousal'], metrics['R2_Valence']]
        best_RMSE = [metrics['RMSE_Arousal'], metrics['RMSE_Valence']]

        # Convert lists to numpy arrays for processing
        all_targets = [np.array(targets) for targets in all_targets]
        best_predictions = [np.array(predictions) for predictions in best_predictions]

    else:
        best_R, _ = pearsonr(labels_list, preds_list)

        # Fitted-line R²/RMSE for single-output
        coef = np.polyfit(preds_list, labels_list, 1)
        poly1d_fn = np.poly1d(coef)
        fitted = poly1d_fn(preds_list)
        best_R2 = r2_score(labels_list, fitted)
        best_RMSE = np.sqrt(mean_squared_error(labels_list, fitted))

        # print('r: {:.4f} R2: {:.4f} RMSE: {:.4f}'.format(best_R, best_R2, best_RMSE))

        all_targets = labels_list
        best_predictions = preds_list

        # After the loop, convert lists to numpy arrays for processing
        all_targets = np.array(all_targets)
        best_predictions = np.array(best_predictions)



    # Return targets and predictions along with other metrics
    return  best_R, best_R2, best_RMSE, all_targets, best_predictions

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



def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.resultfolder):
        os.makedirs(args.resultfolder)

    if not os.path.exists(os.path.join(args.resultfolder, args.logfolder)):
        os.makedirs(os.path.join(args.resultfolder, args.logfolder))

    print('-' * 10)
    print("Preparing training data and it will take a while...")
    # Load full dataset for cross-validation
    train_csv_data = os.path.join(args.data_dir, args.train_csv_data)
    train_image_dir = os.path.join(args.data_dir, args.train_data)
    train_full_dataset = RegressionDataset(csv_file=train_csv_data, root_dir=train_image_dir,
                                     transform=None, is_gist=args.is_gist,
                                     is_saliency=args.is_saliency,
                                     isarousal=args.isarousal,
                                     is_predict_two=args.is_predict_two)

    #load test data
    test_csv_data = os.path.join(args.data_dir, args.test_csv_data)
    test_image_dir = os.path.join(args.data_dir, args.test_data)
    test_full_dataset = RegressionDataset(csv_file=test_csv_data, root_dir=test_image_dir,
                                     transform=None, is_gist=args.is_gist,
                                     is_saliency=args.is_saliency,
                                     isarousal=args.isarousal,
                                     is_predict_two=args.is_predict_two)






    # Initialize a list or dict to store test performances

    all_val_targets = []
    all_val_predictions = []

    all_val_targets_arousal = []
    all_val_targets_valence = []
    all_val_predictions_arousal = []
    all_val_predictions_valence = []

    test_performances = []
    test_performances_arousal = []
    test_performances_valence = []


    val_performances = []
    val_performances_arousal = []
    val_performances_valence = []


    all_test_targets = []
    all_test_predictions = []

    all_test_targets_arousal = []
    all_test_targets_valence = []
    all_test_predictions_arousal = []
    all_test_predictions_valence = []

    # repeat 5 times
    for repeat in range(args.repeat):
        print(f"Repeat {repeat + 1} of {args.repeat}")
        print('-' * 10)
        print('Initializing the model')
        start_epoch = 0

        is_log = args.is_log
        # Initialize model
        model, is_gist, is_saliency = initialize_model(args)




        if args.resume:
            if os.path.isfile(os.path.join(args.model_dir, args.resume)):
                print("=> loading checkpoint '{}'".format(args.resume))

                checkpoint = torch.load(os.path.join(args.model_dir, args.resume), map_location='cuda:0')
                start_epoch = args.start_epoch


                # Filter out the checkpoint to only include keys that exist in your model and have matching sizes
                model_state_dict = model.state_dict()
                filtered_checkpoint = {k: v for k, v in checkpoint.items() if
                                       k in model_state_dict and model_state_dict[k].size() == v.size()}

                # Load the filtered checkpoint into your model
                model.load_state_dict(filtered_checkpoint, strict=False)


                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, start_epoch))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # check which layers are trainable
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        criterion = nn.MSELoss()
        optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # For legacy behavior keep StepLR if schedule is 'step'; otherwise, handle manually per-iteration
        exp_lr_scheduler = None
        if getattr(args, 'lr_schedule_type', 'step') == 'step':
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=0.1)

        print(model)


        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            # Wrap the model with nn.DataParallel
            model = nn.DataParallel(model)
        else:
            if torch.cuda.is_available():
                print("Using single GPU!")
            else:
                print("GPU is not available")
                print("Using CPU!")

        model = model.to(device)

        # Total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params}")

        #split the training data into train and validation with ratio 9:1
        train_size = int(args.split_ratio * len(train_full_dataset))
        val_size = len(train_full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, val_size])

        # Combine indices from each group for this fold
        train_idx = train_dataset.indices
        val_idx = val_dataset.indices

        # Get dataloaders for the current fold
        dataloaders, dataset_sizes = reg_dataloader_KFold(train_full_dataset,
                                                          train_idx, val_idx,
                                                          args.batch_size, is_gist=args.is_gist,
                                                          is_saliency=args.is_saliency, isarousal=args.isarousal,
                                                          image_size=args.image_size)

        # Train the model
        model_name = args.model_name[:-4] + '_repeat' + str(repeat + 1) + '.pth'
        if args.transformation_type == 'noise':
            #add noise variance to the model name
            model_name = model_name[:-4] + '_noiselevel' + str(args.variance) + '.pth'


        # Capture the returned targets and predictions
        model_wts, val_loss, val_R, val_R2, val_RMSE, best_epoch, val_targets, val_predictions, history_rows = train_model(
            dataloaders, dataset_sizes, TRAIN='train', VAL='val', model=model,
            criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
            num_epochs=args.epoch, device=device, model_dir=args.model_dir, model_name=model_name,
            start_epoch=start_epoch, is_gist=is_gist, is_saliency=is_saliency,
            is_predict_two = args.is_predict_two, is_saveModel_epoch = args.is_saveModel_epoch,
            args=args)

        if args.epoch == 0:
            continue


        if args.is_predict_two:
            # Convert lists to numpy arrays for easier handling
            best_predictions_arrays = [np.array(preds) for preds in val_predictions]
            all_targets_arrays = [np.array(targets) for targets in val_targets]


            # Assuming best_predictions_arrays and all_targets_arrays are in the format:
            # [arousal_predictions, valence_predictions] and [arousal_targets, valence_targets]
            arousal_predictions, valence_predictions = best_predictions_arrays
            arousal_targets, valence_targets = all_targets_arrays

            # Process arousal
            variable_name = "Arousal"
            all_val_targets_arousal.append(arousal_targets)  # val_targets for arousal from current fold
            all_val_predictions_arousal.append(arousal_predictions)  # val_predictions for arousal from current fold
            plot_scatter_plot(arousal_targets, arousal_predictions, repeat, args, variable_name=variable_name, data_type='val')

            # Process valence
            variable_name = "Valence"
            all_val_targets_valence.append(valence_targets)  # val_targets for valence from current fold
            all_val_predictions_valence.append(valence_predictions)  # val_predictions for valence from current fold
            plot_scatter_plot(valence_targets, valence_predictions, repeat, args, variable_name=variable_name, data_type='val')

        else:
            # Append the targets and predictions for the fold to the lists
            all_val_targets.append(val_targets)  # val_targets from current fold
            all_val_predictions.append(val_predictions)  # val_predictions from current fold

            plot_scatter_plot(np.array(val_targets), np.array(val_predictions), repeat, args,variable_name='', data_type='val')


        # Save the best model (overwrite same file per repeat)
        save_checkpoint_V2(
            best_R=val_R,
            best_R2=val_R2,
            best_rmse=val_RMSE,
            best_epoch=best_epoch,
            best_state_dict=model_wts,
            optimizer=optimizer_ft,
            model_dir=args.model_dir,
            model_name=model_name,
        )

        # Save per-epoch training/validation history for this repeat
        if history_rows:
            history_df = pd.DataFrame(history_rows)
            history_path = os.path.join(args.resultfolder, args.logfolder, f"history_repeat{repeat + 1}.csv")
            history_df.to_csv(history_path, index=False)
            # Plot training history
            try:
                # Ensure epochs are integers for plotting
                history_df['epoch'] = history_df['epoch'].astype(int)
                # Common: Loss plot (TRAIN vs VAL)
                plt.figure(figsize=(7, 5))
                for phase_name in ['train', 'val']:
                    if (history_df['phase'] == phase_name).any():
                        phase_df = history_df[history_df['phase'] == phase_name]
                        plt.plot(phase_df['epoch'], phase_df['loss'], label=f'{phase_name.upper()}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Loss per Epoch (repeat {repeat + 1})')
                plt.legend()
                plt.grid(True)
                loss_plot_path = os.path.join(args.resultfolder, args.logfolder, f"history_repeat{repeat + 1}_loss.png")
                plt.savefig(loss_plot_path)
                plt.close()

                if args.is_predict_two:
                    # Arousal metrics: R1, R2_1, RMSE1
                    for metric_col, nice_name in [('R1', 'R'), ('R2_1', 'R2'), ('RMSE1', 'RMSE')]:
                        if metric_col in history_df.columns:
                            plt.figure(figsize=(7, 5))
                            for phase_name in ['train', 'val']:
                                if (history_df['phase'] == phase_name).any():
                                    phase_df = history_df[history_df['phase'] == phase_name]
                                    plt.plot(phase_df['epoch'], phase_df[metric_col], label=f'{phase_name.upper()}')
                            plt.xlabel('Epoch')
                            plt.ylabel(nice_name)
                            plt.title(f'Arousal {nice_name} per Epoch (repeat {repeat + 1})')
                            plt.legend()
                            plt.grid(True)
                            plot_path = os.path.join(args.resultfolder, args.logfolder, f"history_repeat{repeat + 1}_arousal_{metric_col}.png")
                            plt.savefig(plot_path)
                            plt.close()

                    # Valence metrics: R2, R2_2, RMSE2
                    for metric_col, nice_name in [('R2', 'R'), ('R2_2', 'R2'), ('RMSE2', 'RMSE')]:
                        if metric_col in history_df.columns:
                            plt.figure(figsize=(7, 5))
                            for phase_name in ['train', 'val']:
                                if (history_df['phase'] == phase_name).any():
                                    phase_df = history_df[history_df['phase'] == phase_name]
                                    plt.plot(phase_df['epoch'], phase_df[metric_col], label=f'{phase_name.upper()}')
                            plt.xlabel('Epoch')
                            plt.ylabel(nice_name)
                            plt.title(f'Valence {nice_name} per Epoch (repeat {repeat + 1})')
                            plt.legend()
                            plt.grid(True)
                            plot_path = os.path.join(args.resultfolder, args.logfolder, f"history_repeat{repeat + 1}_valence_{metric_col}.png")
                            plt.savefig(plot_path)
                            plt.close()
                else:
                    # Single-output metrics: R, R2, RMSE
                    for metric_col, nice_name in [('R', 'R'), ('R2', 'R2'), ('RMSE', 'RMSE')]:
                        if metric_col in history_df.columns:
                            plt.figure(figsize=(7, 5))
                            for phase_name in ['train', 'val']:
                                if (history_df['phase'] == phase_name).any():
                                    phase_df = history_df[history_df['phase'] == phase_name]
                                    plt.plot(phase_df['epoch'], phase_df[metric_col], label=f'{phase_name.upper()}')
                            plt.xlabel('Epoch')
                            plt.ylabel(nice_name)
                            plt.title(f'{nice_name} per Epoch (repeat {repeat + 1})')
                            plt.legend()
                            plt.grid(True)
                            plot_path = os.path.join(args.resultfolder, args.logfolder, f"history_repeat{repeat + 1}_{metric_col}.png")
                            plt.savefig(plot_path)
                            plt.close()
            except Exception as e:
                print(f"Warning: failed to plot history for repeat {repeat + 1}: {e}")


        #test the model on the test data and save the results
        test_dataloader, test_dataset_size = reg_dataloader_testdata(test_full_dataset, args.batch_size, is_gist=args.is_gist,
                                                                is_saliency=args.is_saliency, isarousal=args.isarousal,
                                                                image_size=args.image_size, transformation_type=args.transformation_type,
                                                                variance=args.variance)



        # Load the best model
        best_model = model
        best_model.load_state_dict(model_wts)
        # Test the model
        test_R, test_R2, test_RMSE, test_targets, test_predictions = test_model(
            test_dataloader, test_dataset_size, model=best_model, criterion=criterion, device=device,
            is_predict_two=args.is_predict_two)



        # Record the validation performance
        if args.is_predict_two:
            print('-' * 10)
            print(f"Repeat {repeat + 1}  Test - r_Arousal: {test_R[0]:.4f},"
                  f" R2_Arousal: {test_R2[0]:.4f}, RMSE_Arousal: {test_RMSE[0]:.4f},"
                  f" r_Valence: {test_R[1]:.4f}, R2_Valence: {test_R2[1]:.4f}, RMSE_Valence: {test_RMSE[1]:.4f}")

            test_performances_arousal.append({'repeat': repeat + 1,
                                                    'r': test_R[0], 'R2': test_R2[0],
                                                    'RMSE': test_RMSE[0]})
            test_performances_valence.append({'repeat': repeat + 1,
                                                    'r': test_R[1], 'R2': test_R2[1],
                                                    'RMSE': test_RMSE[1]})

            val_performances_arousal.append({'repeat': repeat + 1,
                                                    'r': val_R[0], 'R2': val_R2[0],
                                                    'RMSE': val_RMSE[0]})
            val_performances_valence.append({'repeat': repeat + 1,
                                                    'r': val_R[1], 'R2': val_R2[1],
                                                    'RMSE': val_RMSE[1]})


            #save the predictions and targets
            all_test_targets_arousal.append(test_targets[0])
            all_test_predictions_arousal.append(test_predictions[0])
            all_test_targets_valence.append(test_targets[1])
            all_test_predictions_valence.append(test_predictions[1])

            #save the scatter plot
            plot_scatter_plot(test_targets[0], test_predictions[0], repeat, args, variable_name='Arousal', data_type='test')
            plot_scatter_plot(test_targets[1], test_predictions[1], repeat, args, variable_name='Valence', data_type='test')


        else:
            print('-' * 10)
            print(f"Repeat {repeat + 1} Test - r: {test_R:.4f},"
              f"R2: {test_R2:.4f}, RMSE: {test_RMSE:.4f}")

            test_performances.append({'repeat': repeat + 1, 'r': test_R,
                                            'R2': test_R2, 'RMSE': test_RMSE,
                                     })

            val_performances.append({'repeat': repeat + 1, 'r': val_R,
                                            'R2': val_R2, 'RMSE': val_RMSE,
                                     })

            #save the predictions and targets
            all_test_targets.append(test_targets)
            all_test_predictions.append(test_predictions)

            #save the scatter plot
            variable_name = 'Valence' if not args.isarousal else 'Arousal'
            plot_scatter_plot(test_targets, test_predictions, repeat, args, variable_name=variable_name, data_type='test')


    if args.epoch == 0:
        return

    # After all folds are completed, print or save the validation performances
    # Save the validation performances to a CSV file
    import csv
    if not args.is_predict_two:
        with open(os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4] + '_TestDATA.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Repeat', 'Test_R', 'Test_R2', 'RMSE'])
            for performance in test_performances:
                writer.writerow([performance['repeat'], performance['r'], performance['R2'], performance['RMSE']])

        # save validation performances to a csv file
        with open(os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4] + '_ValidationDATA.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Repeat', 'Val_R', 'Val_R2', 'RMSE'])
            for performance in val_performances:
                writer.writerow([performance['repeat'], performance['r'], performance['R2'], performance['RMSE']])


        print('-' * 10)
        print("Test performance saved to csv")

        # Extract validation losses and R values from the validation_performances list
        test_Rs = [performance['r'] for performance in test_performances]
        test_R2s = [performance['R2'] for performance in test_performances]
        test_RMSEs = [performance['RMSE'] for performance in test_performances]

        # Convert to Python scalars if they are tensors
        test_Rs = [test_R.item() if hasattr(test_R, 'item') else test_R for test_R in test_Rs]
        test_R2s = [test_R2.item() if hasattr(test_R2, 'item') else test_R2 for test_R2 in test_R2s]
        test_RMSEs = [test_RMSE.item() if hasattr(test_RMSE, 'item') else test_RMSE for test_RMSE in test_RMSEs]

        # Calculate and print the average validation loss and R value
        average_test_R = np.mean(test_Rs)
        average_test_R2 = np.mean(test_R2s)
        average_test_RMSE = np.mean(test_RMSEs)

        print('-' * 10)
        print(f" Average Test R: {average_test_R:.4f}, "
              f"Average Test R2: {average_test_R2:.4f}, Average Test RMSE: {average_test_RMSE:.4f}")


        ######### Record Validation Targets and Predictions###################
        data_frames = []
        for targets, predictions in zip(all_val_targets, all_val_predictions):
            df_repeat = pd.DataFrame({
                'Target': targets,
                'Prediction': predictions
            })
            data_frames.append(df_repeat)

        # Concatenate all DataFrame objects
        data_to_save = pd.concat(data_frames, ignore_index=True)

        #save concatenated_targets and concatenated_predictions into a csv file
        file_path = os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]
                                 +'_ValidationDATA_targets_prediction.csv')
        data_to_save.to_csv(file_path, index=False)

        # Perform quantile analysis on the targets and predictions

        # Perform quantile analysis on arousal data
        quantile_analysis_25 = perform_quantile_analysis(
            actual=data_to_save['Target'],
            predicted=data_to_save['Prediction'],
            n_quantiles=25
        )

        #print the results
        print('-' * 10)
        print('Validation DATA: Quantile Analysis Results')
        print('-' * 10)
        print('25 Quantiles:', quantile_analysis_25)


        ######### Record Test Targets and Predictions###################
        data_frames = []
        for targets, predictions in zip(all_test_targets, all_test_predictions):
            df_repeat = pd.DataFrame({
                'Target': targets,
                'Prediction': predictions
            })
            data_frames.append(df_repeat)

        # Concatenate all DataFrame objects
        data_to_save = pd.concat(data_frames, ignore_index=True)

        #save concatenated_targets and concatenated_predictions into a csv file
        file_path = os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]
                                 + '_TestDATA_targets_prediction.csv')
        data_to_save.to_csv(file_path, index=False)

        # Perform quantile analysis on the targets and predictions
        # using 10, 15, 20, and 25 quantiles, separately
        quantile_analysis_25 = perform_quantile_analysis(
            actual=data_to_save['Target'],
            predicted=data_to_save['Prediction'],
            n_quantiles=25
        )

        #print the results
        print('-' * 10)
        print('Test DATA: Quantile Analysis Results')
        print('-' * 10)
        print('25 Quantiles:', quantile_analysis_25)

    else:

        with open(os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]+'_TestDATA_Arousal.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Repeat', 'Test_R', 'Test_R2', 'RMSE'])
            for performance in test_performances_arousal:
                writer.writerow([performance['repeat'], performance['r'], performance['R2'], performance['RMSE']])
        print('-' * 10)
        print("Test performances Arousal saved to validation_performances_arousal.csv")

        with open(os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]+'_TestDATA_Valence.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Repeat', 'Test_R', 'Test_R2', 'RMSE'])
            for performance in test_performances_valence:
                writer.writerow([performance['repeat'], performance['r'], performance['R2'], performance['RMSE']])
        print('-' * 10)
        print("Test performances Valence saved to validation_performances_valence.csv")


        #save validation performances to a csv file
        with open(os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]+'_ValidationDATA_Arousal.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Repeat', 'Val_R', 'Val_R2', 'RMSE'])
            for performance in val_performances_arousal:
                writer.writerow([performance['repeat'], performance['r'], performance['R2'], performance['RMSE']])
        print('-' * 10)
        print("Validation performances Arousal saved to validation_performances_arousal.csv")

        with open(os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]+'_ValidationDATA_Valence.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Repeat', 'Val_R', 'Val_R2', 'RMSE'])
            for performance in val_performances_valence:
                writer.writerow([performance['repeat'], performance['r'], performance['R2'], performance['RMSE']])
        print('-' * 10)
        print("Validation performances Valence saved to validation_performances_valence.csv")

        # Extract validation losses and R values from the validation_performances list
        test_Rs_arousal = [performance['r'] for performance in test_performances_arousal]
        test_R2s_arousal = [performance['R2'] for performance in test_performances_arousal]
        test_RMSEs_arousal = [performance['RMSE'] for performance in test_performances_arousal]

        test_Rs_valence = [performance['r'] for performance in test_performances_valence]
        test_R2s_valence = [performance['R2'] for performance in test_performances_valence]
        test_RMSEs_valence = [performance['RMSE'] for performance in test_performances_valence]


        # Convert to Python scalars if they are tensors
        test_Rs_arousal = [test_R.item() if hasattr(test_R, 'item') else test_R for test_R in test_Rs_arousal]
        test_R2s_arousal = [test_R2.item() if hasattr(test_R2, 'item') else test_R2 for test_R2 in test_R2s_arousal]
        test_RMSEs_arousal = [test_RMSE.item() if hasattr(test_RMSE, 'item') else test_RMSE for test_RMSE in test_RMSEs_arousal]

        test_Rs_valence = [test_R.item() if hasattr(test_R, 'item') else test_R for test_R in test_Rs_valence]
        test_R2s_valence = [test_R2.item() if hasattr(test_R2, 'item') else test_R2 for test_R2 in test_R2s_valence]
        test_RMSEs_valence = [test_RMSE.item() if hasattr(test_RMSE, 'item') else test_RMSE for test_RMSE in test_RMSEs_valence]


        # Calculate and print the average validation loss and R value
        average_test_R_arousal = np.mean(test_Rs_arousal)
        average_test_R2_arousal = np.mean(test_R2s_arousal)
        average_test_RMSE_arousal = np.mean(test_RMSEs_arousal)

        average_test_R_valence = np.mean(test_Rs_valence)
        average_test_R2_valence = np.mean(test_R2s_valence)
        average_test_RMSE_valence = np.mean(test_RMSEs_valence)



        print('-' * 10)


        print(f"Average Test r Valence: {average_test_R_valence:.4f}, Average Test R2 Valence: {average_test_R2_valence:.4f}, "
                f"Average RMSE Valence: {average_test_RMSE_valence:.4f}")

        
        print(f"Average Test r Arousal: {average_test_R_arousal:.4f}, Average Test R2 Arousal: {average_test_R2_arousal:.4f}, "
              f"Average RMSE Arousal: {average_test_RMSE_arousal:.4f}")



        ######### Record Validation Targets and Predictions ###################

        data_frames_arousal = []
        for targets, predictions in zip(all_val_targets_arousal, all_val_predictions_arousal):
            df_repeat = pd.DataFrame({
                'Target': targets,
                'Prediction': predictions
            })
            data_frames_arousal.append(df_repeat)

        # Concatenate all DataFrame objects
        data_to_save_arousal = pd.concat(data_frames_arousal, ignore_index=True)

        # save concatenated_targets and concatenated_predictions into a csv file
        file_path_arousal = os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]
                                 + '_ValidationDATA_targets_prediction_arousal.csv')
        data_to_save_arousal.to_csv(file_path_arousal, index=False)

        # Perform quantile analysis on the targets and predictions
        # using 10, 15, 20, and 25 quantiles, separately
        quantile_analysis_25_arousal = perform_quantile_analysis(
            actual=data_to_save_arousal['Target'],
            predicted=data_to_save_arousal['Prediction'],
            n_quantiles=25
        )

        #print the results
        print('-' * 10)
        print('Validation DATA: Quantile Analysis Results Arousal')
        print('-' * 10)
        print('25 Quantiles:', quantile_analysis_25_arousal)


        # Similarly, for valence
        data_frames_valence = []
        for targets, predictions in zip(all_val_targets_valence, all_val_predictions_valence):
            df_repeat = pd.DataFrame({
                'Target': targets,
                'Prediction': predictions
            })
            data_frames_valence.append(df_repeat)

        data_to_save_valence = pd.concat(data_frames_valence, ignore_index=True)

        # save concatenated_targets and concatenated_predictions into a csv file
        file_path_valence = os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]
                                 + '_ValidationDATA_targets_prediction_valence.csv')
        data_to_save_valence.to_csv(file_path_valence, index=False)

        # Perform quantile analysis on the targets and predictions
        # using 10, 15, 20, and 25 quantiles, separately
        quantile_analysis_25_valence = perform_quantile_analysis(
            actual=data_to_save_valence['Target'],
            predicted=data_to_save_valence['Prediction'],
            n_quantiles=25
        )
        #print the results
        print('-' * 10)
        print('Validation DATA: Quantile Analysis Results Valence')
        print('-' * 10)
        print('25 Quantiles:', quantile_analysis_25_valence)




        ######### Record Test Targets and Predictions ###################

        data_frames_arousal = []
        for targets, predictions in zip(all_test_targets_arousal, all_test_predictions_arousal):
            df_repeat = pd.DataFrame({
                'Target': targets,
                'Prediction': predictions
            })
            data_frames_arousal.append(df_repeat)

        # Concatenate all DataFrame objects
        data_to_save_arousal = pd.concat(data_frames_arousal, ignore_index=True)

        # save concatenated_targets and concatenated_predictions into a csv file
        file_path_arousal = os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]
                                 + '_TestDATA_targets_prediction_arousal.csv')
        data_to_save_arousal.to_csv(file_path_arousal, index=False)

        # Perform quantile analysis on the targets and predictions
        # using 10, 15, 20, and 25 quantiles, separately
        quantile_analysis_25_arousal = perform_quantile_analysis(
            actual=data_to_save_arousal['Target'],
            predicted=data_to_save_arousal['Prediction'],
            n_quantiles=25
        )

        #print the results
        print('-' * 10)
        print('Test DATA: Quantile Analysis Results Arousal')
        print('-' * 10)
        print('25 Quantiles:', quantile_analysis_25_arousal)

        # Similarly, for valence
        data_frames_valence = []
        for targets, predictions in zip(all_test_targets_valence, all_test_predictions_valence):
            df_repeat = pd.DataFrame({
                'Target': targets,
                'Prediction': predictions
            })
            data_frames_valence.append(df_repeat)

        data_to_save_valence = pd.concat(data_frames_valence, ignore_index=True)

        # save concatenated_targets and concatenated_predictions into a csv file
        file_path_valence = os.path.join(args.resultfolder, args.logfolder, args.validation_performances[:-4]
                                 + '_TestDATA_targets_prediction_valence.csv')
        data_to_save_valence.to_csv(file_path_valence, index=False)

        # Perform quantile analysis on the targets and predictions
        # using 10, 15, 20, and 25 quantiles, separately
        quantile_analysis_25_valence = perform_quantile_analysis(
            actual=data_to_save_valence['Target'],
            predicted=data_to_save_valence['Prediction'],
            n_quantiles=25
        )
        #print the results
        print('-' * 10)
        print('Test DATA: Quantile Analysis Results Valence')
        print('-' * 10)
        print('25 Quantiles:', quantile_analysis_25_valence)



if __name__ == '__main__':

        main()
