import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.VCA import VCA
from utils.utils import initialize_model

from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset, Subset
import PIL
from PIL import Image
from torchvision import transforms

def data_transform(image_size=224):
    """
    Apply data transformations for test image processing.

    Returns:
    dict: Dictionary containing the transformations for each data type.
    """
    print("Applying data transformations...")
    print("Image size: {}".format(image_size))

    data_transforms = {}

    transform_list = [transforms.Resize(size=(image_size, image_size))]
    transform_list.append(transforms.ToTensor())

    data_transforms = transforms.Compose(transform_list)

    return data_transforms

class SubsetWithTransform(Subset):
    """
    Subset of a dataset at specified indices with a transformation applied.

    Args:
        dataset (Dataset): The whole Dataset.
        indices (sequence): Indices in the whole set selected for subset.
        transform (callable): Transform to be applied to each sample.
    """
    def __init__(self, dataset, indices, transform=None):
        super(SubsetWithTransform, self).__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        try:
            sample = self.dataset[self.indices[idx]]

            # Check if sample is a tuple (image, label)
            if isinstance(sample, tuple):
                name, image, label = sample

                # Convert image to PIL Image if it's not already
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)

                # Apply transformation to the image only
                if self.transform:
                    image = self.transform(image)

                return name, image, label
            else:
                # If sample is not a tuple, apply transform directly
                if self.transform:
                    sample = self.transform(sample)

                return sample
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise

class RegressionDataset(Dataset):
    """dataset only used for Regression ."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        self.label_csv = pd.read_csv(csv_file, header=None,  dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.label_csv.iloc[idx, 0]
        if isinstance(name, int):
            name = str(name) +'.'+self.label_csv.iloc[idx, 1]
        elif isinstance(name, float):
            name = str(int(name) if int(name)== name  else name)
            name = name +'.'+self.label_csv.iloc[idx, 1]
        else:
            name = str(name) + '.'+self.label_csv.iloc[idx, 1]

        img_name = os.path.join(self.root_dir, name)
        image = Image.open(img_name).convert('RGB')  # Converts any image to RGB

        label = self.label_csv.iloc[idx, 2:4].values.astype(float)  # Adjusted to fetch two labels
        label = torch.tensor(label, dtype=torch.float)

        if self.transform:
            if image.mode == 'RGBA':
                print('RGBA image')
                image.load()
                background = PIL.Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background

            output_image = self.transform(image)
        else:
            output_image = image

        return name, output_image, label

def reg_dataloader_testdata(full_dataset, batch_size, image_size=224):
    data_transforms = data_transform(image_size=image_size)

    test_dataset = SubsetWithTransform(full_dataset, range(len(full_dataset)), transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return {'test': test_loader}, {'test': len(test_dataset)}

# Model is now imported from models.VCA

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
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model.to(device)

def save_results_to_csv(image_names, actual_vals, predicted_vals, args, data_type='test'):
    if actual_vals is None:
        # Inference-only mode (no ground-truth labels)
        results_df = pd.DataFrame({
            'Image Name': image_names,
            'Predicted Valence': predicted_vals[0],
            'Predicted Arousal': predicted_vals[1] if len(predicted_vals) > 1 else np.nan
        })
    else:
        results_df = pd.DataFrame({
            'Image Name': image_names,
            'Actual Valence': actual_vals[0],
            'Predicted Valence': predicted_vals[0],
            'Actual Arousal': actual_vals[1] if len(actual_vals) > 1 else np.nan,
            'Predicted Arousal': predicted_vals[1] if len(predicted_vals) > 1 else np.nan
        })
    results_df.to_csv(os.path.join(args.resultfolder, f'{data_type}_results.csv'), index=False)

def compute_metrics(labels_list1, preds_list1, labels_list2=None, preds_list2=None):
    metrics = {
        'r_Arousal': pearsonr(labels_list1, preds_list1)[0],
        'R2_Arousal': r2_score(labels_list1, preds_list1),
        'RMSE_Arousal': np.sqrt(mean_squared_error(labels_list1, preds_list1))
    }
    if labels_list2 and preds_list2:
        metrics.update({
            'r_Valence': pearsonr(labels_list2, preds_list2)[0],
            'R2_Valence': r2_score(labels_list2, preds_list2),
            'RMSE_Valence': np.sqrt(mean_squared_error(labels_list2, preds_list2))
        })
    return metrics

def test_model(dataloader, model, device, min_samples_for_metrics=100):
    model.eval()
    all_image_names, all_targets, all_predictions = [], [], []
    labels_list1, preds_list1 = [], []
    labels_list2, preds_list2 = [], []

    with torch.no_grad():
        for batch_idx, (image_names, inputs, labels) in enumerate(dataloader['test']):
            print(f"Processing batch {batch_idx+1}/{len(dataloader['test'])}")
            inputs, labels = inputs.to(device), labels.to(device)
            all_image_names.extend(image_names)

            outputs1, outputs2 = model(inputs)

            labels_list1.extend(labels[:, 0].cpu().numpy())
            preds_list1.extend(outputs1.detach().cpu().numpy())
            labels_list2.extend(labels[:, 1].cpu().numpy())
            preds_list2.extend(outputs2.detach().cpu().numpy())

    if len(labels_list1) >= min_samples_for_metrics:
        metrics = compute_metrics(labels_list1, preds_list1, labels_list2, preds_list2)
    else:
        metrics = {}

    return metrics, [labels_list1, labels_list2], [preds_list1, preds_list2], all_image_names


class ImageFolderDataset(Dataset):
    """Simple dataset that loads all images in a directory (no labels)."""
    def __init__(self, root_dir, transform=None, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        self.root_dir = root_dir
        self.transform = transform
        self.exts = tuple(ext.lower() for ext in exts)
        files = []
        for name in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in self.exts:
                files.append(name)
        self.filenames = files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return name, image


def infer_model(dataloader, model, device):
    model.eval()
    image_names, preds1, preds2 = [], [], []
    with torch.no_grad():
        for batch_idx, (names, inputs) in enumerate(dataloader['test']):
            print(f"Processing batch {batch_idx+1}/{len(dataloader['test'])}")
            inputs = inputs.to(device)
            out1, out2 = model(inputs)
            image_names.extend(list(names))
            preds1.extend(out1.detach().cpu().numpy())
            preds2.extend(out2.detach().cpu().numpy())
    return image_names, [preds1, preds2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test or run inference with a trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--csv_path', type=str, default='', help='Path to labels CSV (optional for inference-only)')
    parser.add_argument('--test_img_dir', type=str, required=True, help='Directory with test/inference images')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--image_size', type=int, required=True, help='Image size (assuming square images)')
    parser.add_argument('--resultfolder', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--model_to_run', default=71, type=int,
                        help='71: VCA (VCA)')
    parser.add_argument('--is_predict_two', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to predict two labels at the same time')
    parser.add_argument('--clip_model_name', default='ViT-L/14', type=str, help='the name of the CLIP model')

    args = parser.parse_args()

    if not os.path.exists(args.resultfolder):
        os.makedirs(args.resultfolder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Initializing model (aligned with testing)')
    model, _, _ = initialize_model(args)
    model = load_model(args.model_path, model, device)

    if args.csv_path:
        dataset = RegressionDataset(csv_file=args.csv_path, root_dir=args.test_img_dir, transform=None)
        test_dataloader, _ = reg_dataloader_testdata(dataset, args.batch_size)

        print("Starting model testing...")
        test_metrics, test_targets, test_predictions, image_names = test_model(test_dataloader, model, device)

        if test_metrics:
            print(f"Test - r_Arousal: {test_metrics['r_Arousal']:.4f}, R2_Arousal: {test_metrics['R2_Arousal']:.4f}, RMSE_Arousal:"
                  f" {test_metrics['RMSE_Arousal']:.4f}, r_Valence: {test_metrics['r_Valence']:.4f}, R2_Valence:"
                  f" {test_metrics['R2_Valence']:.4f}, RMSE_Valence: {test_metrics['RMSE_Valence']:.4f}")

        print("Saving results to CSV...")
        save_results_to_csv(image_names, test_targets, test_predictions, args, data_type='test')
        print("Testing complete.")
    else:
        # Inference-only path: no CSV labels
        print("Starting inference (no CSV labels provided)...")
        tfm = data_transform(image_size=args.image_size)
        infer_ds = ImageFolderDataset(root_dir=args.test_img_dir, transform=tfm)
        infer_loader = DataLoader(SubsetWithTransform(infer_ds, range(len(infer_ds))),
                                  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        dataloader = {'test': infer_loader}
        image_names, predicted_vals = infer_model(dataloader, model, device)
        print("Saving predictions to CSV...")
        save_results_to_csv(image_names, actual_vals=None, predicted_vals=predicted_vals, args=args, data_type='inference')
        print("Inference complete.")
