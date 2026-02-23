"""
Dataloader functions for VCA model.
Extracted and organized from the main dataloader.py file.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageFilter
import pandas as pd
import os
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
def collate_with_names(batch):
    """Custom collate_fn that preserves file names and stacks tensors.
    Batch items are (name: str, image: Tensor or PIL, label: Tensor).
    """
    names, images, labels = zip(*batch)
    # Ensure tensors
    tensor_images = []
    for im in images:
        if isinstance(im, Image.Image):
            t = transforms.ToTensor()(im)
        elif isinstance(im, torch.Tensor):
            t = im
        else:
            # fallback: try to convert from numpy
            t = torch.as_tensor(im)
            if t.dtype == torch.uint8:
                t = t.float().div(255.0)
            if t.ndim == 2:
                t = t.unsqueeze(0)
        tensor_images.append(t)

    # Resize to a fixed size if shapes mismatch
    target_h, target_w = 224, 224
    for i, t in enumerate(tensor_images):
        if t.shape[-2:] != (target_h, target_w):
            # Expect CHW; ensure batch dim for interpolate
            tensor_images[i] = F.interpolate(t.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)

    images = torch.stack(tensor_images, dim=0)
    labels = torch.stack(labels, dim=0)
    return names, images, labels


class RegressionDataset(Dataset):
    """Dataset for regression tasks."""

    def __init__(self, csv_file, root_dir, transform=None, istrain=True,
                 is_gist=False, transform_gist=None, is_saliency=False,
                 transform_saliency=None, isarousal=False, is_predict_two=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_gist: whether use gist features as the input of low-road
            is_saliency: whether use saliency features
            isarousal: whether to use arousal as label
            is_predict_two: whether to predict both valence and arousal
        """
        self.label_csv = pd.read_csv(csv_file, header=None, dtype=str)
        self.root_dir = root_dir
        self.transform = transform
        self.istrain = istrain
        self.is_gist = is_gist
        self.transform_gist = transform_gist
        self.is_saliency = is_saliency
        self.transform_saliency = transform_saliency
        self.isarousal = isarousal
        self.is_predict_two = is_predict_two

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.label_csv.iloc[idx, 0]
        if isinstance(name, int):
            name = str(name) + '.' + self.label_csv.iloc[idx, 1]
        elif isinstance(name, float):
            name = str(int(name) if int(name) == name else name)
            name = name + '.' + self.label_csv.iloc[idx, 1]
        else:
            name = str(name) + '.' + self.label_csv.iloc[idx, 1]

        img_name = os.path.join(self.root_dir, name)
        image = Image.open(img_name).convert('RGB')  # Converts any image to RGB

        if self.is_predict_two:
            label = self.label_csv.iloc[idx, 2:4].values.astype(float)
        elif self.isarousal:
            label = float(self.label_csv.iloc[idx, 2])
        else:
            label = float(self.label_csv.iloc[idx, 3])

        label = torch.tensor(label, dtype=torch.float)

        if self.transform:
            if image.mode == 'RGBA':
                image.load()
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            output_image = self.transform(image)
        else:
            output_image = image

        # For VCA, we only return the image (no gist/saliency)
        # Note: gist and saliency features are not used for VCA
        # but the logic is kept here for compatibility if needed in the future
        return name, output_image, label


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


def data_transform(train_folder=False, val_folder=None, test_folder=None,
                   is_gist=False, is_saliency=False, image_size=224,
                   transformation_type=None, variance=10, hue_shift=0, alpha=0.5):
    """
    Apply data transformations for image processing.

    Parameters:
        train_folder (str): Path to the training data folder.
        val_folder (str, optional): Path to the validation data folder.
        test_folder (str, optional): Path to the test data folder.
        is_gist (bool, optional): Flag to determine if GIST transform should be applied.
        is_saliency (bool, optional): Flag to determine if Saliency transform should be applied.
        image_size (int): Target image size
        transformation_type (str): Type of transformation for test data
        variance (int): Variance for noise transformation
        hue_shift (int): Hue shift for color transformation
        alpha (float): Alpha for blending transformations

    Returns:
        dict: Dictionary containing the transformations for each data type.
    """
    import cv2
    import numpy as np
    import random
    from PIL import ImageFilter
    
    # Helper functions for transformations
    def add_gaussian_noise(image, mean=0, var=10):
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
        noisy_image = cv2.add(image, gauss)
        return noisy_image

    def adjust_color(image, hue_shift=0, saturation_scale=1, brightness_scale=1):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image = np.array(hsv_image, dtype=np.float64)
        hsv_image[:, :, 0] = hsv_image[:, :, 0] + hue_shift
        hsv_image[:, :, 0][hsv_image[:, :, 0] > 255] = 255
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_scale
        hsv_image[:, :, 1][hsv_image[:, :, 1] > 255] = 255
        hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_scale
        hsv_image[:, :, 2][hsv_image[:, :, 2] > 255] = 255
        hsv_image = np.array(hsv_image, dtype=np.uint8)
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return rgb_image

    def adjust_lighting(image, alpha=1.0, beta=0):
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted_image

    def convert_to_grayscale(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    print("Applying data transformations...")
    print("Image size: {}".format(image_size))

    data_transforms = {}

    # Always apply transformations for training data
    if train_folder:
        if transformation_type == 'noise':
            data_transforms[train_folder] = transforms.Compose([
                transforms.Resize(size=(image_size, image_size)),
                transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda img: add_gaussian_noise(np.array(img), mean=0, var=variance)),
                transforms.ToTensor()
            ])
        else:
            # GaussianBlur augmentation class
            class GaussianBlur(object):
                """Gaussian blur augmentation"""
                def __init__(self, sigma=[.1, 2.]):
                    self.sigma = sigma
                def __call__(self, x):
                    sigma = random.uniform(self.sigma[0], self.sigma[1])
                    x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
                    return x
            
            data_transforms[train_folder] = transforms.Compose([
                transforms.Resize(size=(image_size, image_size)),
                transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
                transforms.RandomApply([GaussianBlur([0.1, 2.])], p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    # Apply transformations for validation data if val_folder is provided
    if val_folder:
        data_transforms[val_folder] = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
        ])

    # Apply transformations for test data if test_folder is provided
    if test_folder:
        transform_list = [transforms.Resize(size=(image_size, image_size))]

        if transformation_type == 'color':
            transform_list.append(transforms.Lambda(lambda img: adjust_color(np.array(img), hue_shift=hue_shift)))
        elif transformation_type == 'noise':
            transform_list.append(transforms.Lambda(lambda img: add_gaussian_noise(np.array(img), mean=0, var=variance)))
        elif transformation_type == 'lighting':
            transform_list.append(transforms.Lambda(lambda img: adjust_lighting(np.array(img), alpha=1.3)))
        elif transformation_type == 'grayscale':
            transform_list.append(transforms.Lambda(lambda img: convert_to_grayscale(np.array(img))))

        transform_list.append(transforms.ToTensor())

        data_transforms[test_folder] = transforms.Compose(transform_list)

    return data_transforms


def reg_dataloader_KFold(full_dataset, train_idx, val_idx,
                         batch_size, is_gist=False, is_saliency=False,
                         isarousal=False, image_size=224):
    """
    Create DataLoaders for training and validation sets for KFold cross-validation.

    Args:
        full_dataset: Full dataset object
        train_idx (list of int): Indices for the training set.
        val_idx (list of int): Indices for the validation set.
        batch_size (int): Batch size for DataLoader.
        is_gist (bool, optional): Flag to determine if GIST transform should be applied.
        is_saliency (bool, optional): Flag to determine if Saliency transform should be applied.
        isarousal (bool): Whether to use arousal as label
        image_size (int): Target image size
        
    Returns:
        dict: Dictionary containing training and validation DataLoaders.
        dict: Dictionary containing sizes of training and validation sets.
    """
    # Apply the transformations to the training and validation subsets
    data_transforms = data_transform('train_folder', 'val_folder', image_size=image_size)

    train_dataset = SubsetWithTransform(full_dataset, train_idx, transform=data_transforms['train_folder'])
    val_dataset = SubsetWithTransform(full_dataset, val_idx, transform=data_transforms['val_folder'])

    # Create DataLoaders for the training and validation subsets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_with_names)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_with_names)

    return {'train': train_loader, 'val': val_loader}, {'train': len(train_dataset), 'val': len(val_dataset)}


def reg_dataloader_testdata(full_dataset, batch_size, is_gist=False, is_saliency=False, isarousal=False, image_size=224,
                            transformation_type=None, variance=10, hue_shift=0, alpha=0.5):
    """
    Create DataLoader for test data.

    Args:
        full_dataset: Full dataset object
        batch_size (int): Batch size for DataLoader
        is_gist (bool): Flag for GIST features
        is_saliency (bool): Flag for saliency features
        isarousal (bool): Whether to use arousal as label
        image_size (int): Target image size
        transformation_type (str): Type of transformation to apply
        variance (int): Variance for noise transformation
        hue_shift (int): Hue shift for color transformation
        alpha (float): Alpha for blending transformations
        
    Returns:
        dict: Dictionary containing test DataLoader.
        dict: Dictionary containing size of test set.
    """
    data_transforms = data_transform(test_folder='test_folder', image_size=image_size,
                                     transformation_type=transformation_type,
                                     variance=variance, hue_shift=hue_shift, alpha=alpha)

    test_dataset = SubsetWithTransform(full_dataset, range(len(full_dataset)), transform=data_transforms['test_folder'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_with_names)

    return {'test': test_loader}, {'test': len(test_dataset)}

