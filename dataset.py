import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os
import copy
from skimage import exposure
from scipy import io 
import json


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, data_type='float32', transform=None, sgm=25, ratio=0.9, size_data=(128, 128, 3), size_window=(5, 5), norm_method='minmax'):
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type
        self.sgm = sgm

        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window
        self.norm_method = norm_method

        # Initialize global stats
        self.global_mean = None
        self.global_std = None
        self.global_min = None
        self.global_max = None

        lst_data = os.listdir(data_dir)

        # lst_input = [f for f in lst_data if f.startswith('input')]
        # lst_label = [f for f in lst_data if f.startswith('label')]
        #
        # lst_input.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # lst_label.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        #
        # self.lst_input = lst_input
        # self.lst_label = lst_label

        lst_data.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        # lst_data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.lst_data = lst_data
        
        # Calculate global stats if needed
        if norm_method in ['global-minmax', 'z-score', 'adaptive']:
            self._calculate_global_stats()

    

    def normalize_image(self, img):
        """Apply various normalization methods to the image"""
        if self.norm_method == 'minmax':
            # Simple min-max normalization
            min_val, max_val = img.min(), img.max()
            if max_val > min_val:
                return (img - min_val) / (max_val - min_val)
            return np.zeros_like(img)
            
        elif self.norm_method == 'global-minmax' and self.global_min is not None:
            # Global min-max normalization using dataset stats
            return np.clip((img - self.global_min) / (self.global_max - self.global_min), 0, 1)
            
        elif self.norm_method == 'z-score' and self.global_mean is not None:
            # Z-score normalization using dataset stats
            z = (img - self.global_mean) / (self.global_std + 1e-6)
            return np.clip((z + 3) / 6, 0, 1)  # Rescale to [0,1] range
            
        elif self.norm_method == 'adaptive':
            # First normalize to [0,1]
            if self.global_min is not None:
                norm = np.clip((img - self.global_min) / (self.global_max - self.global_min), 0, 1)
            else:
                min_val, max_val = img.min(), img.max()
                if max_val > min_val:
                    norm = (img - min_val) / (max_val - min_val)
                else:
                    norm = np.zeros_like(img)
            return norm
            
            # # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # try:
            #     return exposure.equalize_adapthist(norm, clip_limit=0.03)
            # except:
            #     return norm
                
        # Default: return image as is
        # return img

    def __getitem__(self, index):
        file_path = os.path.join(self.data_dir, self.lst_data[index])
        
        # Check file type and load accordingly
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.mat'):
            try:
                mat_contents = io.loadmat(file_path)
                if 'noisy_image' in mat_contents:
                    data = mat_contents['noisy_image']
                elif 'clean_image' in mat_contents:
                    data = mat_contents['clean_image']
                data = data.astype(np.float32)
            except Exception as e:
                print(f"Error loading .mat file {file_path}: {e}")
                # Return zeros as fallback
                data = np.zeros(self.size_data, dtype=np.float32)
        else:
            data = plt.imread(file_path)
            if data.dtype == np.uint8:
                data = data / 255.0
      
        # Ensure data has correct shape
        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        # Apply normalization
        data = self.normalize_image(data)

        # For Noise2Void with CT data, we use the noisy data as is
        # No additional noise is added since our data is already noisy
        label = data  # Original noisy data
        input, mask = self.generate_mask(copy.deepcopy(label))

        data = {'label': label, 'input': input, 'mask': mask}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_data)

    def generate_mask(self, input):
        ratio = self.ratio
        size_window = self.size_window
        size_data = self.size_data
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask

    def _calculate_global_stats(self):
        """Calculate global dataset statistics for better normalization using the entire dataset"""
        print("Calculating global dataset statistics on ENTIRE dataset...")
        
        # Process ALL images, not just a sample
        all_values = []
        min_vals = []
        max_vals = []
        
        total_files = len(self.lst_data)
        for idx, file_name in enumerate(self.lst_data):
            try:
                file_path = os.path.join(self.data_dir, file_name)
                
                if file_path.endswith('.npy'):
                    img = np.load(file_path)
                elif file_path.endswith('.mat'):
                    # Handle .mat files
                    mat_contents = io.loadmat(file_path)
                    if 'noisy_image' in mat_contents:
                        img = mat_contents['noisy_image']
                    elif 'clean_image' in mat_contents:
                        img = mat_contents['clean_image']
                else:
                    img = plt.imread(file_path)
                    if img.dtype == np.uint8:
                        img = img / 255.0
                
                all_values.append(img.ravel())
                min_vals.append(img.min())
                max_vals.append(img.max())
                
                # Print progress
                if (idx + 1) % 10 == 0 or idx == total_files - 1:
                    print(f"Processed {idx + 1}/{total_files} files ({(idx + 1) / total_files * 100:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if all_values:
            # Concatenate all values
            all_values = np.concatenate(all_values)
            
            # Calculate global statistics
            self.global_mean = float(np.mean(all_values))
            self.global_std = float(np.std(all_values))
            self.global_min = float(min(min_vals))
            self.global_max = float(max(max_vals))
            
            print(f"Global stats (full dataset): mean={self.global_mean:.4f}, std={self.global_std:.4f}, "
                  f"min={self.global_min:.4f}, max={self.global_max:.4f}")
            
            # Save stats 
            try:
                stats = {
                    'mean': self.global_mean,
                    'std': self.global_std,
                    'min': self.global_min,
                    'max': self.global_max
                }
                with open(stats_file, 'w') as f:
                    json.dump(stats, f)
                print(f"Saved global stats to {stats_file}")
            except Exception as e:
                print(f"Error saving stats to file: {e}")

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        input, label, mask = data['input'], data['label'], data['mask']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label), 'mask': torch.from_numpy(mask)}


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label, mask = data['input'], data['label'], data['mask']

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std

        data = {'input': input, 'label': label, 'mask': mask}
        return data


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        input, label, mask = data['input'], data['label'], data['mask']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)
            mask = np.fliplr(mask)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)
            mask = np.flipud(mask)

        return {'input': input, 'label': label, 'mask': mask}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label = data['input'], data['label']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))

    return {'input': input, 'label': label}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
    id_x = np.arange(left, left + new_w, 1).astype(np.int32)

    # input = input[top: top + new_h, left: left + new_w]
    # label = label[top: top + new_h, left: left + new_w]

    input = input[id_y, id_x]
    label = label[id_y, id_x]
    mask = mask[id_y, id_x]

    return {'input': input, 'label': label, 'mask': mask}


class UnifromSample(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, stride):
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
      self.stride = (stride, stride)
    else:
      assert len(stride) == 2
      self.stride = stride

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    stride_h, stride_w = self.stride
    new_h = h//stride_h
    new_w = w//stride_w

    top = np.random.randint(0, stride_h + (h - new_h * stride_h))
    left = np.random.randint(0, stride_w + (w - new_w * stride_w))

    id_h = np.arange(top, h, stride_h)[:, np.newaxis]
    id_w = np.arange(left, w, stride_w)

    input = input[id_h, id_w]
    label = label[id_h, id_w]
    mask = mask[id_h, id_w]

    return {'input': input, 'label': label, 'mask': mask}


class ZeroPad(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    l = (new_w - w)//2
    r = (new_w - w) - l

    u = (new_h - h)//2
    b = (new_h - h) - u

    input = np.pad(input, pad_width=((u, b), (l, r), (0, 0)))
    label = np.pad(label, pad_width=((u, b), (l, r), (0, 0)))
    mask = np.pad(mask, pad_width=((u, b), (l, r), (0, 0)))

    return {'input': input, 'label': label, 'mask': mask}

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
