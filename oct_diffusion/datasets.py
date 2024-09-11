import torch
import nibabel as nib
from glob import glob
from torch.utils.data import Dataset
# from oct_diffusion.utils import


class PatchDirectoryDataset(Dataset):
    """
    Dataset for .pt label volumes.

    patch_directory : str
        Directory containing patches of data in .pt file format
    """

    def __init__(self,
                 patch_directory: str = None,
                 transform: str = None):
        """
        Dataset for .pt label volumes.

        patch_directory : str
            Directory containing patches of data in .pt file format
        """
        self.patch_directory = patch_directory
        # Get the sorted list of patch paths
        self.patches_list = sorted(glob(f"{self.patch_directory}/*"))[:399]
        # Set transform attribute
        self.transform = transform

    def __len__(self):
        """
        Returns length of patches dataset.

        Returns
        -------
        length : int
            Length of patches dataset.
        """
        return len(self.patches_list)

    def __getitem__(self, idx):
        """
        Get patch and transformed patch.

        Returns
        -------
        x0 : torch.Tensor
            Clean patch.
        zt : torch.Tensor
            Noisy tensor at random timstep in markov chain t.
        """
        # Loading patch on gpu
        x0 = torch.load(self.patches_list[idx]).cuda()
        # Sampling noisy patch at random timestep in markov chain
        zt = self.transform(x0)
        return x0, zt


class Patcher(Dataset):
    """
    A PyTorch dataset for handling patchwise operations on large volumes.

    Parameters
    ----------
    parent_path : str
        Path to volumetric parent tensor from which patches are extracted.
        Shape (D, H, W)
    patch_shape : list[float]
        Spatial dimensions of patch (D, H, W)
    """
    def __init__(self, parent_path, patch_shape: list = [64, 64, 64],
                 redundancy: float = 3):
        super(Patcher, self).__init__()
        self.parent_path = parent_path
        self.nifti = nib.load(self.parent_path)
        self.shape = self.nifti.shape
        self.patch_shape = torch.Tensor(patch_shape)
        self.redundancy = torch.Tensor([redundancy - 1] * 3)
        self.step_shape = self.patch_shape * (1 / (2 ** self.redundancy))

        self.parent_tensor = torch.from_numpy(self.nifti.get_fdata()).cuda()
        self.pad_volume_()
        self.shape = self.parent_tensor.shape
        self.calculate_patch_origins()

    def __len__(self):
        """
        Return the total number of patches.

        Returns
        -------
        len : int
            Total number of patches (length of dataset)
        """
        return len(self.complete_patch_origins)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve a patch by index.

        Parameters
        ----------
        idx (int): The index of the patch to retrieve.

        Returns
        -------
        patch : torch.Tensor
            Subpatch extracted from parent volume.
        coords_list : tuple
            A tuple containing the slice indices.
        """
        x_start, y_start, z_start = self.complete_patch_origins[idx]
        print(x_start)
        x_slice = slice(x_start, x_start + int(self.patch_shape[0].item()))
        y_slice = slice(y_start, y_start + int(self.patch_shape[1].item()))
        z_slice = slice(z_start, z_start + int(self.patch_shape[2].item()))
        patch = self.parent_tensor[x_slice, y_slice, z_slice].detach().cuda()
        return patch, (x_slice, y_slice, z_slice)

    def pad_volume_(self) -> torch.Tensor:
        """
        Applies padding to a tensor to increase its dimensions.
        """
        # Interleave pad shape (D, D, H, H, W, W)
        pad_shape_tuple_int = torch.repeat_interleave(
            torch.clone(self.patch_shape), repeats=2)
        # Convert tensor to integer list
        pad_shape_tuple_int = pad_shape_tuple_int.tolist()
        pad_shape_tuple_int = [int(ele) for ele in pad_shape_tuple_int]
        # Execute padding operation and squeeze to (D, H, W)
        self.parent_tensor = torch.nn.functional.pad(
            input=self.parent_tensor.unsqueeze(0),
            pad=pad_shape_tuple_int,
            mode='reflect'
            ).squeeze()

    def calculate_patch_origins(self):
        """
        Compute the coordinates for slicing the tensor into patches based on
        the defined patch size and step size. Creates complete_patch_origins
        attribute with slice objects.
        """
        self.complete_patch_origins = []
        x_coords = torch.arange(
            0 + self.step_shape[0],
            self.shape[0] - self.patch_shape[0],
            self.step_shape[0],
            dtype=torch.int32)
        y_coords = torch.arange(
            0 + self.step_shape[0],
            self.shape[1] - self.patch_shape[1],
            self.step_shape[1],
            dtype=torch.int32)
        z_coords = torch.arange(
            0 + self.step_shape[0],
            self.shape[2] - self.patch_shape[2],
            self.step_shape[2],
            dtype=torch.int32)
        self.complete_patch_origins = torch.cartesian_prod(
            x_coords, y_coords, z_coords).int()
