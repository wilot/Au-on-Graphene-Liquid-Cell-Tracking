"""training_data_synthesiser_gauss.py

A module for synthesising trianing data for a U-Net operating on atomic-resolution HAADF STEM images of adatoms 
supperted on a sheet of 2D material. Atoms are modelled as 2D gaussian blobs with HAADF contrast.

TODO: Running out of RAM (>250Gb) -> integrate Dask : Works but each stack needs to be loaded into memory 
(sequentially now)
TODO: Add functionality to generate vector-field + heatmaps for labels instead of just bool disks
        - Extract image generation to its own method.

"""

from typing import Dict, Union, Callable, Tuple, List
import multiprocessing
from pathlib import Path

import toml

import dask.array as da
import dask_ml.model_selection
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from sklearn.model_selection import train_test_split
import h5py

import atomai
import ase
import ase.build
from ase.visualize import view as ase_view


CONFIG_FILENAME = "synthesisers/gauss_synthesiser_config.toml"

class TrainingDataFactory:
    """Generates training-image and ground-truth patches for a randomly generated synthetic crystal by modelling atoms
    as 2D gaussians with HAADF STEM contrast.

    Runs a virtual crystal generator which synthesises a substrate sheet. This sheet has a random orientation and
    will have adatoms and holes randomly added. This is then used to create a training image and ground-truth image
    where the atoms are represented as 2D Gaussian blobs with HAADF STEM contrast. The labels are boolean disks. 
    This large, and noise-free, training image and ground-truth image is chopped into patches of a specified size
    and applies noise to the patches.

    Most of the configuration variables are set in a corresponding TOML file.
    """

    def __init__(self, sheet_constructor: Callable[[int, int], ase.Atoms],
                 config_file: Path) -> None:
        """Loads configurations for the TrainingDataFactoryTraining.

        Parameters
        ----------
        sheet_constructor : Callable[[int, int], ase.Atoms]
            A function which can synthesise the 2D substrate sheet to which holes and adatoms will be applied.
        config_file: Path
            The Path of the TML file containing the configuration parameters
        """

        config_dict = toml.load(config_file)
        sheet_config = config_dict["sheet_construction"]
        image_config = config_dict["image_construction"]
        patcher_config = config_dict["patcher_and_scrambler"]
        
        # For the defective sheet construction
        self.substrate_sheet_constructor: Callable[[int, int], ase.Atoms] = sheet_constructor
        
        self.sheet_size: Union[int, Tuple[int, int]] = sheet_config["sheet_size"]
        self.adatom_species: Union[str, List[str]] = sheet_config["adatom_species"]
        self.sheet_holes: int = sheet_config["sheet_holes"]
        self.sheet_adatoms: int = sheet_config["sheet_adatoms"]

        # For the synthesis of training and ground truth images from the sheet
        self.pixel_scale: float = image_config["pixel_scale"]
        self.atom_radius: float = image_config["atom_radius"]
        self.atom_species_order: List[str] = image_config["atom_species_order"]
        self.atom_brightnesses: Union[Dict[str, float], False] = image_config["atom_brightnesses"]
        self.HAADF_zcontrast: float = image_config["haadf_zcontrast_exponent"]

        # For the extraction of patches and noising them
        self.training_patch_size: Union[int, Tuple[int, int]] = patcher_config["training_patch_size"]
        self.zoom: int = patcher_config["zoom"]
        self.poisson: Union[Tuple[float, float], False] = patcher_config["poisson"]
        self.gauss: Union[Tuple[float, float], False] = patcher_config["gauss"]
        self.salt_and_pepper: Union[Tuple[float, float], False] = patcher_config["salt_and_pepper"]
        self.background: bool = patcher_config["background"]
        self.contrast: Union[Tuple[float, float], False] = patcher_config["contrast"]
        self.blur: bool = patcher_config["blur"]
    
    def sheet_constructor(self) -> ase.Atoms:
        """Constructs a defective sheet of 2D material.

        Constructs a monolayer sheet as an ase.Atoms object. Adds adatoms to the surface of the sheet and holes
        to the layer too. Then applies a random rotation.

        Note: The constructor `sheet_constructor` should return a sheet whose cell matrix has the 0th and 1st cell 
        indices nonzero and the 2nd should be zero (since it is 2D). Having the 1st matrix element/column set to zero 
        will cause error!

        Returns
        -------
        ase.Atoms:
            The virtual defective atom sheet
        """

        if isinstance(self.sheet_size, int):
            self.sheet_size = (self.sheet_size, self.sheet_size)

        sheet = self.substrate_sheet_constructor(self.sheet_size[0], self.sheet_size[1])

        def cut_cell(cell: ase.cell.Cell):
            new_cell = np.array(((cell[0, 0], 0, 0),
                                 (0, cell[1, 1], 0),
                                 (0, 0, cell[2, 2])))
            return ase.cell.Cell(new_cell)

        sheet.cell = cut_cell(sheet.cell)
        sheet.wrap()

        if not isinstance(self.adatom_species, list):
            self.adatom_species = [self.adatom_species,]
        for adatom_name in self.adatom_species:
            for _ in range(self.sheet_adatoms):
                position = list(np.random.rand(2))
                position.append(sheet.cell.lengths()[2]-0.5)  # Fixed at Å in Z
                position[0] = position[0] * sheet.cell.lengths()[0] -1
                position[1] = position[1] * sheet.cell.lengths()[1] -1
                atom = ase.Atom(adatom_name, position)
                sheet += atom
        sheet.wrap()
        # Make holes
        max_hole_radius = 120 # angstroms
        min_hole_radius = 30 # angstroms
        atom_in_hole = lambda atom, hole_centre, radius: \
            np.linalg.norm(np.array((atom.position[0], atom.position[1])) - hole_centre) < radius / 10
        deletion_indices = []
        for _ in range(self.sheet_holes):
            xpos_centre = np.random.randint(0, sheet.cell[0, 0])
            ypos_centre = np.random.randint(0, sheet.cell[1, 1])
            hole_centre = np.array((xpos_centre, ypos_centre))
            radius = np.random.random() * (max_hole_radius) + min_hole_radius  # in Å
            indices_within_hole = [atom.index for atom in sheet if atom_in_hole(atom, hole_centre, radius)]
            deletion_indices.extend(indices_within_hole)

        del sheet[deletion_indices]

        random_rotation_angle = np.random.random() * 90.
        sheet.rotate(random_rotation_angle, 'z', center='COP')
        
        return sheet

    def image_constructor(self, sheet: ase.Atoms, verbose=False, 
                          show_full_training_image=False) -> Tuple[np.ndarray, dict, np.ndarray]:
        """Synthesises training data patches from a summation of atom-gaussians.
        
        Parameters are in angstroms

        Parameters
        ----------
        sheet : ase.atoms
            A defectove sheet of atoms
        Returns
        -------
        Tuple[np.ndarray, dict, np.ndarray] :
            A tuple containing the training image, a dict of {species name: mask} for 
            each atom species seperately and a ground truth containing all atom species
        """

        template_size = int(self.atom_radius / self.pixel_scale * 10)  # Size of the atom templates in atom radii

        cell_lengths = sheet.cell.lengths()
        normal_axis = np.argmin(cell_lengths)
        basal_axes = np.indices(cell_lengths.shape)
        basal_axes = basal_axes[basal_axes != normal_axis]
        training_image_size = np.array([cell_lengths[ax] for ax in basal_axes]) / self.pixel_scale  # in the plane
        training_image_size = training_image_size.round().astype(int)
        if verbose:
            print("Training image size:", training_image_size)

        atom_species = list(set(sheet.get_chemical_symbols()))

        if set(atom_species) != set(self.atom_species_order):
            error_message = "There is a mismatch between the atomic species found "
            error_message += "in the atoms object and those supplied in "
            error_message += "the ordered_atom_species parameter."
            error_message += f"\nElements {atom_species} were found but "
            error_message += "{ordered_atom_species} was provided."
            raise ValueError(error_message)
            # atom_species = ordered_atom_species

        atom_numbers = []
        for species in atom_species:
            atom_numbers.append([atom.number for atom in sheet if atom.symbol==species][0])

        # Make atom templates
        xs, ys = np.meshgrid(np.arange(-template_size/2, template_size/2), np.arange(-template_size/2, template_size/2))
        def gauss_2d(x=0, y=0, x0=0, y0=0, A=1.):
            sx = sy = self.atom_radius / self.pixel_scale
            return A / (2. * np.pi * sx * sy) * np.exp(-((x - x0)**2. / (2. * sx**2.) + (y - y0)**2. / (2. * sy**2.)))

        if self.atom_brightnesses:  # Custom brightnesses
            atom_templates = [gauss_2d(xs, ys, A=self.atom_brightnesses[species]) 
                              for species in self.atom_brightnesses]
        else:
            atom_templates = [gauss_2d(xs, ys, A=atomic_number**self.HAADF_zcontrast) 
                              for atomic_number in atom_numbers]
        atom_masks = [atom_template > atom_template.max()/2 
                      for atom_template in atom_templates]  # FWHM

        # Construct image and training labels
        def get_px_positions(atoms, symbol, image_size):
            scaled_positions = atoms.cell.scaled_positions(
                    np.array([atom.position for atom in atoms 
                              if atom.symbol==symbol])
                    )
            scaled_positions = np.delete(scaled_positions, normal_axis, axis=1)  # Remove the coordinates in the axis normal to the basal plane
            outside_cell = np.any(np.logical_or(scaled_positions < 0, scaled_positions > 1), axis=1)  # Remove atoms outside the cell
            scaled_positions = scaled_positions[np.logical_not(outside_cell)]  # Remove atoms outside the cell
            scaled_positions *= image_size
            return scaled_positions.astype(int)

        atom_px_positions = [get_px_positions(sheet, species, training_image_size) 
                             for species in atom_species]
        position_images = [np.zeros(training_image_size, dtype=float) 
                           for _ in atom_species]
        for atom_positions, position_image in zip(atom_px_positions, position_images):
            for atom_position in atom_positions:
                position_image[atom_position[0], atom_position[1]] = 1.
                
        training_image = np.zeros(training_image_size)
        training_masks = [np.zeros(training_image_size) for _ in atom_masks]
        for position_image, atom_template in zip(position_images, atom_templates):
            training_image += convolve(position_image, atom_template)
        for position_image, atom_mask, training_mask in zip(position_images, atom_masks, training_masks):
            training_mask += np.clip(convolve(position_image, atom_mask), 0, 1)
            
        ground_truth = np.zeros(training_image_size)
        for mask_number, training_mask in enumerate(training_masks, 1):
            ground_truth[training_mask > 0] = mask_number

        if show_full_training_image:
            plt.imshow(training_image)
            plt.show(block=True)

        if verbose:
            fig, axes = plt.subplots(1, len(training_masks)+2, sharex=True, sharey=True)
            x_start, y_start, x_stop, y_stop = 0, 0, 512, 512
            axes[0].imshow(np.sqrt(training_image[x_start:x_stop, y_start:y_stop]))
            for index, mask in enumerate(training_masks):
                axes[index+1].imshow(mask[x_start:x_stop, y_start:y_stop])
                axes[index+1].set_title(atom_species[index] + " atom labels")
            axes[-1].imshow(ground_truth[x_start:x_stop, y_start:y_stop])
            axes[-1].set_title("Ground truth")
            fig.suptitle("Training image and ground-truth")
            plt.show(block=True)
        
        training_dict = {atom_species: training_mask for atom_species, training_mask 
                         in zip(atom_species, training_masks)}
        return training_image, training_dict, ground_truth

    def patcher_and_scrambler(self, training_image: np.ndarray, 
                              ground_truth: np.ndarray, show=False) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts patches from the training image and applies noise
        
        Extracts patches from the training image and corresponding ground truth mask(s), applying noise to the training
        image.

        Parameters
        ----------
        training_image : np.ndarray
            The training image to be processed, should be 2D and of same size as the 
            individual mask(s)
        ground_truth : np.ndarray
            The ground truth masks of the training image. Should be either 2D (for n-class training) with n possible 
            pixel values (and zero) or 3D with the last dimension corresponding to the class and all values 0 or 1. 
        training_patch_size : Iterable[int, int]
            The size of the extracted training patches in pixels
        training_patches : Int
            The number of patches to extract
        """

        if isinstance(self.training_patch_size, int):
            self.training_patch_size = (self.training_patch_size, 
                                        self.training_patch_size)

        if any([self.training_patch_size[ax] > training_image.shape[ax] for ax in (0, 1)]):
            error_msg = f"WARNING: Synthesised training image is smaller than the size of the desired patches. \n"
            error_msg += f"{self.training_patch_size=}\n"
            error_msg += f"{training_image.shape=}"
            raise ValueError(error_msg)
        
        training_patch_area = self.training_patch_size[0] * self.training_patch_size[1]
        training_image_area = training_image.shape[0] * training_image.shape[1]
        training_patches = training_image_area // training_patch_area 
        
        training_image, ground_truth = training_image[None, ...], ground_truth[None, ...]
        image_patches, gt_patches = atomai.utils.extract_patches(training_image, 
                                    ground_truth, self.training_patch_size, 
                                    training_patches*2)

        # Add noise to multiclass training data
        ch = len(np.unique(ground_truth)) -1 if len(ground_truth.shape) == 2 \
                                             else ground_truth.shape[-1]
        imaug = atomai.transforms.datatransform(
            n_channels=ch, dim_order_in='channel_last', dim_order_out='channel_first', 
            gauss_noise=self.gauss, poisson_noise=self.poisson, 
            salt_and_pepper=self.salt_and_pepper, zoom=self.zoom, rotation=True, 
            background=self.background, blur=self.blur, contrast=self.contrast,
            squeeze_channels=False, seed=42
        )
        image_patches, gt_patches = imaug.run(image_patches, gt_patches)

        if show:
            num_figs = min(3, len(image_patches))
            show_patch_start = 0
            fig, axes = plt.subplots(2, num_figs)
            for col in range(num_figs):
                axes[0, col].imshow(image_patches[show_patch_start+col].squeeze())
                axes[1, col].imshow(gt_patches[show_patch_start+col].squeeze(), vmin=0, vmax=3)
            fig.suptitle("Training image and ground-truth")
            plt.show(block=True)
            
        return image_patches, gt_patches

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates training-image and ground-truth patches for a randomly generated synthetic crystal.

        Runs a virtual crystal generator which synthesises a substrate sheet. This sheet has a random orientation and
        will have adatoms and holes randomly added. This is then used to create a training image and ground-truth image
        where the atoms are represented as 2D Gaussian blobs with HAADF STEM contrast. The labels are boolean disks. 
        This large, and noise-free, training image and ground-truth image is chopped into patches of a specified size
        and applies noise to the patches."""

        defective_sheet = self.sheet_constructor()
        training_image, ground_truth_dict, ground_truth_flat = self.image_constructor(defective_sheet)
        ground_truth = np.array([ground_truth_dict[species] for species in self.atom_species_order])
        ground_truth = np.moveaxis(ground_truth, 0, -1)  # Channel last for patcher_and_scrambler
        training_images, ground_truths = self.patcher_and_scrambler(training_image, ground_truth)
        ground_truths = np.moveaxis(ground_truths, -1, 1)  # Channel first for pyTorch

        return training_images.astype(np.float32), ground_truths.astype(np.float32)


def graphene_sheet_constructor(size_a: int, size_b: int) -> ase.Atoms:
    """Generates a graphene nanosheet approx. `size_a x size_b` unit cells in size."""

    sheet = ase.build.graphene_nanoribbon(size_a, size_b, sheet=True, vacuum=3.)
    sheet.rotate(90, '-x', rotate_cell=True)
    sheet.cell = ase.cell.Cell(np.array((sheet.cell[0], sheet.cell[2], sheet.cell[1])))
    return sheet


def plot_sample(job: Callable):
    """Plots a sample of the output from a single job/iteration"""

    training_patches, ground_truth_patches = job()
    num_samples = 5
    fig, axes = plt.subplots(3, num_samples, sharex=True, sharey=True)
    for ax_col, training_patch, gt_patch in zip(axes.T, training_patches, ground_truth_patches):
        ax_col[0].imshow(training_patch[0]**0.5)
        ax_col[1].imshow(gt_patch[0])
        ax_col[2].imshow(gt_patch[1])
    axes[0, 0].set_ylabel("Training image ^0.5")
    axes[1, 0].set_ylabel("Ground Truth Lattice")
    axes[2, 0].set_ylabel("Ground Truth Adatom")
    fig.suptitle(f"Training data sample ({num_samples} of training_patches.shape[0] * #iterations)")
    for ax in axes.flatten(): 
        ax.set_axis_off()
    plt.show(block=True)


if __name__ == "__main__":
   
    TESTING_RUN = False
    config_filename = Path(CONFIG_FILENAME)
    iterations, cores = 200, 40  # DANGER!!! Be reasonable, don't melt your PC! No point < 3 iterations per core

    def job():
        data_factory = TrainingDataFactory(graphene_sheet_constructor, config_filename)
        training_patches, ground_truth_patches = data_factory.run()
        return training_patches, ground_truth_patches

    if TESTING_RUN:
        plot_sample(job)
        exit(0)

    with multiprocessing.Pool(cores) as pool:
        print(f"Beginning synthesis, launching {iterations} synthesisers on {cores} cores.")
        future_results = [pool.apply_async(job) for _ in range(iterations)]
        results = [future_result.get() for future_result in future_results]
        print("Synthesis complete, processes rejoined.")

    image_patch_stacks = [result[0] for result in results]
    gt_patch_stacks = [result[1] for result in results]

    print("Saved patches to lists")

    dataset_chunking = (1_000, -1, -1, -1)
    
    image_patches = np.array(image_patch_stacks)
    image_patches = image_patches.reshape(-1, *image_patches.shape[2:])
    print(f"Images saved to numpy array of shape {image_patches.shape}")
    image_patches = da.from_array(image_patches, chunks=dataset_chunking)  # Chunks ~1GB for 512x512 float
    # image_patches = da.moveaxis(image_patches, 1, -1)
    print(f"{image_patches.chunks=}")
    print(f"{image_patches.shape=}")
    print(f"{image_patches.dtype=}")

    gt_patches = np.array(gt_patch_stacks)
    gt_patches = gt_patches.reshape(-1, *gt_patches.shape[2:])
    print(f"Ground-truth masks saved to numpy array of shape {gt_patches.shape}")
    gt_patches = da.from_array(gt_patches, chunks=dataset_chunking)  # Chunks 
    print(f"{gt_patches.chunks=}")
    print(f"{gt_patches.shape=}")
    print(f"{gt_patches.dtype=}")

    # image_patches_train, images_patches_test, gt_patches_train, gt_patches_test = train_test_split(image_patches, 
    #         gt_patches, test_size=0.25, random_state=42)
    # np.savez('au_on_graphene_gaussian_training_data_large.npz', X_train=image_patches_train, 
    #         X_test=images_patches_test, y_train=gt_patches_train, y_test=gt_patches_test)

    images_patches_train, images_patches_test, gt_patches_train, gt_patches_test = \
        dask_ml.model_selection.train_test_split(image_patches, gt_patches, test_size=0.25, random_state=42)

    save_file = h5py.File('synthesisers/au_on_graphene_scrambled_gaussian_training_data_large.hdf5', 'x')

    for stack, stack_dir_name in zip((images_patches_train, images_patches_test, gt_patches_train, gt_patches_test),
                                     ('/images_train', '/images_test', '/labels_train', '/labels_test')):
        in_memory_stack = stack.compute()  # Loads the whole thing into memory. Needed to know the shape for some reason...
        chunk_shape = (min(dataset_chunking[0], in_memory_stack.shape[0]), *in_memory_stack.shape[1:])
        dask_stack = da.from_array(in_memory_stack, chunks=chunk_shape)
        save_dset = save_file.create_dataset(stack_dir_name, in_memory_stack.shape, np.float32, chunks=chunk_shape)
        da.store(dask_stack, save_dset)

    print("Program complete.")
