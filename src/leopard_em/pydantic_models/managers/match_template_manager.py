"""Root-level model for serialization and validation of 2DTM parameters."""

import os
from typing import Annotated, Any, ClassVar, Iterator, Literal, Optional

import mrcfile
import pandas as pd
import torch
from pydantic import ConfigDict, Field, field_validator

from leopard_em.analysis.zscore_metric import gaussian_noise_zscore_cutoff
from leopard_em.backend.core_match_template import core_match_template
from leopard_em.backend.core_match_template_distributed import (
    core_match_template_distributed,
)
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    DefocusSearchConfig,
    MultipleOrientationConfig,
    OrientationSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import (
    OpticsGroup,
    SearchWindow,
    iter_search_windows_from_table,
)
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.pydantic_models.results import MatchTemplateResult
from leopard_em.pydantic_models.utils import (
    calculate_ctf_filter_stack,
    preprocess_image,
    volume_to_rfft_fourier_slice,
)
from leopard_em.utils.data_io import load_mrc_image, load_mrc_volume


# pylint: disable=no-self-argument
class MatchTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running full orientation 2DTM.

    Attributes
    ----------
    micrograph_path : str
        Path to the micrograph .mrc file.
    template_volume_path : str
        Path to the template volume .mrc file.
    micrograph : ExcludedTensor
        Image to run template matching on. Not serialized.
    template_volume : ExcludedTensor
        Template volume to match against. Not serialized.
    optics_group : OpticsGroup
        Optics group parameters for the imaging system on the microscope.
    defocus_search_config : DefocusSearchConfig
        Parameters for searching over defocus values.
    orientation_search_config : OrientationSearchConfig
        Parameters for searching over orientation angles.
    preprocessing_filters : PreprocessingFilters
        Configurations for the preprocessing filters to apply during
        correlation.
    match_template_result : MatchTemplateResult
        Result of the match template program stored as an instance of the
        `MatchTemplateResult` class.
    computational_config : ComputationalConfig
        Parameters for controlling computational resources.
    search_windows : list[SearchWindow], optional
        Optional regions restricting where the match template search is
        evaluated.  When omitted, the full micrograph is searched.
    search_window_table_path : str, optional
        Path to a tabular file (CSV/Parquet/Feather) enumerating search windows.
        The file is streamed row-by-row, allowing very large coordinate sets to be
        processed without loading them entirely into memory.  Providing
        ``search_windows`` takes precedence over the table path if both are
        supplied.
    search_window_table_format : Literal["auto", "csv", "parquet", "feather"]
        Format override for ``search_window_table_path``.  When set to ``"auto"``
        (default) the format is inferred from the file suffix.
    search_window_chunk_size : int, optional
        Number of rows to load at once when streaming a CSV table.  Other formats
        are read eagerly.  Defaults to 1024 rows per chunk when unspecified.
    search_pixel_count : int
        Number of valid correlation pixels covered by the most recent search.

    Methods
    -------
    validate_micrograph_path(v: str) -> str
        Ensure the micrograph file exists.
    validate_template_volume_path(v: str) -> str
        Ensure the template volume file exists.
    __init__(preload_mrc_files: bool = False , **data: Any)
        Constructor which also loads the micrograph and template volume from disk.
        The 'preload_mrc_files' parameter controls whether to read the MRC files
        immediately upon initialization.
    make_backend_core_function_kwargs() -> dict[str, Any]
        Generates the keyword arguments for backend 'core_match_template' call from
        held parameters. Does the necessary pre-processing steps to filter the image
        and template.
    run_match_template(orientation_batch_size: int = 1, do_result_export: bool = True)
        Runs the base match template program in PyTorch.
    results_to_dataframe(
        half_template_width_pos_shift: bool = True,
        exclude_columns: Optional[list] = None,
        locate_peaks_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame
        Converts the basic extracted peak info DataFrame (from the result object) to a
        DataFrame with additional information about reference files, microscope
        parameters, etc.
    save_config(path: str, mode: Literal["yaml", "json"] = "yaml") -> None
        Save this Pydantic model config to disk.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized attributes
    micrograph_path: str
    template_volume_path: str
    optics_group: OpticsGroup
    defocus_search_config: DefocusSearchConfig
    orientation_search_config: OrientationSearchConfig | MultipleOrientationConfig
    preprocessing_filters: PreprocessingFilters
    match_template_result: MatchTemplateResult
    computational_config: ComputationalConfig
    search_windows: Optional[list[SearchWindow]] = None
    search_window_table_path: Optional[str] = None
    search_window_table_format: Literal["auto", "csv", "parquet", "feather"] = "auto"
    search_window_chunk_size: Annotated[int | None, Field(gt=0)] = None

    # Non-serialized large array-like attributes
    micrograph: ExcludedTensor
    template_volume: ExcludedTensor
    search_pixel_count: int = Field(default=0, exclude=True)

    ###########################
    ### Pydantic Validators ###
    ###########################

    @field_validator("micrograph_path")  # type: ignore
    def validate_micrograph_path(cls, v) -> str:
        """Ensure the micrograph file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for micrograph does not exist.")

        return str(v)

    @field_validator("template_volume_path")  # type: ignore
    def validate_template_volume_path(cls, v) -> str:
        """Ensure the template volume file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for template volume does not exist.")

        return str(v)

    @field_validator("search_window_table_path")  # type: ignore
    def validate_search_window_table_path(cls, v: Optional[str]) -> Optional[str]:
        """Ensure the optional search window table exists if provided."""

        if v is None:
            return None

        if not os.path.exists(v):
            raise ValueError(f"Search window table '{v}' does not exist.")

        return str(v)

    def __init__(self, preload_mrc_files: bool = False, **data: Any):
        super().__init__(**data)

        if preload_mrc_files:
            # Load the data from the MRC files
            self.micrograph = load_mrc_image(self.micrograph_path)
            self.template_volume = load_mrc_volume(self.template_volume_path)

    ############################################
    ### Functional (data processing) methods ###
    ############################################

    def _has_window_source(self) -> bool:
        """Return ``True`` when a constrained search source is configured."""

        return bool(self.search_windows) or self.search_window_table_path is not None

    def _iter_search_windows(self) -> Iterator[SearchWindow]:
        """Yield all search windows from either explicit lists or tables."""

        if self.search_windows:
            yield from self.search_windows
            return

        if self.search_window_table_path is not None:
            yield from iter_search_windows_from_table(
                self.search_window_table_path,
                file_format=self.search_window_table_format,
                chunk_size=self.search_window_chunk_size,
            )
            return

        raise ValueError("No search windows configured for the constrained search.")

    def _prepare_shared_core_inputs(self) -> dict[str, torch.Tensor]:
        """Ensure core tensors are loaded and shared across search windows."""

        if self.micrograph is None:
            self.micrograph = load_mrc_image(self.micrograph_path)
        if not isinstance(self.micrograph, torch.Tensor):
            self.micrograph = torch.from_numpy(self.micrograph)
        self.micrograph = self.micrograph.to(torch.float32)

        if self.template_volume is None:
            self.template_volume = load_mrc_volume(self.template_volume_path)
        if not isinstance(self.template_volume, torch.Tensor):
            self.template_volume = torch.from_numpy(self.template_volume)
        self.template_volume = self.template_volume.to(torch.float32)

        template_dft = volume_to_rfft_fourier_slice(self.template_volume)

        defocus_values = self.defocus_search_config.defocus_values.to(torch.float32)
        pixel_size_offsets = torch.tensor([0.0], dtype=torch.float32)
        ctf_filters = calculate_ctf_filter_stack(
            template_shape=(self.template_volume.shape[0], self.template_volume.shape[0]),
            optics_group=self.optics_group,
            defocus_offsets=defocus_values,
            pixel_size_offsets=pixel_size_offsets,
        )
        euler_angles = self.orientation_search_config.euler_angles.to(torch.float32)

        return {
            "image": self.micrograph,
            "template": self.template_volume,
            "template_dft": template_dft,
            "defocus_values": defocus_values,
            "pixel_size_offsets": pixel_size_offsets,
            "ctf_filters": ctf_filters,
            "euler_angles": euler_angles,
        }

    def _make_core_kwargs_for_patch(
        self,
        image_patch: torch.Tensor,
        template: torch.Tensor,
        template_dft: torch.Tensor,
        ctf_filters: torch.Tensor,
        euler_angles: torch.Tensor,
        defocus_values: torch.Tensor,
        pixel_size_offsets: torch.Tensor,
    ) -> dict[str, Any]:
        """Construct keyword arguments for a specific image patch."""

        image_dft = torch.fft.rfftn(image_patch)
        image_dft[0, 0] = 0 + 0j

        bp_config = self.preprocessing_filters.bandpass_filter
        bandpass_filter = bp_config.calculate_bandpass_filter(image_dft.shape)

        cumulative_filter_image = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft,
            output_shape=image_dft.shape,
        )
        cumulative_filter_template = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft,
            output_shape=(template.shape[-2], template.shape[-1] // 2 + 1),
        )

        image_preprocessed_dft = preprocess_image(
            image_rfft=image_dft,
            cumulative_fourier_filters=cumulative_filter_image,
            bandpass_filter=bandpass_filter,
        )

        return {
            "image_dft": image_preprocessed_dft,
            "template_dft": template_dft,
            "ctf_filters": ctf_filters,
            "whitening_filter_template": cumulative_filter_template,
            "euler_angles": euler_angles,
            "defocus_values": defocus_values,
            "pixel_values": pixel_size_offsets,
            "device": self.computational_config.gpu_devices,
        }

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Generates the keyword arguments for backend call from held parameters."""
        shared_inputs = self._prepare_shared_core_inputs()

        return self._make_core_kwargs_for_patch(
            image_patch=shared_inputs["image"],
            template=shared_inputs["template"],
            template_dft=shared_inputs["template_dft"],
            ctf_filters=shared_inputs["ctf_filters"],
            euler_angles=shared_inputs["euler_angles"],
            defocus_values=shared_inputs["defocus_values"],
            pixel_size_offsets=shared_inputs["pixel_size_offsets"],
        )

    # pylint: disable=too-many-locals
    def _run_match_template_with_windows(
        self,
        orientation_batch_size: int,
        do_result_export: bool,
        do_valid_cropping: bool,
    ) -> None:
        """Run match template within user-specified spatial windows."""

        if not self._has_window_source():
            raise ValueError(
                "No search windows configured. Provide 'search_windows' or "
                "'search_window_table_path' in the configuration."
            )

        shared_inputs = self._prepare_shared_core_inputs()
        image = shared_inputs["image"]
        template = shared_inputs["template"]
        template_size = int(template.shape[-1])

        image_height, image_width = int(image.shape[-2]), int(image.shape[-1])
        valid_height = image_height - template_size + 1
        valid_width = image_width - template_size + 1
        if valid_height <= 0 or valid_width <= 0:
            raise ValueError("Template dimensions exceed the micrograph size.")

        dtype = torch.float32
        mip_valid = torch.full((valid_height, valid_width), float("-inf"), dtype=dtype)
        scaled_mip_valid = torch.full((valid_height, valid_width), float("-inf"), dtype=dtype)
        corr_mean_valid = torch.zeros((valid_height, valid_width), dtype=dtype)
        corr_var_valid = torch.zeros((valid_height, valid_width), dtype=dtype)
        psi_valid = torch.zeros((valid_height, valid_width), dtype=dtype)
        theta_valid = torch.zeros((valid_height, valid_width), dtype=dtype)
        phi_valid = torch.zeros((valid_height, valid_width), dtype=dtype)
        defocus_valid = torch.zeros((valid_height, valid_width), dtype=dtype)
        search_mask = torch.zeros((valid_height, valid_width), dtype=torch.bool)

        total_projections = 0
        total_orientations = 0
        total_defocus = 0

        for window in self._iter_search_windows():
            valid_y_bounds, valid_x_bounds = window.get_valid_bounds(
                (image_height, image_width), template_size
            )
            start_valid_y, end_valid_y = valid_y_bounds
            start_valid_x, end_valid_x = valid_x_bounds

            start_img_y = start_valid_y
            end_img_y = end_valid_y + template_size - 1
            start_img_x = start_valid_x
            end_img_x = end_valid_x + template_size - 1

            image_patch = image[start_img_y:end_img_y, start_img_x:end_img_x].contiguous()

            core_kwargs = self._make_core_kwargs_for_patch(
                image_patch=image_patch,
                template=template,
                template_dft=shared_inputs["template_dft"],
                ctf_filters=shared_inputs["ctf_filters"],
                euler_angles=shared_inputs["euler_angles"],
                defocus_values=shared_inputs["defocus_values"],
                pixel_size_offsets=shared_inputs["pixel_size_offsets"],
            )

            results = core_match_template(
                **core_kwargs,
                orientation_batch_size=orientation_batch_size,
                num_cuda_streams=self.computational_config.num_cpus,
            )

            patch_height_valid = end_valid_y - start_valid_y
            patch_width_valid = end_valid_x - start_valid_x
            valid_slice = (slice(0, patch_height_valid), slice(0, patch_width_valid))

            mip_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["mip"][valid_slice].cpu()
            )
            scaled_mip_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["scaled_mip"][valid_slice].cpu()
            )
            corr_mean_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["correlation_mean"][valid_slice].cpu()
            )
            corr_var_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["correlation_variance"][valid_slice].cpu()
            )
            psi_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["best_psi"][valid_slice].cpu()
            )
            theta_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["best_theta"][valid_slice].cpu()
            )
            phi_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["best_phi"][valid_slice].cpu()
            )
            defocus_valid[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = (
                results["best_defocus"][valid_slice].cpu()
            )

            search_mask[start_valid_y:end_valid_y, start_valid_x:end_valid_x] = True

            total_projections = results["total_projections"]
            total_orientations = results["total_orientations"]
            total_defocus = results["total_defocus"]

        searched_pixels = int(search_mask.sum().item())
        if searched_pixels == 0:
            raise ValueError("Search windows did not cover any valid pixels.")

        self.search_pixel_count = searched_pixels

        prevalid_shape = (image_height, image_width)

        def _embed_valid_map(valid_map: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
            embedded = torch.full(prevalid_shape, fill_value, dtype=valid_map.dtype)
            embedded[:valid_height, :valid_width] = valid_map
            return embedded

        results_combined = {
            "mip": _embed_valid_map(mip_valid, fill_value=float("-inf")),
            "scaled_mip": _embed_valid_map(scaled_mip_valid, fill_value=float("-inf")),
            "correlation_mean": _embed_valid_map(corr_mean_valid, fill_value=0.0),
            "correlation_variance": _embed_valid_map(corr_var_valid, fill_value=0.0),
            "best_psi": _embed_valid_map(psi_valid, fill_value=0.0),
            "best_theta": _embed_valid_map(theta_valid, fill_value=0.0),
            "best_phi": _embed_valid_map(phi_valid, fill_value=0.0),
            "best_defocus": _embed_valid_map(defocus_valid, fill_value=0.0),
            "total_projections": total_projections,
            "total_orientations": total_orientations,
            "total_defocus": total_defocus,
        }

        self.match_template_result.match_template_peaks = None
        self._populate_match_template_result(
            results_combined,
            do_result_export=do_result_export,
            do_valid_cropping=do_valid_cropping,
        )

    def run_match_template(
        self,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        do_valid_cropping: bool = True,
    ) -> None:
        """Runs the base match template in pytorch.

        Parameters
        ----------
        orientation_batch_size : int
            The number of projections to process in a single batch. Default is 1.
        do_result_export : bool
            If True, call the `MatchTemplateResult.export_results` method to save the
            results to disk directly after running the match template. Default is True.
        do_valid_cropping : bool
            If True, apply the valid cropping mode to the results. Default is True.

        Returns
        -------
        None
        """
        if self._has_window_source():
            self._run_match_template_with_windows(
                orientation_batch_size=orientation_batch_size,
                do_result_export=do_result_export,
                do_valid_cropping=do_valid_cropping,
            )
            return

        self.search_pixel_count = 0
        core_kwargs = self.make_backend_core_function_kwargs()
        results = core_match_template(
            **core_kwargs,
            orientation_batch_size=orientation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
        )

        # Populate the MatchTemplateResult via a private helper
        self._populate_match_template_result(
            results,
            do_result_export=do_result_export,
            do_valid_cropping=do_valid_cropping,
        )
        self.search_pixel_count = int(self.match_template_result.mip.numel())

    def run_match_template_distributed(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        do_valid_cropping: bool = True,
    ) -> None:
        """Runs the base match template in a distributed, multi-node environment.

        Parameters
        ----------
        world_size : int
            The total number of processes in the distributed job.
        rank : int
            The global rank of this process.
        local_rank : int
            The local rank of this process (used to assign GPU).
        orientation_batch_size : int
            The number of projections to process in a single batch. Default is 1.
        do_result_export : bool
            If True, call the `MatchTemplateResult.export_results` method to save the
            results to disk directly after running the match template. Default is True.
        do_valid_cropping : bool
            If True, apply the valid cropping mode to the results. Default is True.

        Raises
        ------
        RuntimeError
            If the distributed process group has not been initialized.

        Returns
        -------
        None
        """
        if self._has_window_source():
            raise NotImplementedError(
                "Constrained search windows are not currently supported in distributed runs."
            )

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group has not been initialized! "
                "Cannot run distributed match template."
            )

        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            core_kwargs = self.make_backend_core_function_kwargs()
        else:
            core_kwargs = {}

        _ = core_kwargs.pop("device", None)

        results = core_match_template_distributed(
            world_size,
            rank,
            local_rank,
            device,
            orientation_batch_size,
            self.computational_config.num_cpus,
            **core_kwargs,
        )

        # Only populate the results on the first rank
        if torch.distributed.get_rank() == 0:
            self._populate_match_template_result(
                results,
                do_result_export=do_result_export,
                do_valid_cropping=do_valid_cropping,
            )

    def _populate_match_template_result(
        self,
        results: dict[str, Any],
        do_result_export: bool = True,
        do_valid_cropping: bool = True,
    ) -> None:
        """Helper function to populate the MatchTemplateResult object post-core call."""
        # Place results into the `MatchTemplateResult` object
        self.match_template_result.mip = results["mip"]
        self.match_template_result.scaled_mip = results["scaled_mip"]

        self.match_template_result.correlation_average = results["correlation_mean"]
        self.match_template_result.correlation_variance = results[
            "correlation_variance"
        ]
        self.match_template_result.orientation_psi = results["best_psi"]
        self.match_template_result.orientation_theta = results["best_theta"]
        self.match_template_result.orientation_phi = results["best_phi"]
        self.match_template_result.relative_defocus = results["best_defocus"]

        self.match_template_result.total_projections = results["total_projections"]
        self.match_template_result.total_orientations = results["total_orientations"]
        self.match_template_result.total_defocus = results["total_defocus"]

        # Apply the valid cropping mode to the results
        if do_valid_cropping:
            nx = self.template_volume.shape[-1]
            self.match_template_result.apply_valid_cropping((nx, nx))

        # Export the results to disk, if requested
        if do_result_export:
            self.match_template_result.export_results()

    def results_to_dataframe(
        self,
        half_template_width_pos_shift: bool = True,
        exclude_columns: Optional[list] = None,
        locate_peaks_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Converts the match template results to a DataFrame with additional info.

        Data included in this dataframe should be sufficient to do cross-correlation on
        the extracted peaks, that is, all the microscope parameters, defocus parameters,
        etc. are included in the dataframe. Run-specific filter information is *not*
        included in this dataframe; use the YAML configuration file to replicate a
        match_template run.

        Parameters
        ----------
        half_template_width_pos_shift : bool, optional
            If True, columns for the image peak position are shifted by half a template
            width to correspond to the center of the particle. This should be done when
            the position of a peak corresponds to the top-left corner of the template
            rather than the center. Default is True. This should generally be left as
            True unless you know what you are doing.
        exclude_columns : list, optional
            List of columns to exclude from the DataFrame. Default is None and no
            columns are excluded.
        locate_peaks_kwargs : dict, optional
            Keyword arguments to pass to the 'MatchTemplateResult.locate_peaks' method.
            Default is None and no additional keyword arguments are passed.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the match template results.
        """
        # Short circuit if no kwargs and peaks have already been located
        locate_kwargs = dict(locate_peaks_kwargs) if locate_peaks_kwargs else {}

        if self.search_windows and "z_score_cutoff" not in locate_kwargs:
            search_pixels = self.search_pixel_count
            if search_pixels <= 0:
                search_pixels = int(self.match_template_result.mip.numel())
            false_positives = locate_kwargs.get("false_positives", 1.0)
            total_correlations = max(1, search_pixels) * max(
                1, self.match_template_result.total_projections
            )
            locate_kwargs["z_score_cutoff"] = gaussian_noise_zscore_cutoff(
                num_ccg=total_correlations,
                false_positives=false_positives,
            )

        if not locate_kwargs:
            if self.match_template_result.match_template_peaks is None:
                self.match_template_result.locate_peaks()
        else:
            self.match_template_result.locate_peaks(**locate_kwargs)

        # DataFrame comes with the following columns :
        # ['mip', 'scaled_mip', 'correlation_mean', 'correlation_variance',
        # 'total_correlations'. 'pos_y', 'pos_x', 'psi', 'theta', 'phi',
        # 'relative_defocus', ]
        df = self.match_template_result.peaks_to_dataframe()

        # DataFrame currently contains pixel coordinates for results. Coordinates in
        # image correspond with upper left corner of the template. Need to translate
        # coordinates by half template width to get to particle center in image.
        # NOTE: We are assuming the template is cubic
        nx = mrcfile.open(self.template_volume_path).header.nx
        if half_template_width_pos_shift:
            df["pos_y_img"] = df["pos_y"] + nx // 2
            df["pos_x_img"] = df["pos_x"] + nx // 2
        else:
            df["pos_y_img"] = df["pos_y"]
            df["pos_x_img"] = df["pos_x"]

        # Also, the positions are in terms of pixels. Also add columns for particle
        # positions in terms of Angstroms.
        pixel_size = self.optics_group.pixel_size
        df["pos_y_img_angstrom"] = df["pos_y_img"] * pixel_size
        df["pos_x_img_angstrom"] = df["pos_x_img"] * pixel_size

        # Add microscope (CTF) parameters
        df["defocus_u"] = self.optics_group.defocus_u
        df["defocus_v"] = self.optics_group.defocus_v
        df["astigmatism_angle"] = self.optics_group.astigmatism_angle
        df["pixel_size"] = pixel_size
        df["refined_pixel_size"] = pixel_size
        df["voltage"] = self.optics_group.voltage
        df["spherical_aberration"] = self.optics_group.spherical_aberration
        df["amplitude_contrast_ratio"] = self.optics_group.amplitude_contrast_ratio
        df["phase_shift"] = self.optics_group.phase_shift
        df["ctf_B_factor"] = self.optics_group.ctf_B_factor

        # Add paths to the micrograph and reference template
        df["micrograph_path"] = self.micrograph_path
        df["template_path"] = self.template_volume_path

        # Add paths to the output statistic files
        df["mip_path"] = self.match_template_result.mip_path
        df["scaled_mip_path"] = self.match_template_result.scaled_mip_path
        df["psi_path"] = self.match_template_result.orientation_psi_path
        df["theta_path"] = self.match_template_result.orientation_theta_path
        df["phi_path"] = self.match_template_result.orientation_phi_path
        df["defocus_path"] = self.match_template_result.relative_defocus_path
        df["correlation_average_path"] = (
            self.match_template_result.correlation_average_path
        )
        df["correlation_variance_path"] = (
            self.match_template_result.correlation_variance_path
        )

        # Add particle index
        df["particle_index"] = df.index

        # Reorder columns
        df = df.reindex(columns=MATCH_TEMPLATE_DF_COLUMN_ORDER)

        # Drop columns if requested
        if exclude_columns is not None:
            df = df.drop(columns=exclude_columns)

        return df

    def save_config(self, path: str, mode: Literal["yaml", "json"] = "yaml") -> None:
        """Save this Pydandic model to disk. Wrapper around the serialization methods.

        Parameters
        ----------
        path : str
            Path to save the configuration file.
        mode : Literal["yaml", "json"], optional
            Serialization format to use. Default is 'yaml'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an invalid serialization mode is provided.
        """
        if mode == "yaml":
            self.to_yaml(path)
        elif mode == "json":
            self.to_json(path)
        else:
            raise ValueError(f"Invalid serialization mode '{mode}'.")
