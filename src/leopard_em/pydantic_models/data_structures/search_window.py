"""Definition of regions that constrain the match template search grid."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Iterator, Literal

import pandas as pd
from pydantic import Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


class SearchWindow(BaseModel2DTM):
    """Spatial window limiting where match_template evaluates correlations.

    The window is defined in image pixel coordinates using the template-centred
    convention that is used throughout the match template pipeline.  The
    half-width and half-height values describe how far away from the provided
    centre (in pixels) the search should extend when evaluating candidate
    positions.  The final search domain is expressed in the *valid* correlation
    grid where the template lies completely inside the micrograph.

    Parameters
    ----------
    center_y_img : float
        Vertical pixel coordinate of the particle centre in the micrograph.
    center_x_img : float
        Horizontal pixel coordinate of the particle centre in the micrograph.
    half_height : int
        Number of valid correlation pixels to include above and below the
        centre.  A value of zero restricts the search to a single pixel in the
        vertical direction.
    half_width : int, optional
        Number of valid correlation pixels to include left and right of the
        centre.  If omitted, the value from ``half_height`` is used.
    """

    center_y_img: float
    center_x_img: float
    half_height: Annotated[int, Field(ge=0)]
    half_width: Annotated[int | None, Field(ge=0)] = None

    def get_valid_bounds(
        self, image_shape: tuple[int, int], template_size: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the valid-correlation bounds covered by this window.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Height and width of the micrograph in pixels.
        template_size : int
            Width (and height) of the square 2DTM template in pixels.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int]]
            Inclusive-exclusive bounds in the valid grid for ``(y, x)``
            coordinates.  The first tuple contains the vertical ``(start, end)``
            indices and the second contains the horizontal indices.

        Raises
        ------
        ValueError
            If the template does not fit inside the provided image or if the
            window does not intersect the valid correlation grid.
        """

        image_height, image_width = image_shape
        if template_size <= 0:
            raise ValueError("Template size must be positive.")

        valid_height = image_height - template_size + 1
        valid_width = image_width - template_size + 1

        if valid_height <= 0 or valid_width <= 0:
            raise ValueError(
                "Template size is larger than the micrograph dimensions."
            )

        resolved_half_width = self.half_width if self.half_width is not None else self.half_height

        template_half = template_size // 2
        center_y_valid = int(round(self.center_y_img - template_half))
        center_x_valid = int(round(self.center_x_img - template_half))

        start_valid_y = max(0, center_y_valid - self.half_height)
        end_valid_y = min(valid_height, center_y_valid + self.half_height + 1)
        start_valid_x = max(0, center_x_valid - resolved_half_width)
        end_valid_x = min(valid_width, center_x_valid + resolved_half_width + 1)

        if start_valid_y >= end_valid_y or start_valid_x >= end_valid_x:
            raise ValueError(
                "Search window does not overlap the valid correlation grid."
            )

        return (start_valid_y, end_valid_y), (start_valid_x, end_valid_x)


def _dataframe_to_windows(df: pd.DataFrame) -> Iterator[SearchWindow]:
    """Yield :class:`SearchWindow` objects from a chunk of tabular data."""

    required_columns = {"center_y_img", "center_x_img", "half_height"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required search window columns: {missing}")

    has_half_width = "half_width" in df.columns

    for row in df.itertuples(index=False):
        half_width_value = getattr(row, "half_width", None) if has_half_width else None
        half_width: int | None
        if half_width_value is None:
            half_width = None
        elif isinstance(half_width_value, float) and math.isnan(half_width_value):
            half_width = None
        else:
            half_width = int(half_width_value)

        yield SearchWindow(
            center_y_img=float(row.center_y_img),
            center_x_img=float(row.center_x_img),
            half_height=int(row.half_height),
            half_width=half_width,
        )


def _resolve_table_format(
    table_path: Path, requested_format: Literal["auto", "csv", "parquet", "feather"]
) -> Literal["csv", "parquet", "feather"]:
    """Resolve the tabular format used to describe search windows."""

    if requested_format != "auto":
        return requested_format

    suffix = table_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".feather", ".ft"}:
        return "feather"

    raise ValueError(
        "Unable to infer search window table format from suffix. "
        "Specify 'search_window_table_format' explicitly."
    )


def iter_search_windows_from_table(
    table_path: str | Path,
    file_format: Literal["auto", "csv", "parquet", "feather"] = "auto",
    chunk_size: int | None = None,
) -> Iterator[SearchWindow]:
    """Stream :class:`SearchWindow` definitions from a tabular source.

    Parameters
    ----------
    table_path
        Path to the CSV/Parquet/Feather file containing window definitions.
    file_format
        Override for the file format.  When set to ``"auto"`` (default) the
        format is inferred from the filename suffix.
    chunk_size
        Number of rows to read at a time when streaming CSV input.  Other
        formats are loaded eagerly as they do not provide chunked readers in
        pandas.  ``None`` falls back to a default chunk size of 1024 rows.

    Yields
    ------
    SearchWindow
        Parsed window definitions respecting optional ``half_width`` values.
    """

    path = Path(table_path)
    if not path.exists():
        raise FileNotFoundError(f"Search window table '{path}' does not exist.")

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer when provided.")

    resolved_format = _resolve_table_format(path, file_format)
    effective_chunk_size = chunk_size if chunk_size is not None else 1024

    if resolved_format == "csv":
        csv_iterator = pd.read_csv(path, chunksize=effective_chunk_size)
        try:
            for chunk in csv_iterator:
                if chunk.empty:
                    continue
                yield from _dataframe_to_windows(chunk)
        finally:
            csv_iterator.close()
        return

    if resolved_format == "parquet":
        dataframe = pd.read_parquet(path)
        if dataframe.empty:
            return
        yield from _dataframe_to_windows(dataframe)
        return

    if resolved_format == "feather":
        dataframe = pd.read_feather(path)
        if dataframe.empty:
            return
        yield from _dataframe_to_windows(dataframe)
        return

    raise ValueError(f"Unsupported search window table format '{resolved_format}'.")


__all__ = ["SearchWindow", "iter_search_windows_from_table"]

