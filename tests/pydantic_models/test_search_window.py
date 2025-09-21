"""Tests for the SearchWindow utility class."""

import pandas as pd
import pytest

from leopard_em.pydantic_models.data_structures import (
    SearchWindow,
    iter_search_windows_from_table,
)


def test_search_window_basic_bounds() -> None:
    """Search window returns expected valid-grid bounds away from edges."""

    window = SearchWindow(center_y_img=100.0, center_x_img=120.0, half_height=5, half_width=8)
    bounds_y, bounds_x = window.get_valid_bounds(image_shape=(200, 220), template_size=20)

    assert bounds_y == (85, 96)
    assert bounds_x == (102, 119)


def test_search_window_clamps_to_edges() -> None:
    """Bounds are clamped to the valid grid when the window touches image edges."""

    window = SearchWindow(center_y_img=8.0, center_x_img=8.0, half_height=3)
    bounds_y, bounds_x = window.get_valid_bounds(image_shape=(40, 40), template_size=20)

    assert bounds_y == (0, 2)
    assert bounds_x == (0, 2)


def test_search_window_raises_for_invalid_region() -> None:
    """A ValueError is raised when the window does not intersect valid pixels."""

    window = SearchWindow(center_y_img=0.0, center_x_img=0.0, half_height=0)

    with pytest.raises(ValueError):
        window.get_valid_bounds(image_shape=(32, 32), template_size=28)


def test_iter_search_windows_from_table_streams_csv(tmp_path) -> None:
    """CSV window tables are streamed in chunks and parsed correctly."""

    df = pd.DataFrame(
        {
            "center_y_img": [5.0, 10.5, 42.2],
            "center_x_img": [7.0, 11.5, 44.8],
            "half_height": [2, 3, 4],
            "half_width": [1, None, 6],
        }
    )
    table_path = tmp_path / "windows.csv"
    df.to_csv(table_path, index=False)

    windows = list(iter_search_windows_from_table(table_path, chunk_size=2))

    assert len(windows) == 3
    assert windows[0].half_width == 1
    assert windows[1].half_width is None
    assert windows[2].half_height == 4


def test_iter_search_windows_from_table_requires_columns(tmp_path) -> None:
    """Missing required columns raise a ValueError during iteration."""

    df = pd.DataFrame({"center_y_img": [1.0], "center_x_img": [2.0]})
    table_path = tmp_path / "bad.csv"
    df.to_csv(table_path, index=False)

    iterator = iter_search_windows_from_table(table_path)

    with pytest.raises(ValueError):
        next(iterator)


def test_iter_search_windows_from_table_validates_chunk_size(tmp_path) -> None:
    """Zero chunk size is rejected to avoid infinite loops."""

    df = pd.DataFrame(
        {
            "center_y_img": [1.0],
            "center_x_img": [2.0],
            "half_height": [1],
        }
    )
    table_path = tmp_path / "windows.csv"
    df.to_csv(table_path, index=False)

    iterator = iter_search_windows_from_table(table_path, chunk_size=0)

    with pytest.raises(ValueError):
        next(iterator)
