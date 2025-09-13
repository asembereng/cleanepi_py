"""
Comprehensive tests for the remove_duplicates module.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from cleanepi.cleaning.remove_duplicates import remove_duplicates


class TestRemoveDuplicates:
    """Test remove_duplicates functionality."""

    def test_basic_duplicate_removal(self):
        """Test basic duplicate removal."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3],
                "name": ["Alice", "Bob", "Bob", "Charlie"],
                "age": [25, 30, 30, 35],
            }
        )

        result = remove_duplicates(df)

        # Should remove one duplicate row
        assert len(result) == 3
        assert list(result["id"]) == [1, 2, 3]
        assert list(result["name"]) == ["Alice", "Bob", "Charlie"]

    def test_no_duplicates(self):
        """Test with DataFrame that has no duplicates."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        )

        result = remove_duplicates(df)

        # Should return unchanged DataFrame
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)

    def test_all_duplicates(self):
        """Test with DataFrame where all rows are duplicates."""
        df = pd.DataFrame(
            {"id": [1, 1, 1], "name": ["Alice", "Alice", "Alice"], "age": [25, 25, 25]}
        )

        result = remove_duplicates(df)

        # Should keep only one row
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_target_columns(self):
        """Test duplicate removal based on specific columns."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Alice", "Bob"],
                "age": [25, 30, 35, 40],
            }
        )

        result = remove_duplicates(df, target_columns=["name"])

        # Should remove duplicates based only on name column
        assert len(result) == 2
        unique_names = set(result["name"])
        assert len(unique_names) == 2
        assert "Alice" in unique_names
        assert "Bob" in unique_names

    def test_keep_first(self):
        """Test keeping first occurrence of duplicates."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3],
                "value": ["a", "b1", "b2", "c"],
                "score": [10, 20, 25, 30],
            }
        )

        result = remove_duplicates(df, target_columns=["id"], keep="first")

        # Should keep first occurrence of id=2
        assert len(result) == 3
        row_with_id_2 = result[result["id"] == 2]
        assert len(row_with_id_2) == 1
        assert row_with_id_2.iloc[0]["value"] == "b1"  # First occurrence
        assert row_with_id_2.iloc[0]["score"] == 20

    def test_keep_last(self):
        """Test keeping last occurrence of duplicates."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3],
                "value": ["a", "b1", "b2", "c"],
                "score": [10, 20, 25, 30],
            }
        )

        result = remove_duplicates(df, target_columns=["id"], keep="last")

        # Should keep last occurrence of id=2
        assert len(result) == 3
        row_with_id_2 = result[result["id"] == 2]
        assert len(row_with_id_2) == 1
        assert row_with_id_2.iloc[0]["value"] == "b2"  # Last occurrence
        assert row_with_id_2.iloc[0]["score"] == 25

    def test_keep_false(self):
        """Test removing all occurrences of duplicates."""
        df = pd.DataFrame({"id": [1, 2, 2, 3], "value": ["a", "b1", "b2", "c"]})

        result = remove_duplicates(df, target_columns=["id"], keep=False)

        # Should remove all occurrences of id=2
        assert len(result) == 2
        assert 2 not in result["id"].values
        assert list(result["id"]) == [1, 3]

    def test_multiple_target_columns(self):
        """Test duplicate removal based on multiple columns."""
        df = pd.DataFrame(
            {
                "first": ["A", "B", "A", "B", "C"],
                "second": ["X", "Y", "X", "Z", "X"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = remove_duplicates(df, target_columns=["first", "second"])

        # Should remove row where first='A' and second='X' appears twice
        assert len(result) == 4

        # Check that the combination (A,X) appears only once
        ax_rows = result[(result["first"] == "A") & (result["second"] == "X")]
        assert len(ax_rows) == 1

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError):  # Should raise validation error
            remove_duplicates(df)

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({"id": [1], "name": ["Alice"]})

        result = remove_duplicates(df)

        # Should return the same single row
        assert len(result) == 1
        pd.testing.assert_frame_equal(result, df)

    def test_nonexistent_target_columns(self):
        """Test with target columns that don't exist."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})

        with pytest.raises(ValueError):
            remove_duplicates(df, target_columns=["nonexistent"])

    def test_single_column_duplicates(self):
        """Test duplicate removal with single column."""
        df = pd.DataFrame({"id": [1, 1, 2, 2, 3], "other": ["a", "b", "c", "d", "e"]})

        result = remove_duplicates(df, target_columns=["id"])

        # Should keep first occurrence of each id
        assert len(result) == 3
        assert list(result["id"]) == [1, 2, 3]
        assert list(result["other"]) == ["a", "c", "e"]

    def test_mixed_data_types(self):
        """Test with mixed data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 2, 3],
                "float_col": [1.1, 2.2, 2.2, 3.3],
                "str_col": ["a", "b", "b", "c"],
                "bool_col": [True, False, False, True],
            }
        )

        result = remove_duplicates(df)

        # Should handle all data types
        assert len(result) == 3
        assert isinstance(result, pd.DataFrame)

    def test_with_nan_values(self):
        """Test duplicate removal with NaN values."""
        import numpy as np

        df = pd.DataFrame({"id": [1, 2, 2, 3], "value": [np.nan, "b", "b", np.nan]})

        result = remove_duplicates(df)

        # Should handle NaN values properly
        assert len(result) == 3
        assert isinstance(result, pd.DataFrame)

    def test_preserve_index(self):
        """Test that original index is preserved."""
        df = pd.DataFrame(
            {"id": [1, 2, 2, 3], "value": ["a", "b", "b", "c"]}, index=[10, 20, 30, 40]
        )

        result = remove_duplicates(df)

        # Should preserve original index values
        assert len(result) == 3
        assert list(result.index) == [10, 20, 40]  # First occurrence kept

    def test_large_dataframe(self):
        """Test with large DataFrame."""
        n_rows = 10000
        df = pd.DataFrame(
            {
                "id": list(range(n_rows // 2)) * 2,  # Each ID appears twice
                "value": list(range(n_rows)),
            }
        )

        result = remove_duplicates(df, target_columns=["id"])

        # Should remove half the rows (duplicates)
        assert len(result) == n_rows // 2

    @patch("cleanepi.cleaning.remove_duplicates.logger")
    def test_logging_with_duplicates(self, mock_logger):
        """Test logging when duplicates are found."""
        df = pd.DataFrame({"id": [1, 2, 2, 3], "value": ["a", "b", "b", "c"]})

        remove_duplicates(df)

        # Should log that duplicates were removed
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "Removed" in call_args and "duplicate" in call_args

    @patch("cleanepi.cleaning.remove_duplicates.logger")
    def test_logging_no_duplicates(self, mock_logger):
        """Test logging when no duplicates are found."""
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})

        remove_duplicates(df)

        # Should log that no duplicates were found
        mock_logger.info.assert_called_with("No duplicates found")

    def test_duplicate_removal_order(self):
        """Test that duplicate removal preserves correct order."""
        df = pd.DataFrame(
            {
                "group": ["A", "B", "A", "C", "B"],
                "order": [1, 2, 3, 4, 5],
                "value": ["x", "y", "z", "w", "v"],
            }
        )

        result = remove_duplicates(df, target_columns=["group"], keep="first")

        # Should keep first occurrence and maintain order
        assert len(result) == 3
        assert list(result["group"]) == ["A", "B", "C"]
        assert list(result["order"]) == [1, 2, 4]
        assert list(result["value"]) == ["x", "y", "w"]


@pytest.fixture
def sample_dataframe_with_duplicates():
    """Create sample DataFrame with duplicates for testing."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P001", "P003", "P002"],
            "visit_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-01",
                "2023-01-03",
                "2023-01-04",
            ],
            "age": [25, 30, 25, 35, 30],
            "status": ["healthy", "sick", "healthy", "recovered", "sick"],
        }
    )


@pytest.fixture
def sample_dataframe_no_duplicates():
    """Create sample DataFrame without duplicates for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85, 92, 78, 88, 95],
        }
    )
