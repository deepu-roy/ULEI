"""
Dataset loading utilities for evaluation data.

Supports loading datasets from various formats:
- JSONL files
- CSV files
- Pandas DataFrames
- Hugging Face datasets
- In-memory lists/dicts
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ulei.core.schemas import ContextItem, DatasetItem
from ulei.utils.errors import ConfigurationError


class DatasetLoader:
    """Loader for evaluation datasets from various sources."""

    @staticmethod
    def load_dataset(
        source: Union[str, Path, List[Dict[str, Any]], pd.DataFrame],
        format_hint: Optional[str] = None,
        **kwargs,
    ) -> List[DatasetItem]:
        """
        Load dataset from various sources.

        Args:
            source: Dataset source - file path, DataFrame, or list of dicts
            format_hint: Optional format hint ('jsonl', 'csv', 'parquet')
            **kwargs: Additional arguments for specific loaders

        Returns:
            List of DatasetItem objects

        Raises:
            ConfigurationError: If source format is unsupported or data is invalid
        """
        if isinstance(source, (str, Path)):
            return DatasetLoader._load_from_file(Path(source), format_hint, **kwargs)
        elif isinstance(source, list):
            return DatasetLoader._load_from_list(source)
        elif isinstance(source, pd.DataFrame):
            return DatasetLoader._load_from_dataframe(source)
        else:
            raise ConfigurationError(f"Unsupported dataset source type: {type(source)}")

    @staticmethod
    def _load_from_file(
        file_path: Path, format_hint: Optional[str] = None, **kwargs
    ) -> List[DatasetItem]:
        """
        Load dataset from file.

        Args:
            file_path: Path to dataset file
            format_hint: Optional format hint
            **kwargs: Additional arguments for file readers

        Returns:
            List of DatasetItem objects
        """
        if not file_path.exists():
            raise ConfigurationError(f"Dataset file not found: {file_path}")

        # Determine format from extension or hint
        if format_hint:
            format_type = format_hint.lower()
        else:
            format_type = file_path.suffix.lower().lstrip(".")

        if format_type in ["jsonl", "ndjson"]:
            return DatasetLoader._load_jsonl(file_path, **kwargs)
        elif format_type == "csv":
            return DatasetLoader._load_csv(file_path, **kwargs)
        elif format_type == "json":
            return DatasetLoader._load_json(file_path, **kwargs)
        elif format_type == "parquet":
            return DatasetLoader._load_parquet(file_path, **kwargs)
        else:
            raise ConfigurationError(
                f"Unsupported dataset format: {format_type}. "
                f"Supported formats: jsonl, csv, json, parquet"
            )

    @staticmethod
    def _load_jsonl(file_path: Path, **kwargs) -> List[DatasetItem]:
        """
        Load dataset from JSONL file.

        Args:
            file_path: Path to JSONL file
            **kwargs: Additional arguments (unused)

        Returns:
            List of DatasetItem objects
        """
        items = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        item = DatasetLoader._parse_item(data, f"line {line_num}")
                        items.append(item)
                    except json.JSONDecodeError as e:
                        raise ConfigurationError(
                            f"Invalid JSON on line {line_num} in {file_path}: {e}"
                        )
                    except Exception as e:
                        raise ConfigurationError(
                            f"Error parsing line {line_num} in {file_path}: {e}"
                        )

        except OSError as e:
            raise ConfigurationError(f"Error reading JSONL file {file_path}: {e}")

        return items

    @staticmethod
    def _load_csv(file_path: Path, **kwargs) -> List[DatasetItem]:
        """
        Load dataset from CSV file.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments passed to pandas.read_csv

        Returns:
            List of DatasetItem objects
        """
        try:
            # Set default CSV options
            csv_options = {
                "encoding": "utf-8",
                "na_filter": False,  # Don't convert strings to NaN
                **kwargs,
            }

            df = pd.read_csv(file_path, **csv_options)
            return DatasetLoader._load_from_dataframe(df)

        except Exception as e:
            raise ConfigurationError(f"Error reading CSV file {file_path}: {e}")

    @staticmethod
    def _load_json(file_path: Path, **kwargs) -> List[DatasetItem]:
        """
        Load dataset from JSON file.

        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments (unused)

        Returns:
            List of DatasetItem objects
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return DatasetLoader._load_from_list(data)
            elif isinstance(data, dict):
                # Check if it's a single item or has an 'items' key
                if "items" in data:
                    return DatasetLoader._load_from_list(data["items"])
                else:
                    return DatasetLoader._load_from_list([data])
            else:
                raise ConfigurationError("JSON file must contain a list or dict with 'items' key")

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading JSON file {file_path}: {e}")

    @staticmethod
    def _load_parquet(file_path: Path, **kwargs) -> List[DatasetItem]:
        """
        Load dataset from Parquet file.

        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments passed to pandas.read_parquet

        Returns:
            List of DatasetItem objects
        """
        try:
            df = pd.read_parquet(file_path, **kwargs)
            return DatasetLoader._load_from_dataframe(df)

        except Exception as e:
            raise ConfigurationError(f"Error reading Parquet file {file_path}: {e}")

    @staticmethod
    def _load_from_dataframe(df: pd.DataFrame) -> List[DatasetItem]:
        """
        Load dataset from pandas DataFrame.

        Args:
            df: Pandas DataFrame

        Returns:
            List of DatasetItem objects
        """
        items = []

        for idx, row in df.iterrows():
            try:
                # Convert row to dict, handling NaN values
                data = row.to_dict()

                # Clean up NaN values (pandas represents missing values as NaN)
                cleaned_data = {}
                for key, value in data.items():
                    if pd.isna(value):
                        cleaned_data[key] = None
                    else:
                        cleaned_data[key] = value

                item = DatasetLoader._parse_item(cleaned_data, f"row {idx}")
                items.append(item)

            except Exception as e:
                raise ConfigurationError(f"Error parsing row {idx}: {e}")

        return items

    @staticmethod
    def _load_from_list(data: List[Dict[str, Any]]) -> List[DatasetItem]:
        """
        Load dataset from list of dictionaries.

        Args:
            data: List of dictionaries containing item data

        Returns:
            List of DatasetItem objects
        """
        items = []

        for idx, item_data in enumerate(data):
            try:
                item = DatasetLoader._parse_item(item_data, f"item {idx}")
                items.append(item)
            except Exception as e:
                raise ConfigurationError(f"Error parsing item {idx}: {e}")

        return items

    @staticmethod
    def _parse_item(data: Dict[str, Any], context: str) -> DatasetItem:
        """
        Parse a single data item into DatasetItem.

        Args:
            data: Dictionary containing item data
            context: Context string for error reporting

        Returns:
            DatasetItem object

        Raises:
            ConfigurationError: If required fields are missing or invalid
        """
        try:
            # If data already has the correct structure, use it directly
            if "id" in data and "input" in data and "output" in data:
                return DatasetItem(**data)

            # Handle different field naming conventions for legacy data
            item_id = (
                data.get("item_id")
                or data.get("id")
                or str(hash(str(data)))  # Fallback: generate hash
            )

            query = data.get("query") or data.get("question") or data.get("input")

            response = (
                data.get("response")
                or data.get("answer")
                or data.get("output")
                or data.get("prediction")
            )

            context_list = data.get("context", [])

            # Handle different context formats
            if isinstance(context_list, str):
                context_list = [{"text": context_list}]
            elif isinstance(context_list, list):
                # Convert string items to ContextItem format
                normalized_context = []
                for ctx in context_list:
                    if isinstance(ctx, str):
                        normalized_context.append({"text": ctx})
                    elif isinstance(ctx, dict):
                        normalized_context.append(ctx)
                context_list = normalized_context
            else:
                context_list = []

            reference = (
                data.get("reference")
                or data.get("ground_truth")
                or data.get("expected")
                or data.get("target")
            )

            # Normalize reference to dict format
            if reference is not None and not isinstance(reference, dict):
                reference = {"expected": str(reference)}

            metadata = data.get("metadata", {})

            # Add any extra fields to metadata
            extra_fields = {
                k: v
                for k, v in data.items()
                if k
                not in [
                    "item_id",
                    "id",
                    "query",
                    "question",
                    "input",
                    "response",
                    "answer",
                    "output",
                    "prediction",
                    "context",
                    "reference",
                    "ground_truth",
                    "expected",
                    "target",
                    "metadata",
                ]
            }
            metadata.update(extra_fields)

            # Validate required fields
            if not query:
                raise ValueError(f"Missing query/question/input field in {context}")

            if not response:
                raise ValueError(f"Missing response/answer/output field in {context}")

            # Convert context to List[ContextItem]
            context_items = []
            for ctx in context_list:
                if isinstance(ctx, dict):
                    context_items.append(
                        ContextItem(text=ctx.get("text", ""), source=None, metadata={})
                    )
                else:
                    context_items.append(ContextItem(text=str(ctx), source=None, metadata={}))

            return DatasetItem(
                id=item_id,
                input={"query": query} if isinstance(query, str) else query,
                output={"answer": response} if isinstance(response, str) else response,
                context=context_items,
                reference=reference,
                metadata=metadata,
            )

        except Exception as e:
            raise ConfigurationError(f"Error parsing {context}: {e}")

    @staticmethod
    def validate_dataset(items: List[DatasetItem]) -> Dict[str, Any]:
        """
        Validate dataset and return summary statistics.

        Args:
            items: List of DatasetItem objects

        Returns:
            Dictionary with validation results and statistics
        """
        if not items:
            return {"valid": False, "error": "Dataset is empty", "count": 0}

        stats = {
            "valid": True,
            "count": len(items),
            "has_context": 0,
            "has_reference": 0,
            "unique_ids": len({item.id for item in items}),
            "field_coverage": {},
            "errors": [],
        }

        # Check field coverage
        for item in items:
            if item.context:
                stats["has_context"] += 1
            if item.reference:
                stats["has_reference"] += 1

        # Check for duplicate IDs
        if stats["unique_ids"] < stats["count"]:
            stats["errors"].append(
                f"Duplicate item IDs found: {stats['count'] - stats['unique_ids']} duplicates"
            )

        # Calculate percentages
        stats["context_coverage"] = (stats["has_context"] / stats["count"]) * 100
        stats["reference_coverage"] = (stats["has_reference"] / stats["count"]) * 100

        return stats

    @staticmethod
    def save_dataset(
        items: List[DatasetItem], output_path: Union[str, Path], format_type: str = "jsonl"
    ) -> None:
        """
        Save dataset to file.

        Args:
            items: List of DatasetItem objects
            output_path: Output file path
            format_type: Output format ('jsonl', 'csv', 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "jsonl":
            DatasetLoader._save_jsonl(items, output_path)
        elif format_type == "csv":
            DatasetLoader._save_csv(items, output_path)
        elif format_type == "json":
            DatasetLoader._save_json(items, output_path)
        else:
            raise ConfigurationError(f"Unsupported output format: {format_type}")

    @staticmethod
    def _save_jsonl(items: List[DatasetItem], output_path: Path) -> None:
        """Save items as JSONL."""
        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                json.dump(item.model_dump(), f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def _save_csv(items: List[DatasetItem], output_path: Path) -> None:
        """Save items as CSV."""
        # Convert to DataFrame and save
        data = [item.model_dump() for item in items]
        df = pd.json_normalize(data)
        df.to_csv(output_path, index=False, encoding="utf-8")

    @staticmethod
    def _save_json(items: List[DatasetItem], output_path: Path) -> None:
        """Save items as JSON."""
        data = [item.model_dump() for item in items]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Convenience functions for common use cases
def load_jsonl(file_path: Union[str, Path]) -> List[DatasetItem]:
    """Load dataset from JSONL file."""
    return DatasetLoader.load_dataset(file_path, format_hint="jsonl")


def load_csv(file_path: Union[str, Path], **kwargs) -> List[DatasetItem]:
    """Load dataset from CSV file."""
    return DatasetLoader.load_dataset(file_path, format_hint="csv", **kwargs)


def load_json(file_path: Union[str, Path]) -> List[DatasetItem]:
    """Load dataset from JSON file."""
    return DatasetLoader.load_dataset(file_path, format_hint="json")


def create_sample_dataset(size: int = 10) -> List[DatasetItem]:
    """
    Create a sample dataset for testing.

    Args:
        size: Number of items to create

    Returns:
        List of DatasetItem objects
    """
    items = []

    for i in range(size):
        items.append(
            DatasetItem(
                id=f"sample_{i}",
                input={"query": f"What is the capital of country {i}?"},
                output={"answer": f"The capital of country {i} is City {i}."},
                context=[
                    ContextItem(
                        text=f"Country {i} is located in Region {i % 3}.", source=None, metadata={}
                    )
                ],
                reference={"expected": f"City {i}"},
                metadata={"source": "sample_generator", "index": i},
            )
        )

    return items
