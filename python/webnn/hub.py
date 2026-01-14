"""
Simple Hugging Face Hub client for WebNN models.

Downloads WebNN model files (graph, weights, manifest) from Hugging Face Hub.
"""

import os
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError


class Hub:
    """Simple client for downloading WebNN models from Hugging Face Hub."""

    def __init__(self, cache_dir=None):
        """
        Initialize Hub client.

        Args:
            cache_dir: Directory to cache downloaded models.
                      Defaults to ~/.cache/webnn
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "webnn"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, model_id, force=False):
        """
        Download a WebNN model from Hugging Face Hub.

        Args:
            model_id: Model identifier (e.g., "tarekziade/mobilenet-webnn")
            force: Force re-download even if files exist

        Returns:
            dict with paths to downloaded files:
                - "graph": Path to .webnn file
                - "weights": Path to .weights file
                - "manifest": Path to manifest.json file

        Raises:
            ValueError: If model_id format is invalid
            RuntimeError: If download fails
        """
        if "/" not in model_id:
            raise ValueError(
                f"Invalid model_id format: {model_id}. "
                "Expected format: 'username/model-name'"
            )

        # Create model cache directory
        model_cache = self.cache_dir / model_id.replace("/", "--")
        model_cache.mkdir(parents=True, exist_ok=True)

        # Define file paths
        files = {
            "graph": model_cache / "model.webnn",
            "weights": model_cache / "model.weights",
            "manifest": model_cache / "manifest.json",
        }

        # Build download URLs
        base_url = f"https://huggingface.co/{model_id}/resolve/main"
        urls = {
            "graph": f"{base_url}/model.webnn?download=true",
            "weights": f"{base_url}/model.weights?download=true",
            "manifest": f"{base_url}/manifest.json?download=true",
        }

        # Download files
        for key, url in urls.items():
            filepath = files[key]

            # Skip if file exists and not forcing
            if filepath.exists() and not force:
                print(f"   [CACHED] {filepath.name}")
                continue

            try:
                print(f"   [DOWNLOAD] {filepath.name} from {model_id}")
                urlretrieve(url, filepath)
                size_mb = filepath.stat().st_size / 1024 / 1024
                print(f"   [OK] Downloaded {filepath.name} ({size_mb:.1f} MB)")
            except HTTPError as e:
                raise RuntimeError(
                    f"Failed to download {key} from {url}: {e}"
                ) from e

        # Validate graph file (text DSL format)
        try:
            with open(files["graph"]) as f:
                content = f.read()
                if not content.strip():
                    raise ValueError("Graph file is empty")
                # Basic validation: WebNN text format should contain "input" declarations
                if "input" not in content:
                    raise ValueError("Invalid WebNN text format: no input declarations found")
        except ValueError as e:
            raise RuntimeError(
                f"Invalid graph file: {e}"
            ) from e

        return {
            "graph": str(files["graph"]),
            "weights": str(files["weights"]),
            "manifest": str(files["manifest"]),
        }

    def list_cached_models(self):
        """
        List all cached models.

        Returns:
            List of model IDs (e.g., ["tarekziade/mobilenet-webnn"])
        """
        if not self.cache_dir.exists():
            return []

        models = []
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                # Convert back from "username--model" to "username/model"
                model_id = model_dir.name.replace("--", "/")
                models.append(model_id)

        return sorted(models)

    def clear_cache(self, model_id=None):
        """
        Clear cached model files.

        Args:
            model_id: Specific model to clear (clears all if None)
        """
        if model_id is None:
            # Clear entire cache
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                print(f"Cleared cache: {self.cache_dir}")
        else:
            # Clear specific model
            model_cache = self.cache_dir / model_id.replace("/", "--")
            if model_cache.exists():
                import shutil
                shutil.rmtree(model_cache)
                print(f"Cleared cache for: {model_id}")
