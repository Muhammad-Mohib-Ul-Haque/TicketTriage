"""Cached DummyJSON product client (owned by Mohib Ul Haque).

Network is optional: the classifier never depends on it. Lookup order is
    1. live API (status code checked) -> refresh cache
    2. on any network failure: cached JSON from a previous successful call
    3. bundled data/product_sample.json (offline demo only, clearly labelled)
    4. otherwise raise APIUnavailableError
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from . import config
from .exceptions import APIUnavailableError


@dataclass
class ProductLookup:
    """Product data plus where it came from, so the UI can be honest about freshness."""

    data: dict[str, Any]
    source: str  # "live", "cache" or "bundled sample"


class ProductAPIClient:
    """Fetch product context from DummyJSON with on-disk caching."""

    def __init__(
        self,
        base_url: str = config.PRODUCT_API_BASE,
        cache_dir: Path = config.CACHE_DIR,
        sample_path: Path = config.PRODUCT_SAMPLE_PATH,
        timeout: float = config.API_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._cache_dir = cache_dir
        self._sample_path = sample_path
        self._timeout = timeout

    def _cache_path(self, product_id: int) -> Path:
        return self._cache_dir / f"product_{product_id}.json"

    @staticmethod
    def _validate_id(product_id: int) -> int:
        if not isinstance(product_id, int) or isinstance(product_id, bool) or product_id < 1:
            raise ValueError("Product ID must be a positive integer.")
        return product_id

    def fetch_live(self, product_id: int = 1) -> dict[str, Any]:
        """Call the API once; raise APIUnavailableError on any network/HTTP/JSON problem."""
        url = f"{self._base_url}/{self._validate_id(product_id)}"
        try:
            response = requests.get(url, timeout=self._timeout)
        except requests.Timeout as exc:
            raise APIUnavailableError(f"Request timed out after {self._timeout}s.") from exc
        except requests.ConnectionError as exc:
            raise APIUnavailableError("Could not connect (offline?).") from exc
        except requests.RequestException as exc:
            raise APIUnavailableError(f"Request failed: {exc}") from exc

        if response.status_code != 200:
            raise APIUnavailableError(f"API returned HTTP {response.status_code} for {url}.")
        try:
            data = response.json()
        except ValueError as exc:
            raise APIUnavailableError("API response was not valid JSON.") from exc
        if not isinstance(data, dict) or "title" not in data:
            raise APIUnavailableError("API response did not look like a product.")
        return data

    def _write_cache(self, product_id: int, data: dict[str, Any]) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(product_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            # Caching is a convenience; failing to cache must not break the lookup.
            pass

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def get_product(self, product_id: int = 1, use_live: bool = True) -> ProductLookup:
        """Return product info, preferring live data and falling back to cache/sample."""
        self._validate_id(product_id)
        error: APIUnavailableError | None = None
        if use_live:
            try:
                data = self.fetch_live(product_id)
                self._write_cache(product_id, data)
                return ProductLookup(data=data, source="live")
            except APIUnavailableError as exc:
                error = exc

        cached = self._read_json(self._cache_path(product_id))
        if cached is not None:
            return ProductLookup(data=cached, source="cache")

        sample = self._read_json(self._sample_path)
        if sample is not None and sample.get("id") == product_id:
            return ProductLookup(data=sample, source="bundled sample")

        detail = f" ({error})" if error else ""
        raise APIUnavailableError(f"No live data, cache or sample for product {product_id}{detail}.")

    @staticmethod
    def format_product(data: dict[str, Any]) -> str:
        """Readable multi-line summary of the fields a support agent would want."""
        fields = [
            ("Title", data.get("title")),
            ("Brand", data.get("brand")),
            ("Category", data.get("category")),
            ("Price", f"${data['price']}" if "price" in data else None),
            ("Stock", data.get("stock")),
            ("Availability", data.get("availabilityStatus")),
            ("Shipping", data.get("shippingInformation")),
            ("Warranty", data.get("warrantyInformation")),
            ("Return policy", data.get("returnPolicy")),
        ]
        return "\n".join(f"  {k:<14}: {v}" for k, v in fields if v not in (None, ""))
