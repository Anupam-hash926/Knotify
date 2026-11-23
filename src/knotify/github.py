"""Read-only GitHub client: GET requests and GraphQL *queries* only, never writes.

Handles pagination through ``Link`` headers, bounded retries for transient failures and
waits for primary/secondary rate-limit resets when the wait is short. The token is read from
the environment by the caller and is never logged or stored in outputs.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

API_URL = "https://api.github.com"
logger = logging.getLogger(__name__)


class RateLimited(RuntimeError):
    """The API budget is exhausted and the reset is further away than the allowed wait."""


def _resource(path: str) -> str:
    if path.startswith("/search"):
        return "search"
    if path.startswith("/graphql"):
        return "graphql"
    return "core"


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        max_wait: float = 120.0,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "knotify-dataset-tools",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.authenticated = bool(token)
        self.max_wait = max_wait
        self.retries = retries
        self._sleep = sleep
        self._clock = clock
        self._exhausted: dict[str, float] = {}
        self.request_count = 0
        self._http = httpx.Client(
            base_url=API_URL, headers=headers, timeout=60, transport=transport
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> GitHubClient:
        return cls(os.getenv("GITHUB_TOKEN") or None, **kwargs)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport
    def _wait(self, seconds: float, why: str) -> None:
        seconds = max(0.0, seconds)
        if seconds > self.max_wait:
            raise RateLimited(f"{why}; reset in {seconds:.0f}s exceeds {self.max_wait:.0f}s wait")
        if seconds:
            logger.info("Waiting %.0fs: %s", seconds, why)
            self._sleep(seconds)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if method not in {"GET", "POST"} or (method == "POST" and path != "/graphql"):
            raise ValueError("This client is read-only: GET requests and GraphQL queries only.")
        resource = _resource(path)
        for attempt in range(self.retries + 1):
            reset = self._exhausted.pop(resource, None)
            if reset is not None:
                self._wait(reset - self._clock() + 1, f"{resource} rate limit exhausted")
            try:
                response = self._http.request(method, path, **kwargs)
                self.request_count += 1
            except httpx.TransportError as exc:
                if attempt == self.retries:
                    raise
                logger.info("Transient %s; retrying", type(exc).__name__)
                self._sleep(min(2**attempt, 30))
                continue
            headers = response.headers
            if headers.get("x-ratelimit-remaining") == "0" and "x-ratelimit-reset" in headers:
                self._exhausted[resource] = float(headers["x-ratelimit-reset"])
            if response.status_code in (403, 429) and (
                "retry-after" in headers or headers.get("x-ratelimit-remaining") == "0"
            ):
                if attempt == self.retries:
                    break
                if "retry-after" in headers:
                    self._wait(float(headers["retry-after"]), "secondary rate limit")
                continue
            if response.status_code >= 500 and attempt < self.retries:
                self._sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            return response
        raise RateLimited(f"Gave up on {path} after {self.retries + 1} attempts")

    # ------------------------------------------------------------------ helpers
    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params).json()

    def paginate(self, path: str, params: dict[str, Any] | None = None) -> Iterator[dict]:
        """Yield every item from a list endpoint, following ``rel=next`` links."""
        params = {"per_page": 100, **(params or {})}
        url: str | None = path
        while url:
            response = self._request("GET", url, params=params)
            params = None  # the next link already carries the query string
            yield from response.json()
            url = response.links.get("next", {}).get("url")
            if url and url.startswith(API_URL):
                url = url[len(API_URL) :]

    def search_issues(
        self, query: str, *, per_page: int = 1, page: int = 1, sort: str | None = None
    ) -> dict:
        """Issue search with boolean operators (``advanced_search``)."""
        params = {"q": query, "per_page": per_page, "page": page, "advanced_search": "true"}
        if sort:
            params |= {"sort": sort, "order": "asc"}
        return self.get_json("/search/issues", params)

    def search_count(self, query: str) -> dict[str, Any]:
        result = self.search_issues(query)
        return {"total_count": result["total_count"], "incomplete": result["incomplete_results"]}

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        if not self.authenticated:
            raise RateLimited("GitHub GraphQL requires a token (GITHUB_TOKEN).")
        if query.lstrip().startswith("mutation"):
            raise ValueError("Mutations are not allowed.")
        payload = self._request(
            "POST", "/graphql", json={"query": query, "variables": variables or {}}
        ).json()
        if payload.get("errors"):
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        return payload["data"]

    def rate_limits(self) -> dict[str, Any]:
        data = self.get_json("/rate_limit")["resources"]
        return {k: {"limit": v["limit"], "remaining": v["remaining"]} for k, v in data.items()}
