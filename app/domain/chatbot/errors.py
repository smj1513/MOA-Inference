from __future__ import annotations


class ChatbotUpstreamUnavailableError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        upstream_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.upstream_status_code = upstream_status_code
