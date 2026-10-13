"""Application errors mapped to consistent HTTP responses."""


class AppError(Exception):
    """Base error with a public message, HTTP status, and machine-readable code."""

    def __init__(self, detail: str, *, status_code: int = 400, error_code: str = "app_error"):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code


class UnsupportedFileTypeError(AppError):
    def __init__(self, filename: str):
        super().__init__(
            f"Unsupported file type for '{filename}'. Allowed: .pdf, .txt, .md, .markdown.",
            status_code=400,
            error_code="unsupported_file_type",
        )


class EmptyDocumentError(AppError):
    def __init__(self, detail: str = "Uploaded file is empty."):
        super().__init__(detail, status_code=400, error_code="empty_document")


class UnreadableDocumentError(AppError):
    def __init__(self, detail: str = "The file could not be read or parsed."):
        super().__init__(detail, status_code=422, error_code="unreadable_document")


class PayloadTooLargeError(AppError):
    def __init__(self, max_mb: int):
        super().__init__(
            f"File exceeds the maximum upload size of {max_mb} MB.",
            status_code=413,
            error_code="payload_too_large",
        )


class EmptyKnowledgeBaseError(AppError):
    def __init__(self):
        super().__init__(
            "No documents have been indexed yet.",
            status_code=400,
            error_code="empty_knowledge_base",
        )


class EmbeddingModelMismatchError(AppError):
    def __init__(self, stored: str, requested: str):
        super().__init__(
            "The vector collection was created with a different embedding model.",
            status_code=500,
            error_code="embedding_model_mismatch",
        )
        self.stored = stored
        self.requested = requested


class LLMUnavailableError(AppError):
    def __init__(self, detail: str = "The language model provider is unavailable."):
        super().__init__(detail, status_code=503, error_code="llm_unavailable")


class ConfigurationError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=500, error_code="configuration_error")
