import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        r"`torch\.jit\.script` is deprecated\. Please switch to "
        r"`torch\.compile` or `torch\.export`\."
    ),
    category=DeprecationWarning,
    module=r"torch\.jit\._script",
)
