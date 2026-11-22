import warnings

# torch_geometric still calls the deprecated torch.jit.script on import.
warnings.filterwarnings(
    "ignore",
    message=(
        r"`torch\.jit\.script` is deprecated\. Please switch to "
        r"`torch\.compile` or `torch\.export`\."
    ),
    category=DeprecationWarning,
    module=r"torch\.jit\._script",
)
