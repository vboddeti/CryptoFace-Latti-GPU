"""
Image preprocessing utilities for patch-based FHE inference.
"""


def extract_patches(img, patch_size=32):
    """
    Extract non-overlapping patches from an image tensor.

    Args:
        img: (B, C, H, W) image tensor
        patch_size: patch size in pixels (default: 32)

    Returns:
        List of N tensors, each of shape (B, C, patch_size, patch_size),
        ordered row-major (top-left to bottom-right).

    Example:
        >>> patches = extract_patches(img, patch_size=32)  # 64x64 → 4 patches
        >>> len(patches)
        4
    """
    B, C, H, W = img.shape
    P = patch_size
    patches = []
    for h in range(H // P):
        for w in range(W // P):
            patch = img[:, :, h * P:(h + 1) * P, w * P:(w + 1) * P]
            patches.append(patch)
    return patches
