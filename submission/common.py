"""Contract paths and client-side face preprocessing shared by submission stages."""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from utils.preprocessing import extract_patches


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAIR_INDEX_WIDTH = 12


def get_repo_root() -> Path:
    return _REPO_ROOT


def pair_stem(index: int) -> str:
    if index < 0:
        raise ValueError("Pair indices must be non-negative")
    return f"p{index:0{_PAIR_INDEX_WIDTH}d}"


def decode_image(encoded) -> np.ndarray:
    payload = np.asarray(encoded, dtype=np.uint8).tobytes()
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1).copy()


def mute_logs() -> None:
    for name in ("matplotlib", "onnxruntime", "insightface", "PIL"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")


@contextmanager
def suppress_third_party_output():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


def load_submission_config() -> dict:
    with (_REPO_ROOT / "submission" / "config.yml").open() as stream:
        return yaml.safe_load(stream)["cryptoface"]


def get_face_params(size: int):
    harness_dir = str(_REPO_ROOT / "harness")
    if harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)
    from params import InstanceParams
    return InstanceParams(size, rootdir=_REPO_ROOT)


def parse_stage_args(resolve_checkpoint: bool = False) -> tuple:
    del resolve_checkpoint
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).stem} <size>", flush=True)
        raise SystemExit(1)
    mute_logs()
    size = int(sys.argv[1])
    return size, load_submission_config(), get_face_params(size)


def align_face(detector, img_chw_rgb: np.ndarray, output_size: int) -> np.ndarray | None:
    import cv2
    from insightface.utils.face_align import norm_crop

    image_bgr = img_chw_rgb.transpose(1, 2, 0)[:, :, ::-1]
    faces = detector.get(image_bgr)
    if not faces:
        return None
    if output_size > 112:
        aligned_bgr = norm_crop(image_bgr, faces[0].kps, image_size=output_size)
    else:
        aligned_bgr = norm_crop(image_bgr, faces[0].kps, image_size=112)
        if output_size != 112:
            aligned_bgr = cv2.resize(
                aligned_bgr, (output_size, output_size), interpolation=cv2.INTER_LINEAR
            )
    return aligned_bgr[:, :, ::-1].copy()


def _center_crop_resize(img_chw_rgb: np.ndarray, output_size: int) -> np.ndarray:
    import cv2

    image = img_chw_rgb.transpose(1, 2, 0)
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = image[top : top + side, left : left + side]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def to_tensor(img_hwc_rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img_hwc_rgb).permute(2, 0, 1).float()
    return ((tensor / 255.0) - 0.5).div(0.5).unsqueeze(0)


def preprocess_one_image(detector, img_chw_uint8: np.ndarray, input_size: int) -> list:
    aligned = align_face(detector, img_chw_uint8, input_size)
    if aligned is None:
        print("[common] Warning: no face detected; using center crop", flush=True)
        aligned = _center_crop_resize(img_chw_uint8, input_size)
    return extract_patches(to_tensor(aligned))


def limit_face_cpu_affinity() -> None:
    """Keep InsightFace/ONNX from consuming every CPU on large hosts."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return
    try:
        cpu_limit = int(os.environ.get("CRYPTOFACE_FACE_CPU_THREADS", "8"))
    except ValueError as error:
        raise ValueError("CRYPTOFACE_FACE_CPU_THREADS must be a positive integer") from error
    if cpu_limit <= 0:
        raise ValueError("CRYPTOFACE_FACE_CPU_THREADS must be a positive integer")
    available = sorted(os.sched_getaffinity(0))
    if len(available) > cpu_limit:
        selected = set(available[:cpu_limit])
        task_dir = Path("/proc/self/task")
        if task_dir.is_dir():
            for task in task_dir.iterdir():
                try:
                    os.sched_setaffinity(int(task.name), selected)
                except (OSError, ValueError):
                    pass
        os.sched_setaffinity(0, selected)
        torch.set_num_threads(cpu_limit)
        print(f"[common] Limited face preprocessing to {cpu_limit} CPUs", flush=True)


def load_detector():
    from insightface.app import FaceAnalysis

    limit_face_cpu_affinity()
    with suppress_third_party_output():
        detector = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        detector.prepare(ctx_id=-1, det_size=(640, 640))
    return detector


def _release_fd_cache(fd: int) -> None:
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return
    try:
        fadvise(fd, 0, 0, dontneed)
    except OSError:
        pass


def release_file_cache(path: Path) -> None:
    try:
        with Path(path).open("rb", buffering=0) as stream:
            _release_fd_cache(stream.fileno())
    except OSError:
        pass
