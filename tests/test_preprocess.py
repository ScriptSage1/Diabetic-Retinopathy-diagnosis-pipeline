import numpy as np
import pytest

from drscreen import preprocess as pp
from drscreen.synthetic import make_fundus


def test_plain_output_shape_dtype_and_mask():
    img = make_fundus(2, size=300, seed=1)
    out = pp.preprocess_plain(img, 128)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8
    assert out[0, 0].sum() == 0  # corner outside the circular FOV is black
    assert out[64, 64].sum() > 0


def test_black_border_removed():
    img = make_fundus(0, size=300, seed=2, border=True)
    assert img.shape[1] > img.shape[0]  # synthetic image has black side bars
    cropped = pp.crop_black_borders(img)
    assert cropped.shape[1] < img.shape[1]
    assert cropped.shape[1] <= cropped.shape[0] * 1.1  # roughly square after cropping


def test_ben_graham_and_tensor():
    plain = pp.preprocess_plain(make_fundus(1, 200, 3), 96)
    bg = pp.to_model_input(plain, True)
    assert bg.shape == plain.shape and bg.dtype == np.uint8
    t = pp.to_tensor_array(bg)
    assert t.shape == (3, 96, 96) and t.dtype == np.float32 and np.isfinite(t).all()


@pytest.mark.parametrize("bad", [np.zeros((10, 10, 3), np.uint8), np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200), np.uint8), "nope", None])
def test_invalid_arrays_rejected(bad):
    with pytest.raises(pp.ImageError):
        pp.validate_array(bad)


def test_read_image_errors(tmp_path):
    with pytest.raises(pp.ImageError):
        pp.read_image(tmp_path / "missing.png")
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(pp.ImageError):
        pp.read_image(bad)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(pp.ImageError):
        pp.read_image(empty)


def test_to_rgb_uint8_variants():
    rng = np.random.default_rng(0)
    assert pp.to_rgb_uint8(rng.integers(0, 255, (100, 100), dtype=np.uint8)).shape == (100, 100, 3)
    assert pp.to_rgb_uint8(rng.integers(0, 255, (100, 100, 4), dtype=np.uint8)).shape == (100, 100, 3)
    assert pp.to_rgb_uint8(rng.random((100, 100, 3))).dtype == np.uint8
    with pytest.raises(pp.ImageError):
        pp.to_rgb_uint8(None)
