
from functools import lru_cache

from typing import Optional, Union

import pywt
import torch

from ptwt.conv_transform_2 import _construct_2d_filt, _fwt_pad2

from ptwt._util import (
    _get_filter_tensors,
    _postprocess_coeffs,
    _preprocess_tensor,
)

from ptwt.constants import BoundaryMode, Wavelet, WaveletCoeff2d, WaveletDetailTuple2d

def wavedec2(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axes: tuple[int, int] = (-2, -1),
) -> WaveletCoeff2d:

    data, ds = _preprocess_tensor(data, ndim=2, axes=axes)

    dec_filt = _get_filter_tensors_and_construct_2d_filt(wavelet, data.device, data.dtype)

    if level is None:
        level = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst: list[WaveletDetailTuple2d] = []

    res_ll = data
    
    for _ in range(level):
        res_ll = _fwt_pad2(res_ll, wavelet, mode=mode)
        res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        to_append = WaveletDetailTuple2d(
            res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)
        )
        result_lst.append(to_append)

    result_lst.reverse()
    res_ll = res_ll.squeeze(1)
    result: WaveletCoeff2d = res_ll, *result_lst

    result = _postprocess_coeffs(result, ndim=2, ds=ds, axes=axes)

    return result

@lru_cache(maxsize=128)
def _get_filter_tensors_and_construct_2d_filt(wavelet, device, dtype):

    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=device, dtype=dtype
    )
    dec_filt = _construct_2d_filt(lo=dec_lo, hi=dec_hi)

    return dec_filt

