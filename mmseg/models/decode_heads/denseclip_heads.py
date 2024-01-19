from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
import torch.nn as nn


@MODELS.register_module()
class IdentityHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__(
            input_transform=None, **kwargs)
        self.conv_seg = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1)

    def forward(self, inputs):
        return inputs