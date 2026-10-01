"""
Focal Frequency Loss for latent-space frequency domain optimization.

Adapted from: https://github.com/EndlessSora/focal-frequency-loss

Reference:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    https://arxiv.org/pdf/2012.12821.pdf
"""

import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss - a frequency domain loss function for optimizing
    generative models by adaptively focusing on hard-to-synthesize frequency
    components.

    Applied in latent space (B, C, H, W) tensors where each channel is
    treated as an independent 2D signal for DFT analysis.

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix
            for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based
            focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum.
            Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by
            logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix
            using batch-based statistics. Default: False
    """

    def __init__(
        self,
        loss_weight=1.0,
        alpha=1.0,
        patch_factor=1,
        ave_spectrum=False,
        log_matrix=False,
        batch_matrix=False,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        """Convert a spatial tensor to its frequency representation via 2D DFT.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Frequency tensor of shape (N, num_patches, C, H, W, 2)
                where the last dim contains [real, imag] parts.
        """
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            "Patch factor should be divisible by image height and width"
        )
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(
                    x[
                        :,
                        :,
                        i * patch_h : (i + 1) * patch_h,
                        j * patch_w : (j + 1) * patch_w,
                    ]
                )

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm="ortho")
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        """Compute the focal frequency loss from frequency representations.

        Args:
            recon_freq (torch.Tensor): Reconstructed frequency tensor.
            real_freq (torch.Tensor): Real/target frequency tensor.
            matrix (torch.Tensor, optional): Pre-defined spectrum weight matrix.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on
            # current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based
            # statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = (
                    matrix_tmp
                    / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
                )

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            "The values of spectrum weight matrix should be in the range [0, 1], "
            "but got Min: %.10f Max: %.10f"
            % (weight_matrix.min().item(), weight_matrix.max().item())
        )

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).
            target (torch.Tensor): Target tensor of shape (N, C, H, W).
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (calculated online, dynamic).

        Returns:
            torch.Tensor: Scalar focal frequency loss value.
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
