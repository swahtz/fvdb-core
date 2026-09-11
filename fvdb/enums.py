# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

try:
    from enum import IntEnum, StrEnum
except ImportError:  # Python 3.10 does not provide enum.StrEnum.
    from enum import Enum, IntEnum

    class StrEnum(str, Enum):
        """Compatibility implementation of :class:`enum.StrEnum` for Python 3.10."""

        __str__ = str.__str__


class ConvolutionTopologyPolicy(StrEnum):
    """Policy controlling the finite output topology of a convolution plan."""

    COMPLETE = "complete"
    """Generate the complete, uncropped structural topology."""

    RESTRICTED = "restricted"
    """Evaluate the convolution relation only on an explicit target grid."""


class ConvolutionTopologyProvenance(StrEnum):
    """How a convolution plan's finite topology was obtained."""

    GENERATED = "generated"
    """Generated from the source grid using the canonical convolution relation."""

    EXPLICIT_TARGET = "explicit_target"
    """Restricted to a target grid supplied by the caller."""

    EXACT_TRANSPOSE = "exact_transpose"
    """Reversed directly from another plan's stored finite edge set."""


class ConvolutionPhasePolicy(StrEnum):
    """Kernel phase convention used by a convolution plan."""

    TORCH_SAME_PHASE = "torch_same_phase"
    """Use PyTorch ``padding=floor((kernel_size - 1) / 2)`` phase."""


class SmoothingMode(IntEnum):
    """
    Laplacian smoothing mode used to de-staircase a signed distance field in
    :meth:`fvdb.Grid.reinitialize_sdf` / :meth:`fvdb.Grid.rebuild_narrow_band` (and their
    :class:`fvdb.GridBatch` counterparts).

    The number of smoothing passes is controlled separately by the ``smooth`` argument; this enum
    selects *which* umbrella-Laplacian flow each pass applies. Smoothing runs after the redistance
    and is followed by a short second redistance (with a freshly computed sign) so ``|grad phi| = 1``
    is restored; unlike a pure redistance it therefore *moves* the zero crossing to the smoothed
    surface. Thin (about one voxel) features can be eroded by :attr:`MEAN_CURVATURE`; prefer
    :attr:`TAUBIN` or ``smooth=0`` for them. Values mirror the C++
    ``fvdb::detail::ops::SmoothingMode`` enum.
    """

    MEAN_CURVATURE = 0
    """
    Mean-curvature flow: each pass moves every voxel toward the average of its 6 face neighbours.
    Effective at removing staircase artifacts but shrinks the surface (volume loss) if over-applied.
    """

    TAUBIN = 1
    """
    Volume-preserving Taubin smoothing: alternates a positive (shrinking) and a slightly larger
    negative (inflating) Laplacian step per pass, de-staircasing with much less volume loss.
    """
