"""Voxel-based head model for direct image reconstruction in the brain volume.

The :class:`VoxelHeadModel` is an alternative to
:class:`cedalion.dot.head_model.TwoSurfaceHeadModel`.  Instead of projecting
voxel-space fluence onto a triangulated cortex mesh, the brain is represented
as a reduced set of brain voxels.  This keeps depth information in the
reconstructed image — preferable for high-density montages that reach the
cortical and subcortical volume.

Both head model classes expose the same attribute and method names where
possible so that downstream code (``ForwardModel``, ``ImageRecon``) is
duck-typed and works transparently for either representation.  In particular
the sparse mapping is still called ``voxel_to_vertex_brain``: in this class
the columns are voxels rather than mesh vertices, and the alias property
``voxel_to_voxel_brain`` is provided for code that prefers the more honest
name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse
import trimesh
import xarray as xr
from scipy.sparse import coo_array

import cedalion
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import units, xrutils
from cedalion.dot.utils import (
    map_segmentation_mask_to_surface,
    reduce_and_map_brain_voxels,
)
from cedalion.geometry.segmentation import (
    surface_from_segmentation,
    voxels_from_segmentation,
)
from cedalion.io import read_mrk_json, read_segmentation_masks


@dataclass
class VoxelHeadModel:
    """Head model that represents the brain as a reduced set of voxels.

    The "vertex" axis on ``voxel_to_vertex_brain`` indexes voxels here, not
    mesh vertices — this naming is preserved so that ``ForwardModel`` and
    image-reconstruction code work without forks.  Use the
    :attr:`voxel_to_voxel_brain` alias for clearer reading code.

    Attributes:
        segmentation_masks: Segmentation masks of the head for each tissue
            type.
        brain: Brain voxels (reduced set).
        scalp: Scalp surface mesh.
        landmarks: Anatomical landmarks; may be ``None``.
        t_ijk2ras: Affine transform from voxel (ijk) to RAS space.
        t_ras2ijk: Affine transform from RAS to voxel (ijk) space.
        voxel_to_vertex_brain: Sparse matrix of shape
            ``(ncells, n_brain_voxels)`` mapping flat segmentation cells
            to kept brain voxels.
        voxel_to_vertex_scalp: Sparse matrix of shape
            ``(ncells, n_scalp_vertices)`` mapping flat segmentation cells
            to scalp surface vertices.
        brain_mask: 3D boolean ``np.ndarray`` aligned to the segmentation
            grid.  ``True`` at currently-kept voxels.
    """

    segmentation_masks: xr.DataArray
    brain: cdc.Voxels
    scalp: cdc.Surface
    landmarks: cdt.LabeledPoints | None
    t_ijk2ras: cdt.AffineTransform
    t_ras2ijk: cdt.AffineTransform
    voxel_to_vertex_brain: scipy.sparse.spmatrix
    voxel_to_vertex_scalp: scipy.sparse.spmatrix
    brain_mask: np.ndarray

    @classmethod
    def from_segmentation(
        cls,
        segmentation_dir: str,
        mask_files: dict[str, str] = {
            "csf": "csf.nii",
            "gm": "gm.nii",
            "scalp": "scalp.nii",
            "skull": "skull.nii",
            "wm": "wm.nii",
        },
        landmarks_ras_file: str | None = None,
        brain_seg_types: list[str] = ["gm", "wm"],
        scalp_seg_types: list[str] = ["scalp"],
        smoothing: float = 0.5,
        scalp_face_count: int | None = 60000,
        fill_holes: bool = False,
        max_dist: cdt.QLength = 10 * units.mm,
        scalp_surface_file: str | None = None,
    ) -> "VoxelHeadModel":
        """Build a VoxelHeadModel from segmentation masks.

        The brain is represented as the set of voxels in
        ``brain_seg_types`` that lie within ``max_dist`` of the scalp
        surface.  The scalp is meshed and decimated as in
        :class:`TwoSurfaceHeadModel` unless an externally-meshed scalp is
        provided via ``scalp_surface_file``.

        Args:
            segmentation_dir: Folder containing the NIfTI segmentation masks.
            mask_files: Mapping from segmentation type name to NIfTI filename.
            landmarks_ras_file: Optional path to anatomical landmarks in RAS.
            brain_seg_types: Segmentation types that constitute brain tissue.
            scalp_seg_types: Segmentation types that constitute scalp.
            smoothing: Taubin lambda for the scalp mesh smoothing pass
                (ignored if ``scalp_surface_file`` is provided).
            scalp_face_count: Target number of scalp mesh faces (after
                decimation).  Pass ``None`` to skip decimation.  Ignored
                if ``scalp_surface_file`` is provided.
            fill_holes: If ``True``, fill holes in the binary masks before
                marching cubes / voxelisation.
            max_dist: Maximum distance from the scalp at which brain voxels
                are kept.
            scalp_surface_file: Optional path to a precomputed scalp mesh
                (e.g. from FreeSurfer, or shipped with a standard atlas).
        """

        segmentation_masks, t_ijk2ras = read_segmentation_masks(
            segmentation_dir, mask_files
        )

        t_ras2ijk = xrutils.pinv(t_ijk2ras)
        crs_ras = t_ijk2ras.dims[0]

        if landmarks_ras_file is not None:
            if not os.path.isabs(landmarks_ras_file):
                landmarks_ras_file = os.path.join(segmentation_dir, landmarks_ras_file)

            landmarks_ras = read_mrk_json(landmarks_ras_file, crs=crs_ras)
            landmarks_ijk = landmarks_ras.points.apply_transform(t_ras2ijk)
        else:
            landmarks_ijk = None

        brain_voxels_ijk = voxels_from_segmentation(
            segmentation_masks, brain_seg_types, fill_holes_in_mask=fill_holes
        )

        if scalp_surface_file is not None:
            scalp_mesh = trimesh.load(scalp_surface_file)
            scalp_ijk = cdc.TrimeshSurface(
                scalp_mesh, "ijk", cedalion.units.Unit("1")
            )
        else:
            all_seg_types = segmentation_masks.segmentation_type.values
            scalp_ijk = surface_from_segmentation(
                segmentation_masks, all_seg_types, fill_holes_in_mask=fill_holes
            )

            if smoothing > 0:
                scalp_ijk = scalp_ijk.smooth(smoothing)

            if scalp_face_count is not None:
                vtk_scalp_ijk = cdc.VTKSurface.from_trimeshsurface(scalp_ijk)
                reduction = 1.0 - scalp_face_count / scalp_ijk.nfaces
                vtk_scalp_ijk = vtk_scalp_ijk.decimate(reduction)
                scalp_ijk = cdc.TrimeshSurface.from_vtksurface(vtk_scalp_ijk)

        scalp_ijk = scalp_ijk.fix_vertex_normals()

        brain_mask_seg = segmentation_masks.sel(
            segmentation_type=brain_seg_types
        ).any("segmentation_type")
        scalp_mask_seg = segmentation_masks.sel(
            segmentation_type=scalp_seg_types
        ).any("segmentation_type")

        # filter brain voxels to those within max_dist of the scalp surface
        voxel_to_vertex_brain, brain_voxels_ras, kept_mask_3d = (
            reduce_and_map_brain_voxels(
                brain_mask_seg,
                t_ijk2ras,
                scalp_ijk.apply_transform(t_ijk2ras),
                brain_voxels_ijk.apply_transform(t_ijk2ras),
                max_dist=max_dist,
            )
        )
        brain_voxels = brain_voxels_ras.apply_transform(t_ras2ijk)

        voxel_to_vertex_scalp = map_segmentation_mask_to_surface(
            scalp_mask_seg, t_ijk2ras, scalp_ijk.apply_transform(t_ijk2ras)
        )

        return cls(
            segmentation_masks=segmentation_masks,
            brain=brain_voxels,
            scalp=scalp_ijk,
            landmarks=landmarks_ijk,
            t_ijk2ras=t_ijk2ras,
            t_ras2ijk=t_ras2ijk,
            voxel_to_vertex_brain=voxel_to_vertex_brain,
            voxel_to_vertex_scalp=voxel_to_vertex_scalp,
            brain_mask=kept_mask_3d,
        )

    def __repr__(self) -> str:
        tissue_types = ", ".join(self.segmentation_masks.segmentation_type.values)
        return (
            f"VoxelHeadModel(\n"
            f"  crs: {self.crs}\n"
            f"  tissue_types: {tissue_types}\n"
            f"  brain voxels: {self.brain.nvertices} units: {self.brain.units}\n"
            f"  scalp faces: {self.scalp.nfaces} vertices: {self.scalp.nvertices} "
            f"units: {self.scalp.units}\n"
            "  landmarks: "
            f"{len(self.landmarks) if self.landmarks is not None else 'None'}\n"
            ")"
        )

    @property
    def crs(self) -> str:
        """Coordinate reference system of the head model."""
        assert self.brain.crs == self.scalp.crs
        if self.landmarks is not None:
            assert self.scalp.crs == self.landmarks.points.crs
        return self.brain.crs

    @property
    def voxel_to_voxel_brain(self) -> scipy.sparse.spmatrix:
        """Alias of ``voxel_to_vertex_brain`` with a more honest name.

        In a :class:`VoxelHeadModel` the columns of this sparse matrix
        index voxels, not mesh vertices.  The duck-typed name
        ``voxel_to_vertex_brain`` is kept for compatibility with
        ``ForwardModel`` and ``ImageRecon``; user code that wants a
        clearer name can use this alias.
        """
        return self.voxel_to_vertex_brain

    def apply_transform(self, transform: cdt.AffineTransform) -> "VoxelHeadModel":
        """Apply an affine transform to surfaces, voxels and landmarks.

        Args:
            transform: 4x4 affine transform.

        Returns:
            New :class:`VoxelHeadModel` in the target CRS.
        """
        brain = self.brain.apply_transform(transform)
        scalp = self.scalp.apply_transform(transform)
        landmarks = (
            self.landmarks.points.apply_transform(transform)
            if self.landmarks is not None
            else None
        )

        return VoxelHeadModel(
            segmentation_masks=self.segmentation_masks,
            brain=brain,
            scalp=scalp,
            landmarks=landmarks,
            t_ijk2ras=self.t_ijk2ras,
            t_ras2ijk=self.t_ras2ijk,
            voxel_to_vertex_brain=self.voxel_to_vertex_brain,
            voxel_to_vertex_scalp=self.voxel_to_vertex_scalp,
            brain_mask=self.brain_mask,
        )

    def save(self, foldername: str) -> None:
        """Save the head model to a folder.

        Args:
            foldername: Destination directory; created if it does not exist.
        """
        if not os.path.isdir(foldername):
            os.makedirs(foldername, exist_ok=True)

        self.segmentation_masks.to_netcdf(
            os.path.join(foldername, "segmentation_masks.nc")
        )
        np.savez_compressed(
            os.path.join(foldername, "brain_voxels.npz"),
            voxels=self.brain.voxels,
            crs=np.array(self.brain.crs),
            units=np.array(str(self.brain.units)),
        )
        self.scalp.mesh.export(
            os.path.join(foldername, "scalp.ply"), file_type="ply"
        )
        if self.landmarks is not None:
            self.landmarks.drop_vars("type").to_netcdf(
                os.path.join(foldername, "landmarks.nc")
            )
        self.t_ijk2ras.to_netcdf(os.path.join(foldername, "t_ijk2ras.nc"))
        scipy.sparse.save_npz(
            os.path.join(foldername, "voxel_to_vertex_brain.npz"),
            self.voxel_to_vertex_brain,
        )
        scipy.sparse.save_npz(
            os.path.join(foldername, "voxel_to_vertex_scalp.npz"),
            self.voxel_to_vertex_scalp,
        )
        np.savez_compressed(
            os.path.join(foldername, "brain_mask.npz"), brain_mask=self.brain_mask
        )

    @classmethod
    def load(cls, foldername: str) -> "VoxelHeadModel":
        """Load a head model from a folder previously produced by :meth:`save`.

        Args:
            foldername: Folder containing the head-model files.
        """
        required = [
            "segmentation_masks.nc",
            "brain_voxels.npz",
            "scalp.ply",
            "t_ijk2ras.nc",
            "voxel_to_vertex_brain.npz",
            "voxel_to_vertex_scalp.npz",
            "brain_mask.npz",
        ]
        for fn in required:
            if not os.path.exists(os.path.join(foldername, fn)):
                raise ValueError("%s does not exist." % os.path.join(foldername, fn))

        segmentation_masks = xr.load_dataarray(
            os.path.join(foldername, "segmentation_masks.nc")
        )

        brain_npz = np.load(os.path.join(foldername, "brain_voxels.npz"))
        brain = cdc.Voxels(
            brain_npz["voxels"],
            str(brain_npz["crs"]),
            cedalion.units.Unit(str(brain_npz["units"])),
        )

        scalp_mesh = trimesh.load(
            os.path.join(foldername, "scalp.ply"), process=False
        )
        scalp = cdc.TrimeshSurface(scalp_mesh, "ijk", cedalion.units.Unit("1"))

        landmarks_path = os.path.join(foldername, "landmarks.nc")
        if os.path.exists(landmarks_path):
            landmarks_ds = xr.load_dataset(landmarks_path)
            landmarks = xr.DataArray(
                landmarks_ds.to_array()[0],
                coords={
                    "label": ("label", landmarks_ds.label.values),
                    "type": (
                        "label",
                        [cdc.PointType.LANDMARK] * len(landmarks_ds.label),
                    ),
                },
            )
        else:
            landmarks = None

        t_ijk2ras = xr.load_dataarray(os.path.join(foldername, "t_ijk2ras.nc"))
        t_ras2ijk = xrutils.pinv(t_ijk2ras)

        voxel_to_vertex_brain = scipy.sparse.load_npz(
            os.path.join(foldername, "voxel_to_vertex_brain.npz")
        )
        voxel_to_vertex_scalp = scipy.sparse.load_npz(
            os.path.join(foldername, "voxel_to_vertex_scalp.npz")
        )
        brain_mask = np.load(os.path.join(foldername, "brain_mask.npz"))["brain_mask"]

        return cls(
            segmentation_masks=segmentation_masks,
            brain=brain,
            scalp=scalp,
            landmarks=landmarks,
            t_ijk2ras=t_ijk2ras,
            t_ras2ijk=t_ras2ijk,
            voxel_to_vertex_brain=voxel_to_vertex_brain,
            voxel_to_vertex_scalp=voxel_to_vertex_scalp,
            brain_mask=brain_mask,
        )

    def _rebuild_brain_voxel_mapping(
        self, kept_in_current: np.ndarray
    ) -> "VoxelHeadModel":
        """Rebuild ``brain``, ``voxel_to_vertex_brain`` and ``brain_mask``.

        Args:
            kept_in_current: 1D boolean array of length ``self.brain.nvertices``.
                ``True`` at voxels that should be retained.

        Returns:
            New :class:`VoxelHeadModel` with the reduced brain voxel set.
        """
        if kept_in_current.shape != (self.brain.nvertices,):
            raise ValueError(
                "kept_in_current must have shape (n_current_brain_voxels,)"
            )

        # update the 3D brain mask: indices of currently-kept voxels in
        # the segmentation grid are stored in flat order
        flat_indices_current = np.flatnonzero(self.brain_mask)
        assert flat_indices_current.shape[0] == self.brain.nvertices
        flat_indices_kept = flat_indices_current[kept_in_current]

        new_brain_mask = np.zeros_like(self.brain_mask, dtype=bool)
        new_brain_mask.flat[flat_indices_kept] = True

        new_voxels = self.brain.voxels[kept_in_current]
        new_brain = cdc.Voxels(new_voxels, self.brain.crs, self.brain.units)

        ncells = self.voxel_to_vertex_brain.shape[0]
        nvoxels_new = new_brain.nvertices
        cell_indices_kept = flat_indices_kept
        voxel_indices_new = np.arange(nvoxels_new)

        new_voxel_to_vertex_brain = coo_array(
            (
                np.ones(len(cell_indices_kept)),
                (cell_indices_kept, voxel_indices_new),
            ),
            shape=(ncells, nvoxels_new),
        )

        return VoxelHeadModel(
            segmentation_masks=self.segmentation_masks,
            brain=new_brain,
            scalp=self.scalp,
            landmarks=self.landmarks,
            t_ijk2ras=self.t_ijk2ras,
            t_ras2ijk=self.t_ras2ijk,
            voxel_to_vertex_brain=new_voxel_to_vertex_brain,
            voxel_to_vertex_scalp=self.voxel_to_vertex_scalp,
            brain_mask=new_brain_mask,
        )

    def reduce_voxels_to_probe(
        self,
        geo3d: cdt.LabeledPoints,
        max_dist: cdt.QLength = 50 * units.mm,
    ) -> "VoxelHeadModel":
        """Drop brain voxels far from any optode.

        Used to restrict the head model to the region a probe can possibly
        see.  Distances are computed in the head-model CRS after
        transforming both voxels and optode positions to RAS.

        Args:
            geo3d: Optode positions (and any other points to consider).
            max_dist: Maximum distance from any point in ``geo3d`` for a
                voxel to be retained.

        Returns:
            New :class:`VoxelHeadModel` with the reduced voxel set.
        """
        if self.crs == "ijk":
            brain_ras = self.brain.apply_transform(self.t_ijk2ras)
        else:
            brain_ras = self.brain

        if geo3d.points.crs == "ijk":
            geo3d = geo3d.points.apply_transform(self.t_ijk2ras)

        target_units = brain_ras.units
        max_dist_value = float(max_dist.to(target_units).magnitude)

        voxels_ras = brain_ras.voxels
        points = geo3d.pint.to(target_units).pint.dequantify().values

        kept = np.zeros(voxels_ras.shape[0], dtype=bool)
        for p in points:
            kept |= np.linalg.norm(voxels_ras - p, axis=1) < max_dist_value

        return self._rebuild_brain_voxel_mapping(kept)

    def reduce_voxels_by_fluence(
        self,
        fluence_fname: str | Path,
        rel_threshold: float = 1e-3,
    ) -> "VoxelHeadModel":
        """Drop brain voxels with low total fluence.

        Reads the fluence HDF5 file produced by ``ForwardModel`` and computes
        per-voxel ``Σ_optodes |fluence|`` (max over wavelengths).  Voxels whose
        value falls below ``rel_threshold * max`` are dropped.  Run this
        between :meth:`ForwardModel.compute_fluence_mcx` and
        :meth:`ForwardModel.compute_sensitivity` so the subsequently-built
        sensitivity matrix is small.

        Args:
            fluence_fname: Path to the fluence HDF5 file.
            rel_threshold: Relative threshold; voxels with summed fluence
                below ``rel_threshold * max_voxel_value`` are dropped.

        Returns:
            New :class:`VoxelHeadModel` with the reduced brain voxel set.
        """
        from cedalion.io.forward_model import FluenceFile

        with FluenceFile(fluence_fname, "r") as fluence_file:
            optode_labels = fluence_file.optode_labels
            wavelengths = fluence_file.wavelengths

            voxel_sum = None
            for label in optode_labels:
                for wl in wavelengths:
                    f = np.abs(fluence_file.get_fluence(label, wl))
                    if voxel_sum is None:
                        voxel_sum = np.zeros(
                            (len(wavelengths),) + f.shape, dtype=np.float64
                        )
                    i_wl = wavelengths.index(wl)
                    voxel_sum[i_wl] += f

        # max over wavelengths, then flatten to match the segmentation cell order
        per_voxel = voxel_sum.max(axis=0).reshape(-1)

        # restrict to currently-kept brain voxels (column order of
        # voxel_to_vertex_brain matches np.flatnonzero(brain_mask))
        flat_indices_current = np.flatnonzero(self.brain_mask)
        per_brain_voxel = per_voxel[flat_indices_current]

        max_value = per_brain_voxel.max()
        if max_value <= 0:
            raise ValueError(
                "All brain voxels have zero fluence — check the fluence file."
            )

        kept = per_brain_voxel >= rel_threshold * max_value
        return self._rebuild_brain_voxel_mapping(kept)

    def reduce_voxels_to_sensitivity(
        self,
        Adot: xr.DataArray,
        sensitivity_threshold: float = 1e-4,
    ) -> "VoxelHeadModel":
        """Drop brain voxels with low summed sensitivity in ``Adot``.

        Args:
            Adot: Sensitivity matrix with dims ``(channel, vertex,
                wavelength)`` and an ``is_brain`` per-vertex coordinate.
            sensitivity_threshold: Absolute threshold on
                ``Σ_channel Σ_wavelength Adot[:, voxel, :]``; voxels at or
                below this value are dropped.

        Returns:
            New :class:`VoxelHeadModel` with the reduced brain voxel set.
        """
        Adot_brain = np.asarray(Adot.isel(vertex=Adot.is_brain.values).values)
        per_voxel = np.abs(Adot_brain).sum(axis=2).sum(axis=0)

        if per_voxel.shape[0] != self.brain.nvertices:
            raise ValueError(
                "Adot brain-vertex count does not match head_model.brain.nvertices "
                f"({per_voxel.shape[0]} vs {self.brain.nvertices}). Was Adot built "
                "from this head model?"
            )

        kept = per_voxel > sensitivity_threshold
        return self._rebuild_brain_voxel_mapping(kept)

    def scale_to_landmarks(
        self,
        target_landmarks: cdt.LabeledPoints,
        mode: str = "general",
    ) -> "VoxelHeadModel":
        """Scale and align the head model to a set of landmarks.

        Args:
            target_landmarks: Target landmark positions (e.g. from a
                digitiser).  Must contain the same label subset as the
                head model's landmarks.
            mode: ``"trans_rot_isoscale"`` or ``"general"``; see
                ``cedalion.geometry.registration``.

        Returns:
            New :class:`VoxelHeadModel` aligned to ``target_landmarks``.

        Note:
            The voxel grid is no longer integer-aligned after a general
            affine.  Consumers that need integer voxel indices should
            re-snap.
        """
        from cedalion.geometry.registration import (
            register_general_affine,
            register_trans_rot_isoscale,
        )

        if self.crs == "ijk":
            landmarks_ras = self.landmarks.points.apply_transform(self.t_ijk2ras)
        else:
            landmarks_ras = self.landmarks

        if mode == "trans_rot_isoscale":
            t_ras2scaled = register_trans_rot_isoscale(target_landmarks, landmarks_ras)
        elif mode == "general":
            t_ras2scaled = register_general_affine(target_landmarks, landmarks_ras)
        else:
            raise ValueError(f"unexpected mode '{mode}'")

        t_ijk2scaled = t_ras2scaled @ self.t_ijk2ras
        t_scaled2ijk = xrutils.pinv(t_ijk2scaled)

        result = self.apply_transform(t_ijk2scaled)
        result.t_ijk2ras = t_ijk2scaled
        result.t_ras2ijk = t_scaled2ijk

        return result

    def scale_to_headsize(
        self,
        circumference: cdt.QLength,
        nz_cz_iz: cdt.QLength,
        lpa_cz_rpa: cdt.QLength,
    ) -> "VoxelHeadModel":
        """Scale the head model to anthropometric measurements.

        Fits an ellipsoid to the three measurements, derives landmarks,
        and delegates to :meth:`scale_to_landmarks`.

        Args:
            circumference: Head circumference.
            nz_cz_iz: Nasion–Cz–Inion arc length.
            lpa_cz_rpa: Left preauricular–Cz–right preauricular arc length.
        """
        from cedalion.geometry.ellipsoid import get_landmarks_for_headsize

        ellipsoid_landmarks = get_landmarks_for_headsize(
            circumference, nz_cz_iz, lpa_cz_rpa
        )
        return self.scale_to_landmarks(ellipsoid_landmarks)

    def get_brain_mni152_coords(self) -> xr.DataArray:
        """MNI152 coordinates per brain voxel — not implemented for voxel models.

        Voxel-based head models do not carry per-voxel ``mni152_*`` vertex
        coordinates the way the surface-based atlases do.  To get MNI
        positions, transform the brain voxels with an appropriate affine,
        e.g. ``head.brain.apply_transform(t_ijk2mni)``.
        """
        raise NotImplementedError(
            "VoxelHeadModel does not carry per-voxel MNI coordinates. Use "
            "`self.brain.apply_transform(t_ijk2mni)` to obtain MNI positions."
        )

    @cdc.validate_schemas
    def align_and_snap_to_scalp(
        self,
        points: cdt.LabeledPoints,
        mode: str = "general",
    ) -> cdt.LabeledPoints:
        """Align and snap optodes/points to the scalp surface.

        Delegates to the module-level helper in :mod:`cedalion.dot.head_model`.
        """
        from cedalion.dot.head_model import align_and_snap_to_scalp

        return align_and_snap_to_scalp(self.scalp, self.landmarks, points, mode=mode)

    @cdc.validate_schemas
    def snap_to_scalp_voxels(
        self, points: cdt.LabeledPoints
    ) -> cdt.LabeledPoints:
        """Snap optodes/points to the closest scalp voxel.

        Delegates to the module-level helper in :mod:`cedalion.dot.head_model`.
        """
        from cedalion.dot.head_model import snap_to_scalp_voxels

        return snap_to_scalp_voxels(
            self.scalp, self.voxel_to_vertex_scalp, self.segmentation_masks, points
        )
