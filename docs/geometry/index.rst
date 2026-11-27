Photogrammetry
==============

.. py:currentmodule:: cedalion.geometry.photogrammetry.anonymization

Tools for working with photogrammetry head scans.

See the `video tutorial <https://www.youtube.com/watch?v=PMBUWHnLXUo>`_ for a
walkthrough of the photogrammetry workflow.

cedalion.geometry.photogrammetry.anonymization
--------------------------------------------------

Photogrammetry scans capture the whole face, so they identify the
subject. To share or archive fNIRS head scans you have to remove the facial
geometry while keeping the parts co-registration needs: the optode-cap region
and the five anatomical landmarks (Nz, Iz, Cz, LPA, RPA).

This module does that by deletion, not blurring. It cuts away the facial
vertices and the triangles that touch them, then rebuilds the texture so the
exported image carries no face pixels either. What is gone is gone, which is
the property you want for data protection.

The entry point is :func:`anonymize_scan`. Pass it a raw scan surface and the
five landmarks and it returns the anonymized surface plus landmarks, ready for
:func:`save_anonymized_scan`. Every intermediate step is also exported, so you
can re-run or inspect a single stage without reimplementing the chain.
:func:`anonymize_scan` chains the following steps:

1. :func:`orient_y_anterior` rotates the mesh around the scanner's gravity axis
   so Y points anterior, inferred from the nasion against the upper-head
   centroid.
2. :func:`isolate_head` strips shoulders, body, and loose fragments with a
   sphere around the upper-head centroid, intersected with the largest
   connected component.
3. :func:`align_to_ctf` maps the mesh and landmarks into the CTF coordinate
   system (+X anterior, +Y left, +Z up, origin at the LPA-RPA midpoint).
   Degenerate landmarks (LPA equal to RPA, or Nz on the LPA-RPA line) raise a
   ``ValueError`` instead of producing a silent NaN transform.
4. :func:`detect_cap_boundary` finds the Z height where the EEG cap front edge
   sits so the mask can be clamped below it, with a failsafe for flush caps.
   Tune it through :class:`CapDetectionParams`.
5. :func:`face_mask_from_landmarks` unions the forward face region with two ear
   spheres, both clamped below the cap, then carves out preservation spheres
   around each landmark and a midline nasion strip.
6. :func:`delete_masked_vertices` drops the masked triangles and
   :func:`revert_to_einstar_frame` maps the result back into the scanner
   coordinate system.

A post-condition guards against pathological inputs: if masking removes more
than a configurable fraction of the mesh, :func:`anonymize_scan` raises a
``RuntimeError`` rather than handing back a near-empty surface.

Input and output stay in the scanner coordinate system that
:func:`cedalion.io.read_einstar_obj` produces (``crs="digitized"``); the CTF
frame is used only inside the pipeline. ``anonymize_scan`` checks that the
surface and the landmarks share a CRS before it starts, so a frame mismatch
fails loudly instead of corrupting the geometry. Pass ``return_frame="ctf"`` to
get the aligned frame back instead of the scanner one.

A minimal run:

.. code-block:: python

   import cedalion.io
   from cedalion.geometry.photogrammetry.anonymization import (
       anonymize_scan,
       save_anonymized_scan,
   )

   surface, landmarks = cedalion.io.read_einstar_obj("scan.obj")
   surface_anon, landmarks_anon = anonymize_scan(surface, landmarks)

   # For a textured .obj this writes scan_anon.obj + .mtl + a sanitized .jpg
   # whose face-region pixels are blacked out.
   save_anonymized_scan(surface_anon, "scan_anon.obj")

Examples
--------

.. nbgallery::

   ../examples/head_models/53_photogrammetry_anonymization
