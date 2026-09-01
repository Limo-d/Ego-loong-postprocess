# LingLong-H model asset provenance

These files were copied from the user-provided local robot description on
2026-09-01:

```text
/home/lenovo/linglong-h/urdf/linglong-h.urdf
/home/lenovo/linglong-h/meshes/*.STL
```

Only the runtime dependencies of the URDF are included: one URDF and 21 STL
meshes. Development scripts, duplicate meshes, generated decimated OBJ files,
and the source repository's auxiliary MuJoCo scenes are intentionally omitted.

The runtime URDF references the original wrist-yaw meshes. Camera alignment is
implemented with mirrored terminal wrist-yaw joint angles; no derived wrist
mesh is loaded. The OmniPickers remain at the original flange positions and
use per-side mount orientations to stay horizontal.

The source directory did not contain a license file. Keeping these assets in
this repository records the exact model used by the local simulation, but does
not establish permission for external redistribution. Confirm ownership and
distribution terms before publishing the repository or its model assets.
