# Generic-forward public-twiddle code-only reproducer

Prepared for validation-only closeout; not yet captured or armed. Native screen
17407055 passed. Matched small17426203 and its gated medium17426634 are pending.
Capture requires their passing results, exact source/input/reference checks,
and rederived matched gates. No generic reproduction job is submitted yet.

The package contains only hash-verified code and patches, including the ordered
public-twiddle parent generator and the generic-forward extension generator.
Builds, keys, datasets, caches, logs and profiles remain in scratch. It uses real
production images, unchanged parameters/tolerances and trust boundaries, and
small128 before medium256. All native cwd/source/cache paths are scratch-backed;
core dumps and bytecode are disabled and artifact creation is owner-only.

Finalization stays unarmed until capture passes. A fresh build and matching
small/medium replay are required before successful-only registry promotion.
Numerical success is not a cryptographic or system-wide side-channel proof.
The original scoped source/compiled-control review limitations remain in force.

The user authorized current-candidate validation only. Do not use this recipe
as authorization for new optimization ideas; stop after current candidates are
resolved and the registry is updated.
