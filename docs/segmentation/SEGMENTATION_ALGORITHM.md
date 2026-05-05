# Segmentation algorithm: panel-by-panel explanation
Reviewed already. 


This file is meant to be populated by the panel export helper:

```python
from code_files.segmentation_code.segmentation_readme_panels import (
    save_segmentation_readme_panels,
)

save_segmentation_readme_panels(
    ctx_ilm=ilm_ctx,
    ctx_rpe=rpe_ctx,
    out_dir="docs/assets/segmentation_panels/example_volume_z0250",
    prefix="example",
)
```

The function writes cropped PNGs and a `panels_manifest.md` file. Paste the generated manifest below this introduction.

## Narrative

The current RPE segmentation algorithm is easiest to understand as a sequence of increasingly local searches:

1. Find the ILM.
2. Estimate a coarse RPE-adjacent guide using a globally smooth hypersmoother path.
3. Flatten the image to that guide.
4. Build a lower-resolution boundary-enhancement image.
5. Run dynamic programming to obtain an initial globally smooth RPE estimate.
6. Return the estimate to original coordinates.
7. Build high-resolution local differential images near the RPE complex.
8. Run paired-surface DP variants for the original, choroidal-oriented, and EZ-oriented proposals.
9. Vertically refine/align proposal lines.
10. Export compact 1D lines for downstream flattening, en-face projection, and texture analysis.

## Generated panels

Paste `panels_manifest.md` here.
