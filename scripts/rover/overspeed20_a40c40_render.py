"""Render A40/C40 from recorded states with original URDF visuals."""
from pathlib import Path
import speed20_render
speed20_render.common.RUN=Path('/home/agilex/wzy/rover/runs/overspeed20_A40_C40_20261111')
speed20_render.MEDIA=Path('/home/agilex/wzy/videos/review_high_stairs/overspeed20_A40_C40_20261111')
speed20_render.main()
