"""Render diagnostic states using the existing verified visual renderer."""
from pathlib import Path
import speed20_render

speed20_render.common.RUN=Path('/home/agilex/wzy/rover/runs/overspeed20_20261111')
speed20_render.MEDIA=Path('/home/agilex/wzy/videos/review_high_stairs/overspeed20_20261111')
speed20_render.main()
