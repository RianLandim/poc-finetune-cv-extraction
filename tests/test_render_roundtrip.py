"""Every gold value must be findable in the rendered PDF's text (phase 1).

If the renderer hides a field, the gold must say null; if the gold has a value, the PDF
must show it. This is what makes the synthetic labels exact.
"""

import pytest

pytest.skip("phase 1: needs render.render_pdf and extract_text.pdf_to_text", allow_module_level=True)
