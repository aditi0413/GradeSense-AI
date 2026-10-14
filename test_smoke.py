import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

from gradesense.views import PAGES  # noqa: E402


def test_every_page_renders():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240).run()
    assert not at.exception
    for name in PAGES:
        at.sidebar.radio[0].set_value(name).run()
        assert not at.exception, f"{name}: {at.exception}"
