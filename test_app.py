"""Headless verification for app.py (Working Plan step 1 verify gate: "app
launches locally, navigation works, no page has real content yet").

Loads the app and clicks each sidebar stage, asserting zero uncaught
exceptions on the initial load and on every page.

Reuses ONE AppTest session across all clicks rather than re-instantiating
AppTest.from_file() per page: the sibling clin-data-pipeline-scale repo
documented that this environment segfaults after repeatedly re-instantiating
fresh AppTest sessions in one process, while re-running .run() on a single
already-constructed session is stable. Re-running .run() after each sidebar
click is exactly what a real user session does anyway.

Run: python -m pytest test_app.py -v   (or: python test_app.py)
"""

from streamlit.testing.v1 import AppTest

from app import STAGES

STAGE_LABELS = [s[0] for s in STAGES]


def _radio(at: AppTest):
    """The single sidebar stage-selector radio."""
    assert len(at.radio) == 1, f"expected exactly one radio, found {len(at.radio)}"
    return at.radio[0]


def test_app_loads_and_every_stage_renders() -> None:
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception, f"initial load raised: {at.exception}"

    # The radio should expose exactly the stages we defined, in order.
    assert _radio(at).options == STAGE_LABELS

    for label in STAGE_LABELS:
        _radio(at).set_value(label).run()
        assert not at.exception, f"stage {label!r} raised: {at.exception}"


if __name__ == "__main__":
    test_app_loads_and_every_stage_renders()
    print(f"OK — app loaded and all {len(STAGE_LABELS)} stages rendered with no exceptions.")
