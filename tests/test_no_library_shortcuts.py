from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "motion_art_lab"
FORBIDDEN = (
    "cv2.absdiff",
    "cv2.threshold",
    "cv2.blur",
    "cv2.GaussianBlur",
    "cv2.erode",
    "cv2.dilate",
    "cv2.morphologyEx",
    "cv2.connectedComponents",
    "cv2.findContours",
    "cv2.createBackgroundSubtractorMOG2",
    "cv2.createBackgroundSubtractorKNN",
    "cv2.calcOpticalFlowPyrLK",
    "cv2.goodFeaturesToTrack",
    "cv2.resize",
    "cv2.cvtColor",
    "scipy.ndimage",
)


def test_processing_does_not_call_forbidden_library_shortcuts():
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN:
            if name in text:
                offenders.append(f"{path.name}: {name}")
        if "module1_processing" in path.parts and "import pygame" in text:
            offenders.append(f"{path.name}: import pygame")
    assert offenders == []
