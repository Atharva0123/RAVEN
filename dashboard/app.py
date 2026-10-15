"""Faculty-facing, low-latency RAVEN RGB/thermal preview dashboard."""
from __future__ import annotations

import hashlib
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2
import streamlit as st

from src.pipeline import RavenPipeline
from src.video.files import remove_owned_temp_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEMO_VIDEO_DIR = PROJECT_DIR / "demo_videos"
SCENES = {
    "Scene 08: Headlight Glare (RGB Vetoed)": DEMO_VIDEO_DIR / "intersection_glare_side_by_side.mp4",
    "Scene 01: Low-Light Alley (Joint Consensus)": DEMO_VIDEO_DIR / "alley_side_by_side.mp4",
    "Scene 10: Street Sidewalk (Multi-Pedestrian)": DEMO_VIDEO_DIR / "sidewalk_side_by_side.mp4",
}
MAX_PREVIEW_SIDE = 640
ACTION_STYLE = {
    "joint_consensus": ((0, 255, 0), "Joint"),
    "rgb_vetoed_thermal_dominant": ((255, 229, 0), "Thermal Veto"),
    "ir_vetoed_rgb_dominant": ((0, 0, 255), "RGB Dominant"),
    "abstain": ((0, 179, 255), "Review"),
}
PLAYBACK_TICK_SECONDS = 0.1


st.set_page_config(page_title="RAVEN Governance", page_icon="🛡️", layout="wide")


def load_preview_pipeline():
    """Load once: the dashboard uses detector-only preview mode on both feeds."""
    return RavenPipeline(
        config_path=PROJECT_DIR / "configs" / "models.yaml",
        enable_thermal=True,
        dashboard_preview=True,
        disable_threat_detector=True,
    )


def _resize_preview(frame):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= MAX_PREVIEW_SIDE:
        return frame
    scale = MAX_PREVIEW_SIDE / float(longest)
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def _box_iou(first, second):
    if first is None or second is None:
        return 0.0
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    area_b = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    return overlap / max(area_a + area_b - overlap, 1)


def _media_frame_for_clock(anchor_frame, anchor_time, source_fps, speed, now=None):
    """Map wall-clock time to the source frame due at the selected media speed."""
    now = time.monotonic() if now is None else now
    elapsed = max(0.0, now - float(anchor_time))
    return int(anchor_frame) + int(elapsed * float(source_fps) * float(speed))


def _draw_governance(rgb, thermal, results, visual_boxes, *, frame_timestamp, proposal_age):
    """Draw unique boxes in their governance colors; do not let track state recolor them."""
    seen = {"rgb": [], "thermal": []}
    visible_candidates = 0

    def paint(frame, modality, bbox, color, label, thickness=3):
        if not bbox or any(_box_iou(bbox, existing) >= .65 for existing in seen[modality]):
            return False
        seen[modality].append(bbox)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        font_scale = .5
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, 2)
        text_y = max(text_size[1] + 6, y1 - 6)
        cv2.rectangle(frame, (x1, text_y - text_size[1] - 6),
                      (x1 + text_size[0] + 8, text_y + baseline - 2), color, -1)
        cv2.putText(frame, label, (x1 + 4, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        return True

    for result in results:
        detected_at = result.get("video_timestamp_seconds")
        age = max(0.0, frame_timestamp - float(detected_at)) if detected_at is not None else 0.0
        if age > 2.0:
            continue
        candidate_drawn = False
        bbox = result.get("bbox")
        action = result.get("governance_action")
        color, short_action = ACTION_STYLE.get(action, ((235, 235, 235), "Pending"))
        label = f"ID {result.get('track_id') or '?'} | {short_action}"
        if bbox:
            candidate_drawn |= paint(rgb, "rgb", bbox, color, label)
        thermal_bbox = result.get("bbox_thermal")
        if thermal_bbox:
            candidate_drawn |= paint(thermal, "thermal", thermal_bbox, color, label)
        visible_candidates += int(candidate_drawn)

    for proposal in visual_boxes:
        if proposal_age > .75:
            continue
        modality = proposal.get("modality", "rgb")
        bbox = proposal.get("bbox")
        neutral = (230, 230, 230)
        target = thermal if modality == "thermal" else rgb
        visible_candidates += int(paint(
            target, modality, bbox, neutral,
            f"ID {proposal.get('track_id') or '?'} | Checking", thickness=2))
    return visible_candidates


def _close_capture():
    capture = st.session_state.pop("raven_capture", None)
    if capture is not None:
        capture.release()


class _LatestFrameInference:
    """Run detector work off the playback loop; keep only the newest queued frame."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.inputs = queue.Queue(maxsize=1)
        self.outputs = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True, name="raven-preview-inference")
        self.thread.start()

    def submit(self, generation, rgb, thermal, frame_index, timestamp):
        packet = (generation, rgb.copy(), thermal.copy(), frame_index, timestamp)
        try:
            self.inputs.put_nowait(packet)
        except queue.Full:
            try:
                self.inputs.get_nowait()
            except queue.Empty:
                pass
            self.inputs.put_nowait(packet)

    def poll_latest(self):
        latest = None
        while True:
            try:
                latest = self.outputs.get_nowait()
            except queue.Empty:
                return latest

    def _publish(self, result):
        try:
            self.outputs.put_nowait(result)
        except queue.Full:
            try:
                self.outputs.get_nowait()
            except queue.Empty:
                pass
            self.outputs.put_nowait(result)

    def _run(self):
        generation = None
        while True:
            try:
                packet = self.inputs.get(timeout=.25)
            except queue.Empty:
                continue
            current_generation, rgb, thermal, frame_index, timestamp = packet
            if current_generation != generation:
                self.pipeline.reset_video_state()
                generation = current_generation
            started = time.perf_counter()
            try:
                results = self.pipeline.process_frame_pair(
                    rgb, thermal,
                    metadata={"thermal_source": "thermal_imaging", "temperature_calibrated": False,
                              "video_timestamp_seconds": timestamp, "frame_index": frame_index},
                )
                visual_boxes = getattr(self.pipeline, "last_visual_boxes", [])
                result = {"generation": generation, "frame_index": frame_index,
                          "results": results, "visual_boxes": visual_boxes,
                          "inference_seconds": time.perf_counter() - started, "error": None}
            except Exception as exc:
                result = {"generation": generation, "frame_index": frame_index,
                          "results": [], "visual_boxes": [],
                          "inference_seconds": time.perf_counter() - started,
                          "error": str(exc)}
            self._publish(result)


def _set_source(path, signature, *, owned_upload=False, start_frame=0):
    if st.session_state.get("raven_source_signature") == signature:
        return
    _close_capture()
    old_upload = st.session_state.pop("raven_upload_path", None)
    remove_owned_temp_file(old_upload)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        st.session_state.raven_source_error = "This video could not be opened. Try an MP4 encoded with H.264."
        st.session_state.raven_source_signature = signature
        return
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = max(1.0, float(capture.get(cv2.CAP_PROP_FPS) or 25.0))
    if frame_count < 1:
        capture.release()
        st.session_state.raven_source_error = "The video has no readable frames."
        st.session_state.raven_source_signature = signature
        return
    start_frame = max(0, min(int(start_frame), frame_count - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    st.session_state.raven_capture = capture
    st.session_state.raven_source_path = str(path)
    st.session_state.raven_source_signature = signature
    st.session_state.raven_frame_count = frame_count
    st.session_state.raven_source_fps = fps
    st.session_state.raven_start_frame = start_frame
    st.session_state.raven_next_frame = start_frame
    st.session_state.raven_cursor = start_frame
    st.session_state.raven_last_frame = start_frame - 1
    st.session_state.raven_results = []
    st.session_state.raven_rgb = None
    st.session_state.raven_thermal = None
    st.session_state.raven_rgb_raw = None
    st.session_state.raven_thermal_raw = None
    st.session_state.raven_playing = False
    st.session_state.raven_loop_count = 0
    st.session_state.raven_generation = st.session_state.get("raven_generation", 0) + 1
    st.session_state.raven_visual_boxes = []
    st.session_state.raven_detection_frame = -1
    st.session_state.raven_inference_seconds = None
    st.session_state.raven_inference_error = None
    st.session_state.raven_last_update_at = None
    st.session_state.raven_next_frame_due = 0.0
    st.session_state.raven_playback_anchor_frame = start_frame
    st.session_state.raven_playback_anchor_time = 0.0
    st.session_state.raven_playback_speed = None
    st.session_state.raven_display_count = 0
    st.session_state.raven_unique_tracks = set()
    st.session_state.raven_recovery_ids = set()
    st.session_state.pop("raven_source_error", None)
    if owned_upload:
        st.session_state.raven_upload_path = str(path)


def _poll_worker(worker):
    """Apply the newest completed detector result without blocking playback."""
    worker_result = worker.poll_latest()
    if worker_result and worker_result["generation"] == st.session_state.raven_generation:
        st.session_state.raven_results = worker_result["results"]
        st.session_state.raven_visual_boxes = worker_result["visual_boxes"]
        st.session_state.raven_detection_frame = worker_result["frame_index"]
        st.session_state.raven_inference_seconds = worker_result["inference_seconds"]
        st.session_state.raven_inference_error = worker_result["error"]
        st.session_state.raven_last_update_at = time.monotonic()


def _process_next_frame(worker, speed, loop_video):
    capture = st.session_state.raven_capture
    frame_count = st.session_state.raven_frame_count
    frame_index = st.session_state.raven_cursor
    if frame_index >= frame_count:
        if not loop_video:
            st.session_state.raven_playing = False
            return
        start_frame = st.session_state.get("raven_start_frame", 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        st.session_state.raven_next_frame = start_frame
        st.session_state.raven_cursor = start_frame
        st.session_state.raven_loop_count += 1
        st.session_state.raven_generation += 1
        st.session_state.raven_results = []
        st.session_state.raven_visual_boxes = []
        st.session_state.raven_detection_frame = -1
        st.session_state.raven_playback_anchor_frame = start_frame
        st.session_state.raven_playback_anchor_time = time.monotonic()
        frame_index = start_frame

    next_frame = int(st.session_state.raven_next_frame)
    while next_frame < frame_index:
        ok, _ = capture.read()
        if not ok:
            break
        next_frame += 1
    ok, frame = capture.read()
    if not ok:
        if loop_video:
            start_frame = st.session_state.get("raven_start_frame", 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            st.session_state.raven_next_frame = start_frame
            st.session_state.raven_cursor = start_frame
            st.session_state.raven_loop_count += 1
            st.session_state.raven_generation += 1
            st.session_state.raven_results = []
            st.session_state.raven_visual_boxes = []
            st.session_state.raven_detection_frame = -1
            st.session_state.raven_playback_anchor_frame = start_frame
            st.session_state.raven_playback_anchor_time = time.monotonic()
        else:
            st.session_state.raven_playing = False
        return

    split_at = frame.shape[1] // 2
    rgb = _resize_preview(frame[:, :split_at].copy())
    thermal = _resize_preview(frame[:, split_at:].copy())
    timestamp = frame_index / st.session_state.raven_source_fps
    worker.submit(st.session_state.raven_generation, rgb, thermal, frame_index, timestamp)
    _poll_worker(worker)
    st.session_state.raven_rgb_raw = rgb
    st.session_state.raven_thermal_raw = thermal
    st.session_state.raven_last_frame = frame_index
    st.session_state.raven_cursor = frame_index + 1
    st.session_state.raven_next_frame = frame_index + 1
    st.session_state.raven_next_frame_due = time.monotonic()
    if st.session_state.raven_cursor >= frame_count and not loop_video:
        st.session_state.raven_playing = False


def _scene_explanation(scene_name, results, detection_frame, current_frame):
    """Translate current model outputs into plain language for demo viewers."""
    if detection_frame < 0:
        return "The detector is starting. Boxes appear after the first model update."
    if not results:
        return ("No pedestrian box was proposed on the latest detector update. "
                "That means the model did not find one; it does not prove the scene is empty.")
    counts = {}
    for row in results:
        action = row.get("governance_action", "")
        if action == "joint_consensus":
            label = "seen by both cameras"
        elif action == "rgb_vetoed_thermal_dominant":
            label = "seen by thermal; RGB was suppressed"
        elif action == "ir_vetoed_rgb_dominant":
            label = "seen by RGB; thermal was suppressed"
        elif action == "abstain" or row.get("needs_review"):
            label = "uncertain; sent for human review"
        else:
            label = "waiting for a decision"
        counts[label] = counts.get(label, 0) + 1
    summary = "; ".join(f"{n} {label}" for label, n in counts.items())
    context = {
        "Scene 08: Headlight Glare (RGB Vetoed)": "Look at the left RGB feed: headlights can wash out a person. A cyan thermal box means the right camera found the person and RAVEN relied on thermal.",
        "Scene 01: Low-Light Alley (Joint Consensus)": "In this dim alley, compare both feeds. A green box means both cameras support the same person.",
        "Scene 10: Street Sidewalk (Multi-Pedestrian)": "This clip contains several people, including small or distant pedestrians the compact detector can miss. The displayed count is the number of model proposals, not the number of people actually present.",
    }.get(scene_name, "Each box is one current pedestrian candidate.")
    return f"**What RAVEN sees now:** {summary}.  {context}"


@st.fragment(run_every=PLAYBACK_TICK_SECONDS)
def _render_playback_tick(worker, speed, loop_video, scene_name):
    """Advance at media-time cadence while leaving sidebar controls responsive."""
    capture = st.session_state.get("raven_capture")
    if capture is None:
        st.info("Choose a demo scene or upload a synchronized RGB + thermal video.")
        return

    if st.session_state.get("raven_playing"):
        now = time.monotonic()
        if st.session_state.get("raven_playback_speed") != speed:
            st.session_state.raven_playback_speed = speed
            st.session_state.raven_playback_anchor_frame = st.session_state.raven_cursor
            st.session_state.raven_playback_anchor_time = now
        anchor_frame = st.session_state.get("raven_playback_anchor_frame", st.session_state.raven_cursor)
        anchor_time = st.session_state.get("raven_playback_anchor_time", now)
        target_frame = _media_frame_for_clock(
            anchor_frame, anchor_time, st.session_state.raven_source_fps, speed, now)
        if target_frame >= st.session_state.raven_cursor:
            # If rendering falls behind, skip to the source frame scheduled by
            # media time. This avoids accumulating delay and preserves the
            # requested playback rate instead of slowly drifting slower.
            st.session_state.raven_cursor = min(target_frame, st.session_state.raven_frame_count - 1)
            _process_next_frame(worker, speed, loop_video)
        else:
            _poll_worker(worker)
    else:
        _poll_worker(worker)

    rgb_raw = st.session_state.get("raven_rgb_raw")
    thermal_raw = st.session_state.get("raven_thermal_raw")
    rgb = rgb_raw.copy() if rgb_raw is not None else None
    thermal = thermal_raw.copy() if thermal_raw is not None else None
    if rgb is not None and thermal is not None:
        frame_timestamp = max(0, st.session_state.get("raven_last_frame", 0)) / st.session_state.raven_source_fps
        detection_frame = st.session_state.get("raven_detection_frame", -1)
        proposal_age = (max(0.0, frame_timestamp - detection_frame / st.session_state.raven_source_fps)
                        if detection_frame >= 0 else float("inf"))
        st.session_state.raven_display_count = _draw_governance(
            rgb, thermal, st.session_state.get("raven_results", []),
            st.session_state.get("raven_visual_boxes", []),
            frame_timestamp=frame_timestamp, proposal_age=proposal_age)
        st.session_state.raven_rgb = rgb
        st.session_state.raven_thermal = thermal
    left, right = st.columns(2, gap="medium")
    with left:
        st.subheader("📷 Visible Camera (RGB)")
        if rgb is not None:
            st.image(rgb, channels="BGR", width="stretch", output_format="JPEG")
        else:
            st.info("Press Play to start the selected scene.")
    with right:
        st.subheader("🌡️ Thermal Camera (Infrared)")
        if thermal is not None:
            st.image(thermal, channels="BGR", width="stretch", output_format="JPEG")
        else:
            st.info("The synchronized infrared feed appears here.")

    results = st.session_state.get("raven_results", [])
    candidates = st.session_state.get("raven_display_count", 0)
    summary_cols = st.columns(4)
    summary_cols[0].metric("Current pedestrian boxes", candidates)
    status = "Playing" if st.session_state.get("raven_playing") else (
        "Finished" if st.session_state.get("raven_last_frame", -1) >= st.session_state.raven_frame_count - 1 else "Paused")
    summary_cols[1].metric("Video", status)
    summary_cols[2].metric("Speed", f"{speed:.1f}×")
    current_frame = max(0, st.session_state.get("raven_last_frame", -1) + 1)
    summary_cols[3].metric("Frame", f"{min(current_frame, st.session_state.raven_frame_count)} / {st.session_state.raven_frame_count}")
    st.caption("The box count is for the latest detector update, not a count of unique people across the whole clip. Model updates can lag slightly while inference runs.")
    st.markdown(_scene_explanation(scene_name, results,
                                   st.session_state.get("raven_detection_frame", -1),
                                   current_frame))
    if st.session_state.get("raven_inference_error"):
        st.warning(f"The latest detector update failed: {st.session_state.raven_inference_error}")
    elif st.session_state.get("raven_detection_frame", -1) >= 0:
        st.caption(f"Last model update: frame {st.session_state.raven_detection_frame + 1}. Preview inference is resized to {MAX_PREVIEW_SIDE}px per camera.")


with st.sidebar:
    st.title("RAVEN")
    scene_name = st.selectbox("Select Demo Scene", list(SCENES), key="raven_scene")
    custom_video = st.file_uploader(
        "Upload Custom Video",
        type=("mp4", "avi", "mov"),
        help="Use a synchronized side-by-side video: visible RGB on the left and thermal infrared on the right.",
        key="raven_custom_video",
    )
    speed = st.selectbox(
        "Playback Speed", (0.5, 1.0, 1.5), index=0,
        format_func=lambda v: f"{v:.1f}×",
        help="These demo clips are 10 FPS and 12 seconds long: 0.5× takes about 24 seconds, 1.0× about 12 seconds, and 1.5× about 8 seconds. Heavy inference can make playback slower, never faster than selected.",
    )
    loop_video = st.checkbox("Loop Video", value=True, key="raven_loop_video")
    start, pause, restart = st.columns(3)
    play_clicked = start.button("▶ Play", width="stretch")
    pause_clicked = pause.button("⏸ Pause", width="stretch")
    restart_clicked = restart.button("↺ Restart", width="stretch")

st.title("RAVEN: Multi-Modal Pedestrian Detection & Decision Governance")
if "raven_pipeline" not in st.session_state:
    try:
        with st.spinner("Preparing the compact detector preview…"):
            st.session_state.raven_pipeline = load_preview_pipeline()
    except Exception as exc:
        st.error(f"Could not load the preview models: {exc}")
        st.stop()
pipeline = st.session_state.raven_pipeline
if "raven_inference_worker" not in st.session_state:
    st.session_state.raven_inference_worker = _LatestFrameInference(pipeline)
worker = st.session_state.raven_inference_worker

governance = pipeline.analysis_capabilities().get("risk_governance", {})
if governance.get("available"):
    st.success("🟢 Governance Head: Active (S-DEC Config B, 867 Params)")
    st.caption("It compares the visible and thermal camera proposals, then chooses which feed to trust or asks for human review.")
else:
    st.warning("Decision system offline. Ensure the S-DEC Config B model checkpoint is installed.")

with st.container(border=True):
    st.markdown("**How to read the detection boxes**")
    st.caption("A box is a model proposal for a pedestrian, not a ground-truth annotation. No box means the detector did not propose a person in that frame.")
    legend = st.columns(4)
    legend[0].markdown("🟢 **Joint Consensus**  \nBoth RGB & thermal agree — high confidence")
    legend[1].markdown("🩵 **Thermal Dominant**  \nRGB blinded (glare / dark) — thermal used")
    legend[2].markdown("🟠 **Operator Review**  \nAmbiguous — human review recommended")
    legend[3].markdown("🔴 **RGB Dominant**  \nThermal vetoed — RGB used alone")

scene_path = SCENES[scene_name]
if custom_video is not None:
    upload_bytes = custom_video.getvalue()
    upload_signature = (custom_video.name, hashlib.sha256(upload_bytes).hexdigest())
    if st.session_state.get("raven_custom_signature") != upload_signature:
        _close_capture()
        old_upload = st.session_state.pop("raven_upload_path", None)
        remove_owned_temp_file(old_upload)
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=Path(custom_video.name).suffix or ".mp4")
        handle.write(upload_bytes)
        handle.close()
        st.session_state.raven_custom_signature = upload_signature
        _set_source(handle.name, upload_signature, owned_upload=True)
elif st.session_state.get("raven_custom_signature") is not None:
    st.session_state.pop("raven_custom_signature", None)
    old_upload = st.session_state.pop("raven_upload_path", None)
    remove_owned_temp_file(old_upload)
    st.session_state.pop("raven_source_signature", None)

if custom_video is None:
    if scene_path.is_file():
        scene_signature = (str(scene_path), scene_path.stat().st_mtime_ns, scene_path.stat().st_size)
        # Scene 10's first visible pedestrian enters at ~3 s; start there so
        # viewers see an example detection immediately instead of an empty lead-in.
        scene_start_frame = 30 if scene_name.startswith("Scene 10:") else 0
        _set_source(scene_path, scene_signature, start_frame=scene_start_frame)
    else:
        st.error(f"Demo clip not found: {scene_path}")

if play_clicked:
    st.session_state.raven_playing = True
    st.session_state.raven_playback_anchor_frame = st.session_state.raven_cursor
    st.session_state.raven_playback_anchor_time = time.monotonic()
    st.session_state.raven_playback_speed = speed
if pause_clicked:
    st.session_state.raven_playing = False
if restart_clicked:
    st.session_state.raven_playing = True
    restart_at = st.session_state.get("raven_start_frame", 0)
    st.session_state.raven_cursor = restart_at
    st.session_state.raven_next_frame = restart_at
    st.session_state.raven_last_frame = restart_at - 1
    st.session_state.raven_capture.set(cv2.CAP_PROP_POS_FRAMES, restart_at)
    st.session_state.raven_next_frame_due = 0.0
    st.session_state.raven_playback_anchor_frame = restart_at
    st.session_state.raven_playback_anchor_time = time.monotonic()
    st.session_state.raven_playback_speed = speed
    st.session_state.raven_generation += 1
    st.session_state.raven_results = []
    st.session_state.raven_visual_boxes = []
    st.session_state.raven_detection_frame = -1
    st.session_state.raven_unique_tracks = set()
    st.session_state.raven_recovery_ids = set()

if st.session_state.get("raven_source_error"):
    st.error(st.session_state.raven_source_error)
elif st.session_state.get("raven_capture") is not None:
    if st.session_state.get("raven_rgb") is None:
        _process_next_frame(worker, speed, loop_video)
    _render_playback_tick(worker, speed, loop_video, scene_name)
else:
    st.info("Choose a demo scene or upload a synchronized side-by-side RGB + thermal video.")
