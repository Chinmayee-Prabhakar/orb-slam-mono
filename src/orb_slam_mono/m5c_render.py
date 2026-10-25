"""Render genuine KITTI imagery and frozen segment-local SLAM outputs.

No estimator or ground-truth pose is imported here. Inlier circles are saved
RANSAC support, never independent landmark-identity certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .m5c_data import CLIPS, STORYBOARD, frustum_world, load_rows, pose_geometry, read_ply, sha256, validate_sources
from .datasets import KittiGraySequence


W, H = 1920, 1080
NAVY = (30, 25, 20)
WHITE = (245, 245, 245)
CYAN = (225, 200, 60)
GREEN = (85, 225, 105)
AMBER = (55, 185, 245)
RED = (65, 75, 230)
GREY = (145, 145, 145)
PANEL = (53, 45, 36)
FPS = 5.0


def label(canvas: np.ndarray, value: str, x: int, y: int, *, scale: float = 0.65,
          color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_overlay(gray: np.ndarray, capture: dict[str, Any], *, max_orb: int = 110) -> np.ndarray:
    """Draw only measured feature pixels from an equivalence-checked replay."""
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    keypoints = capture["keypoints_xy"]
    if keypoints:
        indices = np.linspace(0, len(keypoints)-1, min(max_orb, len(keypoints)), dtype=int)
        for index in indices:
            x, y = np.rint(keypoints[index]).astype(int)
            cv2.circle(image, (x, y), 2, CYAN, 1, cv2.LINE_AA)
    support = set(capture["ransac_inlier_indices"])
    for i, item in enumerate(capture["associations"]):
        x, y = np.rint(item["pixel_xy"]).astype(int)
        if i in support:
            cv2.circle(image, (x, y), 5, GREEN, 2, cv2.LINE_AA)
            cv2.circle(image, (x, y), 1, GREEN, -1, cv2.LINE_AA)
        else:
            cv2.circle(image, (x, y), 5, AMBER, 2, cv2.LINE_AA)
    return image


class Projection:
    """Fixed equal-unit orthographic projection for one independent map panel."""

    def __init__(self, xyz: np.ndarray, rect: tuple[int, int, int, int], margin: int = 48):
        self.x, self.y, self.w, self.h = rect
        eye = np.asarray([0.80, -0.62, -0.75], dtype=float)
        forward = -eye / np.linalg.norm(eye)
        up_ref = np.asarray([0.0, -1.0, 0.0])
        right = np.cross(forward, up_ref)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.basis = np.vstack((right, up))
        uv = np.asarray(xyz, dtype=float).reshape(-1, 3) @ self.basis.T
        if not len(uv) or not np.isfinite(uv).all():
            raise ValueError("Projection requires finite 3D points")
        low, high = uv.min(axis=0), uv.max(axis=0)
        self.mid = (low + high) / 2
        span = np.maximum(high - low, 1e-6)
        self.scale = float(min((self.w-2*margin)/span[0], (self.h-2*margin)/span[1]))

    def xy(self, xyz: np.ndarray) -> np.ndarray:
        uv = np.asarray(xyz, dtype=float).reshape(-1, 3) @ self.basis.T
        return np.rint(np.column_stack((self.x + self.w/2 + (uv[:, 0]-self.mid[0])*self.scale,
                                         self.y + self.h/2 - (uv[:, 1]-self.mid[1])*self.scale))).astype(int)


def _frustum_edges(canvas: np.ndarray, points: np.ndarray, projection: Projection,
                   color: tuple[int, int, int], thick: int = 2) -> None:
    xy = projection.xy(points)
    for corner in range(1, 5):
        cv2.line(canvas, tuple(xy[0]), tuple(xy[corner]), color, thick, cv2.LINE_AA)
    for first, second in ((1, 2), (2, 3), (3, 4), (4, 1)):
        cv2.line(canvas, tuple(xy[first]), tuple(xy[second]), color, thick, cv2.LINE_AA)


def _axes(canvas: np.ndarray, projection: Projection, unit: float = 1.5) -> None:
    origin = projection.xy(np.zeros((1, 3)))[0]
    for name, vector, color in (("+X", [unit, 0, 0], RED),
                                ("+Y", [0, unit, 0], GREEN),
                                ("+Z", [0, 0, unit], CYAN)):
        endpoint = projection.xy(np.asarray(vector))[0]
        cv2.arrowedLine(canvas, tuple(origin), tuple(endpoint), color, 2, cv2.LINE_AA, tipLength=.16)
        label(canvas, name, int(endpoint[0])+3, int(endpoint[1])-3, scale=.47, color=color)


def prepare_projection(rows: list[dict[str, Any]], segment: int, map_xyz: np.ndarray,
                       K: np.ndarray, width: int, height: int) -> tuple[Projection, Projection]:
    first, lost = CLIPS[segment]
    valid = [rows[i] for i in range(first, lost)]
    cameras = np.asarray([pose_geometry(r)[0] for r in valid])
    frustums = np.vstack([frustum_world(r, K, width, height) for r in valid])
    axes = np.asarray([[0., 0., 0.], [1.5, 0., 0.], [0., 1.5, 0.], [0., 0., 1.5]])
    trajectory_points = np.vstack((cameras, frustums, axes))
    map_points = np.vstack((map_xyz, trajectory_points))
    return (Projection(trajectory_points, (30, 692, 910, 326), 62),
            Projection(map_points, (980, 692, 910, 326), 46))


def _panel(canvas: np.ndarray, projection: Projection, title: str, map_xyz: np.ndarray | None,
           rows: list[dict[str, Any]], segment: int, frame_id: int,
           K: np.ndarray, width: int, height: int) -> None:
    x, y, w, h = projection.x, projection.y, projection.w, projection.h
    cv2.rectangle(canvas, (x, y), (x+w, y+h), PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (90, 83, 75), 1)
    if map_xyz is not None:
        for point in projection.xy(map_xyz):
            if x+4 <= point[0] < x+w-4 and y+4 <= point[1] < y+h-4:
                cv2.circle(canvas, tuple(point), 2, (165, 148, 122), -1, cv2.LINE_AA)
    first, lost = CLIPS[segment]
    accepted = [rows[i] for i in range(first, min(frame_id, lost-1)+1) if rows[i]["pose_valid"]]
    centers = np.asarray([pose_geometry(row)[0] for row in accepted]).reshape(-1, 3)
    pixels = projection.xy(centers)
    for a, b in zip(pixels[:-1], pixels[1:]):
        cv2.line(canvas, tuple(a), tuple(b), CYAN, 3, cv2.LINE_AA)
    for pixel in pixels[:-1]:
        cv2.circle(canvas, tuple(pixel), 3, (185, 170, 110), -1, cv2.LINE_AA)
    _axes(canvas, projection)
    if rows[frame_id]["pose_valid"]:
        current = rows[frame_id]
        _frustum_edges(canvas, frustum_world(current, K, width, height), projection, GREEN)
        cv2.circle(canvas, tuple(pixels[-1]), 7, GREEN, -1, cv2.LINE_AA)
    else:
        # Last accepted point only; no current-frame 3D marker or frustum.
        cv2.circle(canvas, tuple(pixels[-1]), 7, AMBER, 2, cv2.LINE_AA)
        label(canvas, f"last accepted: frame {lost-1}; no current pose",
              x+22, y+h-20, scale=.53, color=AMBER)
    cv2.rectangle(canvas, (x+1, y+1), (x+w-1, y+39), PANEL, -1)
    label(canvas, title, x+18, y+28, scale=.67, thickness=2)


def video_frame(frame_id: int, segment: int, gray: np.ndarray,
                row: dict[str, Any], capture: dict[str, Any], rows: list[dict[str, Any]],
                map_xyz: np.ndarray, projections: tuple[Projection, Projection],
                K: np.ndarray, time_seconds: float | None) -> np.ndarray:
    canvas = np.full((H, W, 3), NAVY, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (W-1, 66), (55, 47, 38), -1)
    timestamp = "unknown" if time_seconds is None else f"{time_seconds:.3f} s"
    label(canvas, f"KITTI 00  |  SOURCE FRAME {frame_id:06d}  |  t = {timestamp}", 31, 43,
          scale=.92, thickness=2)
    state = "TRACKING / ACCEPTED" if row["pose_valid"] else "TRACKING LOST / NO ACCEPTED POSE"
    label(canvas, state, 1190, 42, scale=.72,
          color=GREEN if row["pose_valid"] else RED, thickness=2)
    view = draw_overlay(gray, capture)
    image_x, image_y, image_w = 30, 80, 1860
    image_h = int(round(gray.shape[0] * image_w / gray.shape[1]))
    view = cv2.resize(view, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    canvas[image_y:image_y+image_h, image_x:image_x+image_w] = view
    cv2.rectangle(canvas, (image_x, image_y), (image_x+image_w, image_y+image_h),
                  GREEN if row["pose_valid"] else RED, 3)
    d = row["diagnostics"]
    num_orb = d.get("features_detected")
    shown = min(110, len(capture["keypoints_xy"]))
    assoc = d.get("pnp_correspondences")
    inliers = d.get("pnp_inliers")
    info = (f"SEGMENT {segment:03d}  |  ORB {num_orb if num_orb is not None else 'n/a'} "
            f"({shown} shown)  |  retained 2D-3D pairs {assoc if assoc is not None else 'n/a'} "
            f"|  RANSAC support {inliers if inliers is not None else 'n/a'}")
    cv2.rectangle(canvas, (30, 650), (1890, 685), (55, 47, 38), -1)
    label(canvas, info, 38, 676, scale=.62, thickness=2)
    label(canvas, "cyan: sampled ORB  |  green: RANSAC support  |  amber: other retained pairs",
          34, 1040, scale=.51, color=(196, 195, 185))
    if frame_id == CLIPS[segment][0]:
        label(canvas, "Accepted second-view initialization; map PnP not applicable", 34, 1065,
              scale=.50, color=AMBER)
    if not row["pose_valid"]:
        cv2.rectangle(canvas, (image_x+15, image_y+image_h-80),
                      (image_x+image_w-15, image_y+image_h-18), NAVY, -1)
        label(canvas, "TRACKING LOST - NO ACCEPTED POSE; candidate RANSAC support is rejected",
              image_x+30, image_y+image_h-39, scale=.74, color=RED, thickness=2)
        label(canvas, f"Cause: {row['failure_reason']} (source frame {frame_id})", 34, 1065,
              scale=.50, color=RED)
    _panel(canvas, projections[0], f"ESTIMATED CAMERA TRAJECTORY  |  local segment {segment:03d}",
           None, rows, segment, frame_id, K, gray.shape[1], gray.shape[0])
    _panel(canvas, projections[1], f"FINAL SEGMENT MAP - OFFLINE VIEW  |  {len(map_xyz)} real landmarks",
           map_xyz, rows, segment, frame_id, K, gray.shape[1], gray.shape[0])
    label(canvas, "Projection: fixed orthographic XYZ; local arbitrary units", 982, 1040,
          scale=.45, color=(200, 190, 175))
    label(canvas, "KITTI Vision Benchmark Suite / Geiger, Lenz, Urtasun (CVPR 2012) / CC BY-NC-SA 3.0",
          982, 1065, scale=.45, color=(180, 180, 175))
    return canvas


def render_clip(root: Path, segment: int, rows: list[dict[str, Any]],
                capture: dict[str, Any], sequence: KittiGraySequence,
                output: Path) -> dict[str, Any]:
    first, last = CLIPS[segment]
    map_xyz = read_ply(root / "artifacts/m5b-final-evaluation/run-a1/maps" / f"segment-{segment:03d}.ply")
    K = sequence.camera.matrix
    projections = prepare_projection(rows, segment, map_xyz, K,
                                     sequence.camera.width, sequence.camera.height)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 mp4v encoder unavailable; provide PNG sequence fallback")
    order = []
    for frame_id in range(first, last+1):
        sample = sequence[frame_id]
        frame = video_frame(frame_id, segment, sample.image, rows[frame_id], capture[str(frame_id)],
                            rows, map_xyz, projections, K, sample.timestamp_seconds)
        writer.write(frame)
        order.append(frame_id)
    writer.release()
    check = cv2.VideoCapture(str(output))
    fourcc = int(check.get(cv2.CAP_PROP_FOURCC))
    actual_codec = "".join(chr((fourcc >> (8*i)) & 255) for i in range(4))
    decoded = 0
    last_frame = None
    while True:
        ok, image = check.read()
        if not ok:
            break
        if image.shape != (H, W, 3):
            raise ValueError("Encoded video resolution changed")
        decoded += 1
        last_frame = image
    check.release()
    if decoded != len(order) or last_frame is None:
        raise ValueError("MP4 did not decode through its declared final frame")
    return {"path": output.relative_to(root).as_posix(), "sha256": sha256(output),
            "bytes": output.stat().st_size, "codec": actual_codec,
            "requested_encoder": "mp4v", "size_px": [W, H],
            "playback_fps": FPS, "source_hz": 10.0, "playback_note": "2x slowed visualization; not real-time processing",
            "source_ids": order, "decoded_frame_count": decoded, "decoded_final_frame": order[-1],
            "first_online_valid": first, "last_online_valid": last-1,
            "last_frame_pose_valid": rows[last]["pose_valid"], "map_segment": segment,
            "last_frame_decode_sha256": hashlib.sha256(last_frame.tobytes()).hexdigest()}


def render_map_figure(root: Path, segment: int, rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    first, lost = CLIPS[segment]
    xyz = read_ply(root / "artifacts/m5b-final-evaluation/run-a1/maps" / f"segment-{segment:03d}.ply")
    accepted = [rows[i] for i in range(first, lost)]
    centers = np.asarray([pose_geometry(r)[0] for r in accepted])
    fig = plt.figure(figsize=(16, 9), dpi=170, facecolor="#101820")
    ax = fig.add_subplot(121, projection="3d", facecolor="#101820")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="#e8ac62", s=6, alpha=.7, depthshade=False)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], c="#5fe1cf", lw=3, label="Accepted online camera centers")
    for index in (0, len(accepted)//2, len(accepted)-1):
        row = accepted[index]
        ax.scatter(*centers[index], color="#8cf2a2", s=45)
        ax.text(*(centers[index] + .22), str(row["frame_id"]), color="white", fontsize=10)
    cloud = np.vstack((xyz, centers))
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
           xlabel="local X", ylabel="local Y", zlabel="local Z")
    ax.set_box_aspect(np.maximum(high-low, .3))
    ax.view_init(elev=20, azim=-65)
    ax.set_title(f"Segment {segment:03d}: {len(xyz)} final landmarks and {len(accepted)} online poses",
                 color="white", fontsize=16, pad=25)
    ax.tick_params(colors="white", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((.08, .12, .16, 1))
        axis.pane.set_edgecolor((.25, .35, .42, 1))
        axis.label.set_color("white")
    ax.legend(facecolor="#17232e", edgecolor="#8fa4b1", labelcolor="white", loc="upper left")
    ax2 = fig.add_subplot(122, facecolor="#101820")
    ax2.scatter(xyz[:, 0], xyz[:, 2], c="#e8ac62", s=7, alpha=.6, label="Final PLY landmarks")
    ax2.plot(centers[:, 0], centers[:, 2], c="#5fe1cf", lw=3, marker=".", ms=4,
             label="Accepted camera centers")
    ax2.scatter([0], [0], c="white", marker="x", s=80, label="map anchor")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set(xlabel="local X (arbitrary units)", ylabel="local Z (arbitrary units)")
    ax2.set_title("Local X/Z inspection; no GT or metric scale", color="white", fontsize=15)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#8fa4b1")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    ax2.legend(facecolor="#17232e", edgecolor="#8fa4b1", labelcolor="white")
    fig.suptitle("Final segment map - offline visualization; no temporal map snapshots saved",
                 color="white", fontsize=19)
    fig.text(.5, .025, "KITTI Vision Benchmark Suite / Geiger, Lenz, Urtasun (CVPR 2012) / CC BY-NC-SA 3.0",
             ha="center", color="#c7d0d8", fontsize=10)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"path": output.relative_to(root).as_posix(), "sha256": sha256(output),
            "landmarks": len(xyz), "online_poses": len(accepted), "source_range": [first, lost-1],
            "segment_id": segment, "data_stage": "final offline PLY plus accepted online poses"}


def _story_card(canvas: np.ndarray, root: Path, row: dict[str, Any], capture: dict[str, Any],
                rect: tuple[int, int, int, int], subtitle: str) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x+w, y+h), PANEL, -1)
    frame_id = row["frame_id"]
    gray = cv2.imread(str(root / "data/sequences/00/image_0" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Missing source frame {frame_id}")
    im = draw_overlay(gray, capture, max_orb=70)
    iw = w-22
    ih = int(round(gray.shape[0] * iw / gray.shape[1]))
    canvas[y+48:y+48+ih, x+11:x+11+iw] = cv2.resize(im, (iw, ih))
    state = "ACCEPTED POSE" if row["pose_valid"] else ("LOST / NO POSE" if row["state"] == "lost" else "INITIALIZING / NO POSE")
    color = GREEN if row["pose_valid"] else (RED if row["state"] == "lost" else AMBER)
    label(canvas, f"SOURCE {frame_id:06d}  |  {state}", x+14, y+32, scale=.62, color=color, thickness=2)
    d = row["diagnostics"]
    segment = "none" if row["segment_id"] is None else f"{row['segment_id']:03d}"
    label(canvas, f"Map segment: {segment}; {subtitle}", x+14, y+48+ih+31, scale=.51)
    assoc = d.get("pnp_correspondences")
    inliers = d.get("pnp_inliers")
    label(canvas, f"retained pairs: {assoc if assoc is not None else 'n/a'}; RANSAC: {inliers if inliers is not None else 'n/a'}",
          x+14, y+48+ih+61, scale=.51)
    reason = row["failure_reason"] or ("two-view initialization" if d.get("initialization_pair") else "online tracking")
    label(canvas, f"{reason[:65]}", x+14, y+48+ih+90, scale=.48, color=color)


def render_storyboard(root: Path, rows: list[dict[str, Any]], capture: dict[str, Any],
                       ids: tuple[int, ...], output: Path, title: str) -> dict[str, Any]:
    cols = 3 if len(ids) == 6 else 2
    rows_n = int(np.ceil(len(ids)/cols))
    cw, ch = 680, 450
    canvas = np.full((90 + rows_n*ch + 110, 40+cols*cw, 3), NAVY, dtype=np.uint8)
    label(canvas, title, 42, 54, scale=1.05, thickness=2)
    for index, frame in enumerate(ids):
        previous = None if index == 0 else ids[index-1]
        skipped = "first selection" if previous is None or frame == previous+1 else f"IDs {previous+1}-{frame-1} omitted"
        _story_card(canvas, root, rows[frame], capture[str(frame)],
                    (35+(index % cols)*cw, 77+(index//cols)*ch, cw-20, ch-16), skipped)
    foot = "Nonconsecutive selections; no trajectory joins maps 004, 005, 006 and 007."
    label(canvas, foot, 42, canvas.shape[0]-67, scale=.70, color=AMBER, thickness=2)
    label(canvas, "122-177 and 207-251: no accepted poses. Segment 005 at 178-201 is intentionally omitted here.",
          42, canvas.shape[0]-36, scale=.57)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(output)
    return {"path": output.relative_to(root).as_posix(), "sha256": sha256(output),
            "selected_original_source_ids": list(ids), "nonconsecutive": True,
            "data_stage": "genuine KITTI frames, equivalence-checked coordinate replay and frozen online state"}


def render_all(root: Path, output: Path) -> dict[str, Any]:
    if (output / "videos").exists() or (output / "figures").exists():
        raise FileExistsError("M5-C media already exists; preserve it before a new render")
    validate_sources(root)
    capture_dir = output / "capture"
    equivalence = json.loads((capture_dir / "replay_equivalence.json").read_text(encoding="utf-8"))
    if not equivalence["passed"]:
        raise ValueError("Feature replay did not preserve frozen estimation")
    capture = json.loads((capture_dir / "coordinate_capture.json").read_text(encoding="utf-8"))
    rows = load_rows(root / "artifacts/m5b-final-evaluation/run-a1/per_frame.jsonl")
    sequence = KittiGraySequence(root / "data/sequences/00")
    output.mkdir(parents=True, exist_ok=True)
    (output / "videos").mkdir()
    (output / "figures").mkdir()
    videos = []
    figures = []
    for segment, (first, last) in CLIPS.items():
        path = output / "videos" / f"segment_{segment:03d}_tracking_{first:03d}_{last:03d}.mp4"
        videos.append(render_clip(root, segment, rows, capture, sequence, path))
        path = output / "figures" / f"segment_{segment:03d}_sparse_map.png"
        figures.append(render_map_figure(root, segment, rows, path))
    figures.append(render_storyboard(root, rows, capture, STORYBOARD,
                                     output / "figures/final_demonstration_storyboard.png",
                                     "Tracking loss and independent map restarts - selected KITTI frames"))
    figures.append(render_storyboard(root, rows, capture, (121, 122, 207, 252),
                                     output / "figures/tracking_loss_and_restart.png",
                                     "Two failure mechanisms and a later independent map"))
    manifest = {"videos": videos, "figures": figures,
                "camera_frame_annotation": "measured ORB keypoints and RANSAC support from read-only replay; support does not prove landmark identity",
                "map_stage": "final segment PLY throughout clips; future-created points are not shown as online historical snapshots",
                "video_geometry": "online T_cw camera centers and R_wc = R_cw.T; fixed orthographic projection per clip",
                "ground_truth_in_online_media": False}
    (output / "MEDIA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render_all(args.root.resolve(), args.output.resolve())
    print(json.dumps({"videos": [(v["path"], v["decoded_frame_count"]) for v in result["videos"]],
                      "figures": [v["path"] for v in result["figures"]]}, indent=2))


if __name__ == "__main__":
    main()
