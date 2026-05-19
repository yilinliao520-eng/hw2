from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize tracking outputs and flag possible ID switch cases.")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--crossings", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--jump-threshold", type=float, default=2.5)
    args = parser.parse_args()

    detections = pd.read_csv(args.detections)
    crossings = pd.read_csv(args.crossings)
    out = Path(args.out) if args.out else Path(args.detections).with_name("track_analysis.md")

    lines = ["# Tracking Analysis", ""]
    if detections.empty:
        lines.append("No tracked detections were found.")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(out)
        return 0

    grouped = detections.groupby("track_id")
    track_summary = grouped.agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        frames=("frame", "nunique"),
        mean_conf=("confidence", "mean"),
        cls=("class", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
    ).reset_index()
    track_summary["duration_frames"] = track_summary["last_frame"] - track_summary["first_frame"] + 1

    possible_switches = []
    for track_id, df in grouped:
        df = df.sort_values("frame").copy()
        dx = df["cx"].diff()
        dy = df["cy"].diff()
        diag = ((df["x2"] - df["x1"]) ** 2 + (df["y2"] - df["y1"]) ** 2) ** 0.5
        jump = ((dx**2 + dy**2) ** 0.5) / diag.shift(1)
        suspicious = df.loc[jump > args.jump_threshold, ["frame", "class", "cx", "cy"]]
        for _, row in suspicious.iterrows():
            possible_switches.append(
                {
                    "track_id": int(track_id),
                    "frame": int(row["frame"]),
                    "class": row["class"],
                    "reason": f"center jump > {args.jump_threshold}x previous bbox diagonal",
                }
            )

    lines.extend(
        [
            f"- Total detections: {len(detections)}",
            f"- Unique tracking IDs: {detections['track_id'].nunique()}",
            f"- Video frames covered: {detections['frame'].min()} to {detections['frame'].max()}",
            f"- Line crossings: {len(crossings)}",
            "",
            "## Per-Class Observations",
            "",
        ]
    )
    class_counts = detections.groupby("class")["track_id"].nunique().sort_values(ascending=False)
    for cls, count in class_counts.items():
        lines.append(f"- {cls}: {int(count)} unique IDs")

    lines.extend(["", "## Longest Tracks", ""])
    for _, row in track_summary.sort_values("frames", ascending=False).head(10).iterrows():
        lines.append(
            f"- ID {int(row.track_id)} ({row.cls}): frames {int(row.first_frame)}-{int(row.last_frame)}, "
            f"observed {int(row.frames)} frames, mean conf {row.mean_conf:.3f}"
        )

    lines.extend(["", "## Possible ID Switch Candidates", ""])
    if possible_switches:
        for item in possible_switches[:20]:
            lines.append(f"- ID {item['track_id']} near frame {item['frame']} ({item['class']}): {item['reason']}")
    else:
        lines.append("- No large trajectory jumps were detected by the simple heuristic.")

    lines.extend(
        [
            "",
            "## Notes for the Report",
            "",
            "- Use the annotated video to manually inspect occlusion or crossing moments.",
            "- If an object keeps the same Tracking ID before and after occlusion/crossing, record it as successful ID continuity.",
            "- If two nearby objects exchange IDs or one object receives a new ID after reappearing, record it as an ID switch or ID loss case.",
        ]
    )

    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    print(json.dumps({"possible_switches": possible_switches[:20]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
