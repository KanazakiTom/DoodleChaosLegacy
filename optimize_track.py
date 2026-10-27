import json
import os
import numpy as np
import librosa
from cma import CMAEvolutionStrategy
import subprocess
import tempfile
import sys

# ---------------------------------------------------
# Helper: run the JavaScript Line‑Rider physics engine
# ---------------------------------------------------
# The engine lives in the cloned repo "lr-core". We invoke a tiny node
# script (see ``simulate.js``) that reads a track JSON from stdin and
# prints rider states (x, y, vx, vy, offSled flag) for each tick.
# ---------------------------------------------------
def simulate_track(track_json_path: str, ticks: int = 2000) -> list:
    """Return a list of rider states for each tick.
    Each element is a dict with keys: x, y, vx, vy, offSled.
    The node script must be present at ``lr-core/run_sim.js``.
    """
    cmd = ["node", "run_sim.js", str(ticks), track_json_path]
    # The node script prints a JSON array on stdout.
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

# ---------------------------------------------------
# Track representation – list of cubic Bézier segments
# ---------------------------------------------------
class TrackGenome:
    def __init__(self, n_segments: int = 180, init_spread: float = 20.0):
        self.n = n_segments * 8  # 8 numbers per Bézier (x1,y1,x2,y2,x3,y3,x4,y4)
        self.mean = np.zeros(self.n)  # start from origin, will be shifted later
        self.sigma = init_spread
        self.n_segments = n_segments

    def decode(self, vec: np.ndarray) -> np.ndarray:
        """Convert flat vector → (segment, 4, 2) array of control points.
        Enforces continuity: the last point of segment i becomes the first
        point of segment i+1.
        """
        pts = vec.reshape(self.n_segments, 4, 2)
        # continuity constraints
        for i in range(self.n_segments - 1):
            pts[i + 1, 0] = pts[i, 3]
        return pts

# ---------------------------------------------------
# Build JSON from Bézier points (compatible with linerider.com)
# ---------------------------------------------------
def build_track_json(bezier_pts: np.ndarray, label: str = "AI_Optimised") -> dict:
    """Create a Line‑Rider ``.track.json`` dict from Bézier control points.
    Each Bézier segment is flattened into a series of tiny straight lines
    (num_samples per segment). This keeps the physics engine happy because
    it only knows about line segments.
    """
    lines = []
    line_id = 1
    for seg in bezier_pts:
        # sample the curve
        num_samples = 12
        t = np.linspace(0, 1, num_samples)[:, None]
        p0, p1, p2, p3 = seg
        points = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        # create line segments between successive points
        for a, b in zip(points[:-1], points[1:]):
            lines.append({
                "id": line_id,
                "type": 0,
                "x1": round(float(a[0]), 2),
                "y1": round(float(a[1]), 2),
                "x2": round(float(b[0]), 2),
                "y2": round(float(b[1]), 2),
                "flipped": False,
                "leftExtended": False,
                "rightExtended": False,
            })
            line_id += 1
    track = {
        "label": label,
        "creator": "Antigravity_Optimiser",
        "version": "6.2",
        "startPosition": {"x": 0, "y": 0},
        "riders": [{"startPosition": {"x": 0, "y": 0}, "startVelocity": {"x": 0.4, "y": 0}, "remountable": True}],
        "lines": lines,
    }
    return track

# ---------------------------------------------------
# Loss function – combines beat‑error, crash penalty, smoothness
# ---------------------------------------------------
def compute_loss(vec: np.ndarray, beats: np.ndarray, beat_xy: np.ndarray, simulator) -> float:
    genome = TrackGenome()
    bez = genome.decode(vec)
    track_dict = build_track_json(bez)
    # write temporary json for the JS simulator
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tf:
        json.dump(track_dict, tf, indent=2)
        tmp_path = tf.name
    # Run physics simulation (synchronously)
    try:
        states = simulator(tmp_path, ticks=int(beats[-1] * 40) + 200)  # a bit beyond last beat
    finally:
        os.unlink(tmp_path)
    # ------- beat alignment error -------
    err = 0.0
    for b_tick, (bx, by) in zip(beats * 40, beat_xy):
        t = int(round(b_tick))
        if t < len(states):
            s = states[t]
            err += np.hypot(s["x"] - bx, s["y"] - by)
    err /= len(beats)
    # ------- crash penalty -------
    crash = sum(1 for s in states if s.get("offSled", False)) * 150.0
    # ------- curvature smoothness (simple) -------
    curvature = 0.0
    for i in range(1, len(bez)):
        diff = np.linalg.norm(bez[i, 0] - bez[i - 1, 3])
        curvature += diff
    curvature = curvature / len(bez)
    # Combine
    return err * 12.0 + crash + curvature * 0.3

# ---------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------
def optimise(audio_path: str, out_path: str = "optimised.track.json", duration: float = 65.0, n_segments: int = 200):
    print("Loading audio and extracting beats …")
    y, sr = librosa.load(audio_path, sr=22050, duration=duration)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # convert beats to (x, y) positions – we use a simple flat‑ground model
    # where the ground is at y = 0 and x = tick * 8 (approx 8 px per tick)
    beat_xy = np.column_stack((beat_times * 40 * 8.0, np.zeros_like(beat_times)))

    genome = TrackGenome(n_segments=n_segments)
    es = CMAEvolutionStrategy(genome.mean, genome.sigma, {
        "popsize": 48,
        "maxfevals": 12000,
    })

    # optimisation loop
    while not es.stop():
        candidates = es.ask()
        scores = []
        for vec in candidates:
            loss = compute_loss(np.array(vec), beat_times, beat_xy, simulate_track)
            scores.append(loss)
        es.tell(candidates, scores)
        es.disp()

    best_vec = es.result.xbest
    best_bez = genome.decode(np.array(best_vec))
    final_track = build_track_json(best_bez, label="AI_Optimised_Beethoven5th")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_track, f, indent=2)
    print(f"✅ Optimisation finished – track written to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CMA‑ES optimiser for Line‑Rider tracks synced to music")
    parser.add_argument("--audio", required=True, help="Path to the MP3 (or OGG) file")
    parser.add_argument("--out", default="optimised.track.json", help="Output .track.json file")
    parser.add_argument("--duration", type=float, default=65.0, help="How many seconds of audio to consider")
    parser.add_argument("--segments", type=int, default=200, help="Number of Bézier segments (higher = finer control)")
    args = parser.parse_args()
    optimise(args.audio, args.out, args.duration, args.segments)
