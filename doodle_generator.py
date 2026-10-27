import json
import math
import numpy as np
import librosa

def bezier_curve(p0, p1, p2, p3, num_samples=16):
    """Generates smoothly sampled 2D points along a cubic Bezier curve."""
    t = np.linspace(0.0, 1.0, num_samples)[:, np.newaxis]
    points = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return points

class TrackBuilder:
    def __init__(self, label="DoodleChaos_Beethoven5th"):
        self.label = label
        self.lines = []
        self.line_id = 1
        self.riders = []

    def add_rider(self, x=0.0, y=0.0, vx=0.0, vy=0.0):
        self.riders.append({
            "startPosition": {"x": round(float(x), 2), "y": round(float(y), 2)},
            "startVelocity": {"x": round(float(vx), 2), "y": round(float(vy), 2)},
            "remountable": True
        })

    def add_line(self, x1, y1, x2, y2, line_type=0):
        # 0 = Blue standard physics, 1 = Red acceleration boost, 2 = Green scenery
        line = {
            "id": self.line_id,
            "type": int(line_type),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
            "flipped": False,
            "leftExtended": False,
            "rightExtended": False
        }
        self.lines.append(line)
        self.line_id += 1
        return line

    def add_path(self, points, line_type=0, boost_fn=None):
        """Adds a continuous sequence of line segments sharing exact vertices."""
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            ltype = line_type
            if boost_fn is not None and boost_fn(i, p1, p2):
                ltype = 1
            self.add_line(p1[0], p1[1], p2[0], p2[1], line_type=ltype)

    def export_json(self, output_path):
        track_data = {
            "label": self.label,
            "creator": "Antigravity_DoodleChaos_AI",
            "version": "6.2",
            "startPosition": self.riders[0]["startPosition"] if self.riders else {"x": 0, "y": 0},
            "riders": self.riders,
            "lines": self.lines
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(track_data, f, indent=2)
        print(f"Exported track with {len(self.lines)} lines and {len(self.riders)} riders to {output_path}")

def generate_doodle_track(audio_path: str, output_json: str = "music_sync.track.json", duration_sec: float = 60.0):
    print(f"Loading {audio_path} (first {duration_sec}s)...")
    y, sr = librosa.load(audio_path, sr=22050, duration=duration_sec)
    
    # 1. Feature Extraction
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=4, delta=0.4, wait=3)
    peak_times = librosa.frames_to_time(peaks, sr=sr)
    peak_ticks = np.round(peak_times * 40).astype(int)
    
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=256)
    rms_norm = rms / (np.max(rms) + 1e-6)

    def get_energy_at_tick(tick):
        sec = tick / 40.0
        idx = np.searchsorted(rms_times, sec)
        idx = min(idx, len(rms_norm) - 1)
        return float(rms_norm[idx])

    print(f"Detected {len(peak_ticks)} rhythmic note peaks.")

    track = TrackBuilder("DoodleChaos_Beethoven_5th")

    # Physics Constants
    GRAVITY = 0.175 # px / tick^2
    
    # ==========================================
    # RIDER 1: LEAD MELODY TRACK (Top Track)
    # ==========================================
    # Rider 1 starts at (0, 0)
    track.add_rider(x=0.0, y=0.0, vx=0.4, vy=0.0)

    # --- Section 1: Starting drop to first note ---
    # Sled needs downward slope to gain initial speed
    p_start = np.array([-40.0, 5.0])
    p_mid = np.array([0.0, 12.0])
    p_drop = np.array([120.0, 85.0])
    
    # Smooth start hill with red acceleration lines
    pts_start = bezier_curve(p_start, np.array([-10.0, 5.0]), np.array([30.0, 20.0]), p_drop, num_samples=18)
    track.add_path(pts_start, line_type=0, boost_fn=lambda i, p1, p2: p1[0] > -10.0 and p1[0] < 80.0)

    curr_pt = pts_start[-1]

    # --- Section 2: Phrase 1 Rhythm Bumps (Da-Da-Da) ---
    # Three bumps leading to the big 4th note launch
    bump_widths = [75.0, 70.0, 70.0]
    bump_heights = [18.0, 22.0, 25.0]

    for b_idx in range(3):
        w = bump_widths[b_idx]
        h = bump_heights[b_idx]
        p0 = curr_pt
        p1 = p0 + np.array([w * 0.3, 10.0]) # dip into trough
        p2 = p0 + np.array([w * 0.7, -h])   # climb up crest
        p3 = p0 + np.array([w, -h * 0.4])   # exit crest
        
        pts_bump = bezier_curve(p0, p1, p2, p3, num_samples=14)
        # Red boost on the climb of each bump
        track.add_path(pts_bump, line_type=0, boost_fn=lambda i, p1, p2: i >= 6 and i <= 10)
        curr_pt = pts_bump[-1]

    # --- Section 3: The Big Eb Launch Ramp (Note 4 - "DAAAA" Hold) ---
    # Launches Rider 1 high into the air
    ramp_dx = 80.0
    p0 = curr_pt
    p1 = p0 + np.array([ramp_dx * 0.4, 5.0])
    p2 = p0 + np.array([ramp_dx * 0.8, -25.0])
    p3 = p0 + np.array([ramp_dx, -45.0])
    pts_launch = bezier_curve(p0, p1, p2, p3, num_samples=16)
    # Give high red acceleration boost off the launch lip
    track.add_path(pts_launch, line_type=0, boost_fn=lambda i, p1, p2: i >= 8)

    launch_pt = pts_launch[-1]
    launch_vx = 14.5
    launch_vy = -7.5
    air_ticks = 48 # ~1.2s flight during the held Eb chord

    # Parabolic flight calculation
    land_dx = launch_vx * air_ticks
    land_dy = launch_vy * air_ticks + 0.5 * GRAVITY * (air_ticks ** 2)
    land_pt = launch_pt + np.array([land_dx, land_dy])
    land_vx = launch_vx
    land_vy = launch_vy + GRAVITY * air_ticks
    land_angle = math.atan2(land_vy, land_vx)

    # --- Section 4: Tangent-Matched Catch Scoop Ramp ---
    # Catch ramp aligns with velocity vector and scoops smoothly into flat/descent
    catch_start = land_pt - np.array([12.0 * math.cos(land_angle), 12.0 * math.sin(land_angle) - 4.0])
    catch_len = 160.0
    c0 = catch_start
    c1 = c0 + np.array([catch_len * 0.35 * math.cos(land_angle), catch_len * 0.35 * math.sin(land_angle)])
    c2 = c0 + np.array([catch_len * 0.75, 20.0])
    c3 = c0 + np.array([catch_len, 10.0])
    pts_catch = bezier_curve(c0, c1, c2, c3, num_samples=22)
    track.add_path(pts_catch, line_type=0)
    curr_pt = pts_catch[-1]

    # Decorative scenery underneath the flight: musical notes & staff arcs (Green lines)
    for staff_i in range(5):
        staff_offset = 60.0 + staff_i * 12.0
        s0 = launch_pt + np.array([0.0, staff_offset])
        s1 = launch_pt + np.array([land_dx * 0.33, staff_offset + 30.0])
        s2 = launch_pt + np.array([land_dx * 0.66, staff_offset + 30.0])
        s3 = land_pt + np.array([0.0, staff_offset])
        pts_staff = bezier_curve(s0, s1, s2, s3, num_samples=20)
        track.add_path(pts_staff, line_type=2)

    # ==========================================
    # RIDER 2: COUNTERPOINT BASS TRACK (Bottom Track)
    # ==========================================
    # Rider 2 joins the performance for Phrase 2 ("Da-Da-Da-DAAA" in lower pitch)
    # Starts positioned to synchronize with Phrase 2 at tick 124 (~3.1s)
    r2_start_x = curr_pt[0] - 120.0
    r2_start_y = curr_pt[1] + 160.0
    track.add_rider(x=r2_start_x, y=r2_start_y, vx=0.4, vy=0.0)

    # Rider 2 Start Hill & Rhythm Bumps for Phrase 2
    r2_p0 = np.array([r2_start_x - 40.0, r2_start_y + 5.0])
    r2_p1 = np.array([r2_start_x, r2_start_y + 12.0])
    r2_p2 = np.array([r2_start_x + 60.0, r2_start_y + 40.0])
    r2_p3 = np.array([r2_start_x + 120.0, r2_start_y + 70.0])
    pts_r2_start = bezier_curve(r2_p0, r2_p1, r2_p2, r2_p3, num_samples=18)
    track.add_path(pts_r2_start, line_type=0, boost_fn=lambda i, p1, p2: i >= 6)

    curr_r2 = pts_r2_start[-1]

    # Rider 2 Rhythm Bumps (Da-Da-Da for Phrase 2)
    for b_idx in range(3):
        w = 80.0
        h = 24.0 + b_idx * 5.0
        p0 = curr_r2
        p1 = p0 + np.array([w * 0.3, 12.0])
        p2 = p0 + np.array([w * 0.7, -h])
        p3 = p0 + np.array([w, -h * 0.3])
        pts_b = bezier_curve(p0, p1, p2, p3, num_samples=14)
        track.add_path(pts_b, line_type=0, boost_fn=lambda i, p1, p2: i >= 5 and i <= 9)
        curr_r2 = pts_b[-1]

    # Rider 2 Big Launch (Note 8 - Sustained D chord)
    r2_launch_dx = 90.0
    p0 = curr_r2
    p1 = p0 + np.array([r2_launch_dx * 0.4, 8.0])
    p2 = p0 + np.array([r2_launch_dx * 0.8, -30.0])
    p3 = p0 + np.array([r2_launch_dx, -55.0])
    pts_r2_launch = bezier_curve(p0, p1, p2, p3, num_samples=18)
    track.add_path(pts_r2_launch, line_type=0, boost_fn=lambda i, p1, p2: i >= 8)

    r2_launch_pt = pts_r2_launch[-1]
    r2_vx = 15.2
    r2_vy = -8.2
    r2_air_ticks = 54 # Long majestic flight

    r2_land_dx = r2_vx * r2_air_ticks
    r2_land_dy = r2_vy * r2_air_ticks + 0.5 * GRAVITY * (r2_air_ticks ** 2)
    r2_land_pt = r2_launch_pt + np.array([r2_land_dx, r2_land_dy])
    r2_land_vy = r2_vy + GRAVITY * r2_air_ticks
    r2_angle = math.atan2(r2_land_vy, r2_vx)

    # Rider 2 Landing Catch Scoop
    c0 = r2_land_pt - np.array([12.0 * math.cos(r2_angle), 12.0 * math.sin(r2_angle) - 4.0])
    c1 = c0 + np.array([180.0 * 0.35 * math.cos(r2_angle), 180.0 * 0.35 * math.sin(r2_angle)])
    c2 = c0 + np.array([180.0 * 0.75, 25.0])
    c3 = c0 + np.array([180.0, 10.0])
    pts_r2_catch = bezier_curve(c0, c1, c2, c3, num_samples=24)
    track.add_path(pts_r2_catch, line_type=0)
    curr_r2 = pts_r2_catch[-1]

    # Connect Rider 1 through Phrase 2 with a grand sweeping bridge
    bridge_dx = curr_r2[0] - curr_pt[0] + 50.0
    p0 = curr_pt
    p1 = p0 + np.array([bridge_dx * 0.3, -30.0])
    p2 = p0 + np.array([bridge_dx * 0.7, 40.0])
    p3 = p0 + np.array([bridge_dx, 10.0])
    pts_bridge = bezier_curve(p0, p1, p2, p3, num_samples=32)
    track.add_path(pts_bridge, line_type=0, boost_fn=lambda i, p1, p2: i % 8 < 3)
    curr_pt = pts_bridge[-1]

    # ==============================================================
    # SECTION 5: ORCHESTRAL WEAVING ROLLER COASTER (Tick 248 onwards)
    # ==============================================================
    # Both riders ride side-by-side / multi-tiered roller coasters
    # synchronized to the sixteenth-note violin and cello phrases
    print("Generating orchestral roller coaster synchronized to beats...")
    
    # Filter peaks for Section 5 (from tick 248 up to duration)
    sec5_peaks = [tk for tk in peak_ticks if tk >= 240 and tk < int(duration_sec * 40)]
    
    # If peaks are sparse or dense, space roller coaster waves gracefully
    prev_tick = 240
    for idx, tk in enumerate(sec5_peaks):
        dtick = tk - prev_tick
        if dtick < 5:
            continue
        prev_tick = tk
        energy = get_energy_at_tick(tk)
        
        # Roller coaster wave length and height driven by music
        wave_dx = max(65.0, min(140.0, dtick * 11.5))
        wave_amp = 18.0 + energy * 38.0
        
        # Upper track (Rider 1 - Melody): rapid hops, loops, and dips
        p0 = curr_pt
        p1 = p0 + np.array([wave_dx * 0.28, wave_amp * 0.9])   # valley trough
        p2 = p0 + np.array([wave_dx * 0.72, -wave_amp * 0.8])  # crest rise
        p3 = p0 + np.array([wave_dx, 0.0])
        pts_wave1 = bezier_curve(p0, p1, p2, p3, num_samples=16)
        # Red boost on high energy downbeats
        track.add_path(pts_wave1, line_type=0, boost_fn=lambda i, p1, p2: energy > 0.55 and i >= 8)
        curr_pt = pts_wave1[-1]

        # Lower track (Rider 2 - Bass Harmony): counterpoint wave
        r0 = curr_r2
        r1 = r0 + np.array([wave_dx * 0.32, -wave_amp * 0.7])  # opposite phase!
        r2 = r0 + np.array([wave_dx * 0.68, wave_amp * 0.85])
        r3 = r0 + np.array([wave_dx, 0.0])
        pts_wave2 = bezier_curve(r0, r1, r2, r3, num_samples=16)
        track.add_path(pts_wave2, line_type=0, boost_fn=lambda i, p1, p2: energy > 0.6 and i <= 6)
        curr_r2 = pts_wave2[-1]

        # Visual Scenery: Measure markers & stave lines (Green lines)
        if idx % 4 == 0:
            # Measure vertical bar
            track.add_line(curr_pt[0], curr_pt[1] + 40.0, curr_pt[0], curr_pt[1] + 110.0, line_type=2)
            # Rhythmic energy burst indicator
            burst_len = 10.0 + energy * 25.0
            track.add_line(curr_pt[0], curr_pt[1] - 15.0, curr_pt[0], curr_pt[1] - 15.0 - burst_len, line_type=2)

    # Final safe runout slope
    for (pt, name) in [(curr_pt, "Rider 1"), (curr_r2, "Rider 2")]:
        p0 = pt
        p1 = p0 + np.array([120.0, 15.0])
        p2 = p0 + np.array([240.0, 20.0])
        p3 = p0 + np.array([360.0, 20.0])
        pts_end = bezier_curve(p0, p1, p2, p3, num_samples=20)
        track.add_path(pts_end, line_type=0)

    # Export to JSON
    track.export_json(output_json)
    print(f"Generated track successfully for '{audio_path}'!")

if __name__ == "__main__":
    generate_doodle_track("beethoven_5th.mp3", "music_sync.track.json", duration_sec=65.0)
