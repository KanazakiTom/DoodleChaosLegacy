import argparse
import sys
from doodle_generator import generate_doodle_track

def main():
    parser = argparse.ArgumentParser(description="AI DoodleChaos-Style Music Track Generator for Line Rider")
    parser.add_argument("--audio", type=str, default="beethoven_5th.mp3", help="Path to input audio file")
    parser.add_argument("--output", type=str, default="music_sync.track.json", help="Output .track.json path")
    parser.add_argument("--duration", type=float, default=65.0, help="Track duration in seconds to generate (default: 65s)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎵 AI Line Rider - DoodleChaos Symphony Simulator 🎵")
    print(f"Audio file: {args.audio}")
    print(f"Target duration: {args.duration}s")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    generate_doodle_track(
        audio_path=args.audio,
        output_json=args.output,
        duration_sec=args.duration
    )
    print("\n✅ Generation complete! You can now load 'music_sync.track.json' into https://www.linerider.com/")

if __name__ == "__main__":
    main()
