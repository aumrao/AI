import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess
import shutil
import unittest
import tempfile

from core.utils import (
    get_ffmpeg_exe,
    seconds_to_timestamp,
    timestamp_to_seconds,
    get_video_duration
)
from core.downloader import is_valid_youtube_url
from core.llm_summarizer import (
    HighlightClip,
    optimize_highlight_boundaries,
    format_transcript_with_timestamps
)
from core.video_processor import (
    extract_single_clip,
    extract_clips,
    generate_snapshots,
    concatenate_clips
)


class TestVideoSummarizerPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_vids_")
        self.ffmpeg_exe = get_ffmpeg_exe()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ffmpeg_detected(self):
        """Verify FFmpeg is found and executable."""
        self.assertTrue(os.path.exists(self.ffmpeg_exe) or shutil.which(self.ffmpeg_exe) is not None)
        cmd = [self.ffmpeg_exe, "-version"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ffmpeg", res.stdout.lower())

    def test_timestamp_conversions(self):
        """Verify timestamp parsing and formatting."""
        self.assertEqual(seconds_to_timestamp(65), "01:05")
        self.assertEqual(seconds_to_timestamp(3665), "01:01:05")
        self.assertEqual(timestamp_to_seconds("01:05"), 65.0)
        self.assertEqual(timestamp_to_seconds("01:01:05"), 3665.0)

    def test_youtube_url_validator(self):
        """Verify YouTube URL validation regex."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
        ]
        invalid_urls = [
            "https://google.com",
            "https://vimeo.com/123456",
            "not_a_url",
        ]
        for u in valid_urls:
            self.assertTrue(is_valid_youtube_url(u), f"Failed for {u}")
        for u in invalid_urls:
            self.assertFalse(is_valid_youtube_url(u), f"Should fail for {u}")

    def test_highlight_boundary_optimization(self):
        """Verify highlight merging and boundary padding."""
        raw_highlights = [
            HighlightClip(start_time=10.0, end_time=15.0, title="Topic 1", reason="Important"),
            # Overlapping / adjacent clip
            HighlightClip(start_time=15.5, end_time=20.0, title="Topic 1 cont", reason="Continues"),
            # Distant clip
            HighlightClip(start_time=50.0, end_time=55.0, title="Topic 2", reason="Conclusion"),
        ]
        optimized = optimize_highlight_boundaries(raw_highlights, total_duration=100.0)
        
        # First two clips should be merged together
        self.assertEqual(len(optimized), 2)
        self.assertLessEqual(optimized[0].start_time, 10.0)
        self.assertGreaterEqual(optimized[0].end_time, 20.0)

    def test_synthetic_video_processing_pipeline(self):
        """Creates a synthetic test video and tests clip cutting, snapshots, and concatenation."""
        test_video_path = os.path.join(self.test_dir, "synthetic_input.mp4")
        
        # Generate a 12-second test video with audio using FFmpeg testsrc and sine wave
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=12:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-c:a", "aac",
            test_video_path
        ]
        gen_res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(gen_res.returncode, 0, f"Synthetic video generation failed: {gen_res.stderr}")
        self.assertTrue(os.path.exists(test_video_path))

        # Check duration
        duration = get_video_duration(test_video_path)
        self.assertAlmostEqual(duration, 12.0, delta=1.0)

        # Test highlights extraction
        test_highlights = [
            HighlightClip(start_time=1.0, end_time=4.0, title="Intro Segment", reason="Opening"),
            HighlightClip(start_time=7.0, end_time=10.0, title="Key Segment", reason="Core context"),
        ]

        clips_dir = os.path.join(self.test_dir, "clips")
        extracted_clips = extract_clips(test_video_path, test_highlights, clips_dir)
        self.assertEqual(len(extracted_clips), 2)
        for clip in extracted_clips:
            self.assertTrue(os.path.exists(clip))

        # Test snapshot generation
        snap_dir = os.path.join(self.test_dir, "snapshots")
        snapshots = generate_snapshots(test_video_path, test_highlights, snap_dir)
        self.assertEqual(len(snapshots), 2)
        for snap in snapshots:
            self.assertTrue(os.path.exists(snap["image_path"]))

        # Test concatenation into summary.mp4
        final_summary = os.path.join(self.test_dir, "summary.mp4")
        out = concatenate_clips(extracted_clips, final_summary)
        self.assertTrue(os.path.exists(out))

        summary_duration = get_video_duration(out)
        self.assertGreater(summary_duration, 4.0)
        self.assertLess(summary_duration, 9.0)
        print(f"Summary video created successfully! Duration: {summary_duration:.2f}s")

    def test_whisper_device_detection(self):
        """Verify detect_whisper_device runs safely without throwing unhandled exceptions."""
        from core.transcriber import detect_whisper_device
        dev, comp = detect_whisper_device()
        self.assertIn(dev, ["cpu", "cuda"])
        self.assertIn(comp, ["int8", "float16", "default"])

    def test_gemini_key_pool(self):
        """Verify GeminiKeyPool parses keys and rotates correctly."""
        from core.key_pool import GeminiKeyPool
        keys_str = "AIzaSyKey1_123456789, AIzaSyKey2_987654321 \n AIzaSyKey3_1122334455"
        parsed = GeminiKeyPool.parse_keys_str(keys_str)
        self.assertEqual(len(parsed), 3)

        pool = GeminiKeyPool(parsed)
        self.assertEqual(pool.size(), 3)
        k1 = pool.get_next_key()
        k2 = pool.get_next_key()
        self.assertNotEqual(k1, k2)

        sharded = pool.get_sharded_keys(4)
        self.assertEqual(len(sharded), 4)


if __name__ == "__main__":
    unittest.main()


