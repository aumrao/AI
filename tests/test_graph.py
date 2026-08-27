import os
import sys
from pathlib import Path
import unittest
import tempfile
import shutil

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.graph import create_video_summarizer_graph
from core.utils import get_ffmpeg_exe
import subprocess


class TestGraphWorkflow(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_graph_")
        self.ffmpeg_exe = get_ffmpeg_exe()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_graph_compilation(self):
        """Ensure the LangGraph compiles and has all expected nodes and edges."""
        graph = create_video_summarizer_graph()
        self.assertIsNotNone(graph)
        nodes = graph.nodes
        for expected_node in ["prepare_media", "transcribe", "select_highlights", "extract_clips", "generate_snapshots", "concatenate_summary"]:
            self.assertIn(expected_node, nodes)


if __name__ == "__main__":
    unittest.main()
