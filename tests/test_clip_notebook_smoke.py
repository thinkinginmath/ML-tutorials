"""Smoke checks for the CLIP text-to-image notebook.

These checks do not download COCO or the CLIP weights. They validate the
notebook schema and the ranking helpers. A reduced end-to-end run is a
separate command: set CLIP_TUTORIAL_MAX_IMAGES=8 and execute the notebook.
The default of 200 images is the run whose scores should be quoted.
"""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Tutorial_CLIP_Text_to_Image_Search.ipynb"
MNIST_PATH = ROOT / "Tutorial_Neural_Network_From_Scatch_(MNIST_with_NumPy_only).ipynb"
MNIST_SHA256 = "e26b2b2f8b4f13e4b51204d4019f2cefb3443c0706a4a4cd0bec356bf08265b0"

HEADINGS = [
    "# Tutorial: Text-to-Image Search with a Pretrained CLIP Model",
    "## How this differs from the NumPy MNIST tutorial",
    "## Setup, installs, and what this run costs",
    "## Cosine similarity, before any photographs",
    "## The public image collection",
    "## Load CLIP and preprocess with its own processor",
    "## Embed images in batches",
    "## Embed a text query and rank by cosine similarity",
    "## Try several queries",
    "## Retrieval evaluation",
    "## Inspect a failure",
    "## Closed-set ranking is not open-ended search",
    "## Limitations",
]


def _source(cell):
    source = cell["source"]
    if isinstance(source, list):
        return "".join(source)
    return source


class NotebookSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def test_notebook_is_nbformat_4(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(len(self.notebook["cells"]), 20)
        self.assertIn("python3", self.notebook["metadata"]["kernelspec"]["name"])

    def test_original_mnist_notebook_is_unchanged(self):
        digest = hashlib.sha256(MNIST_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, MNIST_SHA256)

    def test_colab_badge_points_at_thinkinginmath_review_branch(self):
        first = _source(self.notebook["cells"][0])
        self.assertIn("thinkinginmath/ML-tutorials", first)
        self.assertIn("cursor/clip-text-to-image-search-9992", first)
        self.assertIn("Tutorial_CLIP_Text_to_Image_Search.ipynb", first)
        self.assertIn("blob/main/Tutorial_CLIP_Text_to_Image_Search.ipynb", first)
        self.assertNotIn("weidongshao", first)

    def test_headings_are_in_teaching_order(self):
        markdown = "\n".join(
            _source(cell) for cell in self.notebook["cells"] if cell["cell_type"] == "markdown"
        )
        cursor = 0
        for heading in HEADINGS:
            found = markdown.find(heading, cursor)
            self.assertGreaterEqual(found, 0, msg=f"Missing or out of order: {heading}")
            cursor = found + len(heading)

    def test_code_cells_have_no_saved_outputs_or_local_paths(self):
        code = []
        for cell in self.notebook["cells"]:
            source = _source(cell)
            self.assertNotIn("/home/ubuntu", source)
            self.assertNotIn("/workspace", source)
            self.assertNotIn("/content", source)
            if cell["cell_type"] != "code":
                continue
            code.append(source)
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
        joined = "\n".join(code)
        for required in (
            "torch.inference_mode()",
            "model.eval()",
            "return_dict=True",
            "pooler_output",
            "openai/clip-vit-base-patch32",
            "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "CLIP_TUTORIAL_MAX_IMAGES",
            "a dog playing on the beach",
            "def l2_normalize",
            "def true_pair_ranks",
        ):
            self.assertIn(required, joined)
        self.assertIn("https://download.pytorch.org/whl/cpu", joined)
        self.assertNotIn("pip install torch torchvision", joined)

    def test_ranking_helpers_match_the_notebook(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed; schema checks already ran.")
        helper = None
        for cell in self.notebook["cells"]:
            source = _source(cell)
            if source.startswith("# CLIP_TUTORIAL_PURE_HELPERS"):
                helper = source
                break
        self.assertIsNotNone(helper)
        namespace = {"torch": torch}
        exec(compile(helper, "clip_helpers", "exec"), namespace)

        unit = namespace["l2_normalize"](torch.tensor([[3.0, 0.0], [0.0, 4.0], [3.0, 4.0]]))
        scores = namespace["cosine_similarity_matrix"](unit[:1], unit[1:])
        self.assertAlmostEqual(float(scores[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(scores[0, 1]), 0.6, places=5)

        paired = torch.tensor(
            [
                [0.9, 0.9, 0.1],
                [0.2, 0.8, 0.1],
                [0.4, 0.4, 0.4],
            ]
        )
        ranks = namespace["true_pair_ranks"](paired)
        self.assertEqual(ranks.tolist(), [1, 0, 2])
        self.assertAlmostEqual(namespace["recall_at_k"](ranks, 1), 1 / 3, places=5)
        self.assertAlmostEqual(namespace["chance_recall_at_k"](5, 200), 0.025, places=6)
        self.assertAlmostEqual(namespace["chance_median_rank"](200), 100.5, places=6)
        self.assertAlmostEqual(
            namespace["multilabel_recall_at_k"](["dog", "cat", "car"], {"dog", "boat"}, 2),
            0.5,
            places=6,
        )

        ids = list(range(100))
        first = namespace["choose_image_ids"](ids, 8, seed=0)
        second = namespace["choose_image_ids"](list(reversed(ids)), 8, seed=0)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 8)
        self.assertTrue(set(first) <= set(ids))
        self.assertEqual(namespace["validate_query"]("  a dog playing on the beach  "), "a dog playing on the beach")
        self.assertEqual(namespace["validate_top_k"](5, 200), 5)
        with self.assertRaises(ValueError):
            namespace["validate_query"]("   ")
        with self.assertRaises(ValueError):
            namespace["validate_top_k"](9, 200)
        with self.assertRaises(ValueError):
            namespace["l2_normalize"](torch.zeros(4))


if __name__ == "__main__":
    unittest.main()
