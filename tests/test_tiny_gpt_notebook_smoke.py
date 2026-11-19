"""Smoke checks for the tiny GPT notebook.

Schema and pure helpers only. This does not download Shakespeare or train a model.
A reduced end-to-end run sets TINY_GPT_TRAIN_STEPS=2. The default of 400 steps,
or 2000 when TINY_GPT_LONG=1, is the run whose losses should be quoted.
"""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Tutorial_Understand_a_Tiny_GPT.ipynb"
MNIST_PATH = ROOT / "Tutorial_Neural_Network_From_Scatch_(MNIST_with_NumPy_only).ipynb"
MNIST_SHA256 = "e26b2b2f8b4f13e4b51204d4019f2cefb3443c0706a4a4cd0bec356bf08265b0"
ORIGINAL_NOTEBOOKS = {
    "Investigating_Suspicious_Datasets.ipynb",
    "Linear_Regression_Essentials_ElasticNet.ipynb",
    "Linear_regression_with_regularization.ipynb",
    "simple_linear_regression_exercise.ipynb",
    "titanic_analysis_with_chatgpt.ipynb",
    "Tutorial_Neural_Network_From_Scatch_(MNIST_with_NumPy_only).ipynb",
}
HEADINGS = [
    "# Tutorial: Understand a Tiny GPT by Experimenting with It",
    "## Which small GPT this is",
    "## Setup and what a CPU run can show",
    "## The text",
    "## Tokens, embeddings, and positions",
    "## Causal self-attention",
    "## The rest of the block",
    "## Checks before training",
    "## Train two models for the same number of steps",
    "## Generate, and turn the temperature knob",
    "## Positions: a retrained comparison and a broken-model intervention",
    "## Context length",
    "## Look at one attention map",
    "## Limitations",
]


def _source(cell):
    source = cell["source"]
    if isinstance(source, list):
        return "".join(source)
    return source


class TinyGptNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def test_notebook_is_nbformat_4(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(len(self.notebook["cells"]), 20)

    def test_only_the_new_notebook_was_added(self):
        names = {path.name for path in ROOT.glob("*.ipynb")}
        self.assertEqual(names - ORIGINAL_NOTEBOOKS, {"Tutorial_Understand_a_Tiny_GPT.ipynb"})
        digest = hashlib.sha256(MNIST_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, MNIST_SHA256)

    def test_colab_badge_uses_the_review_branch(self):
        first = _source(self.notebook["cells"][0])
        self.assertIn("thinkinginmath/ML-tutorials", first)
        self.assertIn("cursor/tiny-gpt-character-tutorial-9992", first)
        self.assertIn("blob/main/Tutorial_Understand_a_Tiny_GPT.ipynb", first)
        self.assertNotIn("weidongshao", first)

    def test_headings_and_clean_sources(self):
        markdown = "\n".join(
            _source(cell) for cell in self.notebook["cells"] if cell["cell_type"] == "markdown"
        )
        cursor = 0
        for heading in HEADINGS:
            found = markdown.find(heading, cursor)
            self.assertGreaterEqual(found, 0, msg=f"Missing or out of order: {heading}")
            cursor = found + len(heading)
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
            "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
            "def attention_weights",
            "def train_model",
            "position_scale",
            "torch.inference_mode()",
            "F.cross_entropy",
            "TINY_GPT_LONG",
            "karpathy/char-rnn",
        ):
            self.assertIn(required, joined)
        self.assertIn("microgpt", markdown)
        self.assertIn("nanoGPT", markdown)
        self.assertIn("minGPT", markdown)

    def test_attention_and_sampling_helpers(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed; schema checks already ran.")
        helper = None
        for cell in self.notebook["cells"]:
            source = _source(cell)
            if source.startswith("# TINY_GPT_PURE_HELPERS"):
                helper = source
                break
        self.assertIsNotNone(helper)
        namespace = {}
        exec(compile(helper, "tiny_gpt_helpers", "exec"), namespace)

        weights = namespace["attention_weights"](torch.zeros(4, 4), causal=True)
        self.assertTrue(torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(weights[1], torch.tensor([0.5, 0.5, 0.0, 0.0])))
        future = torch.triu(torch.ones(4, 4), diagonal=1).bool()
        self.assertEqual(float(weights[future].sum()), 0.0)
        open_weights = namespace["attention_weights"](torch.zeros(1, 1, 4, 4), causal=False)
        self.assertGreater(float(open_weights[0, 0][future].sum()), 0.0)

        self.assertEqual(namespace["validate_temperature"](0), 0.0)
        with self.assertRaises(ValueError):
            namespace["validate_temperature"](-0.1)
        logits = torch.tensor([[0.0, 3.0, 1.0]])
        chosen = namespace["sample_next_id"](logits, temperature=0.0, generator=None)
        self.assertEqual(int(chosen[0, 0]), 1)
        row = torch.tensor([[1, 2, 3, 4, 5]])
        self.assertEqual(namespace["crop_context"](row, 3).tolist(), [[3, 4, 5]])
        with self.assertRaises(ValueError):
            namespace["encode_characters"]("ab!", {"a": 0, "b": 1})
        train, val = namespace["split_characters"]("a" * 80 + "b" * 20, 0.9)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(val), 10)


if __name__ == "__main__":
    unittest.main()
