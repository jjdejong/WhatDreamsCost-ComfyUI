import importlib.util
import sys
import types
from pathlib import Path
from unittest import TestCase, mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "whatdreamscost_looping_director_tests"
MODULE_NAME = f"{PACKAGE_NAME}.ltx_looping_director"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_looping_director():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]

    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package

    class Routes:
        def get(self, *args, **kwargs):
            return lambda function: function

        post = get

    class PromptServer:
        instance = types.SimpleNamespace(routes=Routes())

    server_module = types.ModuleType("server")
    server_module.PromptServer = PromptServer
    previous_server = sys.modules.get("server")
    sys.modules["server"] = server_module
    previous_argv = sys.argv[:]
    sys.argv[:] = [sys.argv[0], "--cpu"]
    import comfy.options

    previous_args_parsing = comfy.options.args_parsing
    try:
        comfy.options.args_parsing = True
        _load_module(f"{PACKAGE_NAME}.ltx_director", ROOT / "ltx_director.py")
        return _load_module(MODULE_NAME, ROOT / "ltx_looping_director.py")
    finally:
        comfy.options.args_parsing = previous_args_parsing
        sys.argv[:] = previous_argv
        if previous_server is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = previous_server


class TestLTXLoopingDirector(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_looping_director()
        cls.director = cls.module.LTXLoopingDirector

    def test_schema_default_has_one_prompt_per_sampler_chunk(self):
        schema = self.director.define_schema()
        timeline_input = next(item for item in schema.inputs if item.id == "timeline_data")

        _, prompts, _, chunks = self.director._validate(
            timeline_input.default, 241, 240, 64
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(prompts), len(chunks))

    def test_sampler_chunk_boundaries_match_looping_sampler(self):
        for frame_count, tile_size, overlap in (
            (9, 240, 64),
            (121, 240, 64),
            (241, 240, 64),
            (481, 320, 96),
        ):
            latent_frames = (frame_count - 1) // 8 + 1
            tile_frames = tile_size // 8
            overlap_frames = overlap // 8
            step = tile_frames - overlap_frames
            expected = [
                (start, min(end, latent_frames))
                for start, end in zip(
                    range(0, latent_frames + tile_frames - overlap_frames, step),
                    range(tile_frames, latent_frames + tile_frames - overlap_frames, step),
                )
            ]

            self.assertEqual(
                self.module._temporal_chunks(frame_count, tile_size, overlap), expected
            )

    def test_conditionings_keep_global_and_first_tile_prompt_scoped(self):
        with mock.patch.object(
            self.director,
            "_encode",
            side_effect=lambda clip, text, frame_rate: text,
        ) as encode:
            conditionings = self.director._build_conditionings(
                object(),
                "global prompt",
                "first tile setup",
                ["tile one", "tile two"],
                [(0, 30), (22, 53)],
                24,
            )

        self.assertEqual(
            conditionings,
            [
                "global prompt\n\nfirst tile setup\n\ntile one",
                "global prompt\n\ntile two",
            ],
        )
        self.assertEqual(len(conditionings), 2)
        self.assertEqual(encode.call_count, 2)

    def test_empty_tile_prompts_encode_as_global_only(self):
        with mock.patch.object(
            self.director,
            "_encode",
            side_effect=lambda clip, text, frame_rate: text,
        ):
            conditionings = self.director._build_conditionings(
                object(), "global prompt", "", ["", ""], [(0, 30), (22, 53)], 24
            )

        self.assertEqual(conditionings, ["global prompt", "global prompt"])

    def test_invalid_timing_and_keyframes_fail_clearly(self):
        one_tile = '{"version": 1, "tile_prompts": [""], "keyframes": []}'
        for timing, message in (
            ((10, 240, 64), r"8n\+1"),
            ((121, 241, 64), "multiple of 8"),
            ((121, 240, 240), "smaller than"),
        ):
            with self.subTest(timing=timing), self.assertRaisesRegex(ValueError, message):
                self.director._validate(one_tile, *timing)

        for frame, message in ((7, "8-frame grid"), (121, "outside frame_count")):
            timeline = (
                '{"version": 1, "tile_prompts": [""], '
                f'"keyframes": [{{"frame": {frame}, "imageFile": "ref.png"}}]}}'
            )
            with self.subTest(frame=frame), self.assertRaisesRegex(ValueError, message):
                self.director._validate(timeline, 121, 240, 64)

    def test_prompt_count_must_match_all_temporal_chunks(self):
        with self.assertRaisesRegex(ValueError, "expected 2 tile prompts"):
            self.director._validate(
                '{"version": 1, "tile_prompts": ["only one"], "keyframes": []}',
                241,
                240,
                64,
            )

    def test_keyframes_are_sorted_and_frame_zero_is_the_start_image(self):
        colors = {"late.png": 0.8, "start.png": 0.2, "end.png": 0.5}

        def load_keyframe(image_file):
            return torch.full((1, 32, 32, 3), colors[image_file])

        with (
            mock.patch.object(self.module, "_load_keyframe", side_effect=load_keyframe),
            mock.patch.object(
                self.module, "_process_keyframe", side_effect=lambda image, *args: image
            ),
        ):
            images, indices, start_image = self.director._build_keyframes(
                [
                    {"frame": 80, "imageFile": "late.png"},
                    {"frame": 0, "imageFile": "start.png"},
                    {"frame": 88, "imageFile": "end.png"},
                ],
                121,
                0,
                0,
                "maintain aspect ratio",
                32,
                0,
            )

        self.assertEqual(indices, "0,80,88")
        self.assertEqual(images.shape, (3, 32, 32, 3))
        self.assertTrue(torch.all(images[:, 0, 0, 0] == torch.tensor([0.2, 0.8, 0.5])))
        self.assertTrue(torch.all(start_image == 0.2))

    def test_no_keyframes_returns_empty_outputs(self):
        self.assertEqual(
            self.director._build_keyframes(
                [], 121, 0, 0, "maintain aspect ratio", 32, 0
            ),
            (None, "", None),
        )
