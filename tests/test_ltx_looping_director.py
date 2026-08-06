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

    def test_seconds_schedule_matches_tile_helper_defaults(self):
        frame_count, tile_size, overlap, chunks, references = self.module._calculate_schedule(
            24, 48, 10, 2
        )
        self.assertEqual((frame_count, tile_size, overlap), (1145, 240, 48))
        self.assertEqual(len(chunks), 6)
        self.assertEqual(references, [0, 216, 408, 600, 792, 984, 1144])

    def test_schema_default_has_one_prompt_per_sampler_tile(self):
        schema = self.director.define_schema()
        timeline_input = next(item for item in schema.inputs if item.id == "timeline_data")
        data, prompts, _, chunks, frame_count, tile_size, overlap = self.director._validate(
            timeline_input.default,
            24,
            48,
            10,
            2,
        )
        self.assertEqual(len(prompts), len(chunks))
        self.assertEqual((frame_count, tile_size, overlap), (1145, 240, 48))
        self.assertEqual(data["version"], 1)

    def test_conditionings_keep_global_and_tile_prompt_scoped(self):
        with mock.patch.object(
            self.director,
            "_encode",
            side_effect=lambda clip, text, frame_rate: text,
        ) as encode:
            conditionings = self.director._build_conditionings(
                object(),
                "global prompt",
                ["tile one", "tile two"],
                [(0, 30), (22, 53)],
                24,
            )

        self.assertEqual(
            conditionings,
            ["global prompt\n\ntile one", "global prompt\n\ntile two"],
        )
        self.assertEqual(encode.call_count, 2)

    def test_invalid_timing_and_prompt_count_fail_clearly(self):
        timeline = '{"version":1,"tile_prompts":[""],"keyframes":[]}'
        with self.assertRaisesRegex(ValueError, "expected 6 tile prompts"):
            self.director._validate(timeline, 24, 48, 10, 2)

        with self.assertRaisesRegex(ValueError, "timing values"):
            self.director._validate('{"tile_prompts":[]}', 0, 48, 10, 2)

    def test_keyframes_are_sorted_and_frame_zero_is_the_start_image(self):
        colors = {"late.png": 0.8, "start.png": 0.2, "end.png": 0.5}

        def load_keyframe(image_file):
            return torch.full((1, 32, 32, 3), colors[image_file])

        with mock.patch.object(self.module, "_load_keyframe", side_effect=load_keyframe):
            images, indices, start_image = self.director._build_keyframes(
                [
                    {"frame": 80, "imageFile": "late.png"},
                    {"frame": 0, "imageFile": "start.png"},
                    {"frame": 88, "imageFile": "end.png"},
                ],
                121,
            )

        self.assertEqual(indices, "0,80,88")
        self.assertEqual(images.shape, (3, 32, 32, 3))
        self.assertTrue(torch.all(images[:, 0, 0, 0] == torch.tensor([0.2, 0.8, 0.5])))
        self.assertTrue(torch.all(start_image == 0.2))

    def test_start_image_defines_dimensions_and_reference_is_distinct(self):
        images = {
            "start.png": torch.full((1, 32, 64, 3), 0.2),
            "late.png": torch.full((1, 16, 16, 3), 0.8),
        }

        with mock.patch.object(self.module, "_load_keyframe", side_effect=images.__getitem__):
            cond_images, indices, start_image, width, height, reference = (
                self.director._build_keyframes_and_reference(
                    [
                        {"frame": 80, "imageFile": "late.png"},
                        {"frame": 0, "imageFile": "start.png"},
                    ],
                    121,
                    1088,
                    0,
                )
            )

        self.assertEqual(indices, "0,80")
        self.assertEqual(cond_images.shape, (2, 32, 64, 3))
        self.assertEqual((width, height), (1088, 544))
        self.assertEqual(reference.shape, (1, 1088, 2176, 3))
        self.assertTrue(torch.all(reference == 0.8))
        self.assertTrue(torch.all(start_image == 0.2))

    def test_reference_index_must_select_an_existing_keyframe(self):
        with mock.patch.object(
            self.module,
            "_load_keyframe",
            return_value=torch.zeros((1, 32, 32, 3)),
        ):
            with self.assertRaisesRegex(ValueError, "reference_keyframe_index"):
                self.director._build_keyframes_and_reference(
                    [{"frame": 0, "imageFile": "start.png"}],
                    121,
                    1088,
                    1,
                )

    def test_keyframes_require_a_frame_zero_start(self):
        with self.assertRaisesRegex(ValueError, "frame 0"):
            self.director._prepare_keyframes(
                [{"frame": 8, "imageFile": "late.png"}],
                121,
            )

    def test_no_keyframes_keep_the_t2v_empty_path(self):
        self.assertEqual(
            self.director._build_keyframes_and_reference([], 121, 1088, 0),
            (None, "", None, 0, 0, None),
        )

    def test_director_no_longer_contains_conditioning_compression(self):
        self.assertFalse(hasattr(self.module, "_compress_image"))
        self.assertFalse(hasattr(self.module, "_process_keyframe"))
