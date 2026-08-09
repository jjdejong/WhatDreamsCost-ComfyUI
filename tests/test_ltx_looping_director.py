import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "whatdreamscost_looping_director_tests"
MODULE_NAME = f"{PACKAGE_NAME}.ltx_looping_director"
SAMPLER_MODULE_NAME = f"{PACKAGE_NAME}.ltx_looping_director_sampler"


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

    @staticmethod
    def _sampler_module():
        sampler_module = sys.modules.get(SAMPLER_MODULE_NAME)
        if sampler_module is None:
            sampler_module = _load_module(
                SAMPLER_MODULE_NAME,
                ROOT / "ltx_looping_director_sampler.py",
            )
        return sampler_module

    def test_combined_sampler_exposes_two_pass_contract_without_base_sampler_inputs(self):
        sampler_module = self._sampler_module()
        schema = sampler_module.LTXLoopingDirectorSampler.define_schema()
        input_names = [item.id for item in schema.inputs]
        output_names = [item.display_name for item in schema.outputs]
        self.assertIn("pass1_sampler", input_names)
        self.assertIn("pass2_sampler", input_names)
        self.assertIn("pass1_latent", input_names)
        self.assertIn("pass2_latent", input_names)
        self.assertIn("execution_mode", input_names)
        self.assertIn("external_pass1_video_latent", input_names)
        self.assertIn("external_pass1_audio_latent", input_names)
        self.assertIn("pass1_horizontal_tiles", input_names)
        self.assertIn("pass2_horizontal_tiles", input_names)
        self.assertIn("checkpoint_policy", input_names)
        self.assertEqual(output_names, ["av_latent", "video_latent", "audio_latent"])

    def test_pass_two_from_inputs_rejects_stage_one_initialization_latent(self):
        sampler_module = self._sampler_module()
        plan = {
            "version": 1,
            "frame_count": 9,
            "frame_rate": 24,
            "chunks": [{"start": 0, "end": 2}],
            "temporal_tile_size": 2,
            "temporal_overlap": 0,
            "media": {"video_segments": [], "ic_segments": [], "audio_segments": []},
        }
        with self.assertRaisesRegex(ValueError, "external Pass 1 video latent"):
            sampler_module.LTXLoopingDirectorSampler.execute(
                plan,
                [],
                audio_vae=object(),
                pass_count=2,
                execution_mode="pass_2_from_inputs",
                pass1_latent={"samples": torch.zeros(1, 4, 2, 2, 2)},
                pass2_noise=object(),
                pass2_sampler=object(),
                pass2_sigmas=object(),
                pass2_guider=object(),
            )

    def test_checkpoint_off_skips_fingerprint_and_pass_snapshot(self):
        sampler_module = self._sampler_module()
        plan = {
            "version": 1,
            "frame_count": 9,
            "frame_rate": 24,
            "chunks": [{"start": 0, "end": 2}],
            "temporal_tile_size": 2,
            "temporal_overlap": 0,
            "stage_1_width": 64,
            "stage_1_height": 64,
            "media": {"video_segments": [], "ic_segments": [], "audio_segments": []},
        }
        video = {"samples": torch.zeros(1, 4, 2, 2, 2), "noise_mask": torch.ones(1, 1, 2, 2, 2)}
        audio = {"samples": torch.zeros(1, 1, 2, 1), "noise_mask": torch.ones(1, 1, 2, 1)}
        initial = sampler_module._concat_av(video, audio)
        with mock.patch.object(sampler_module, "_checkpoint_fingerprint", side_effect=AssertionError), \
             mock.patch.object(sampler_module.LTXLoopingDirectorSampler, "_run_pass", return_value=initial), \
             mock.patch.object(sampler_module, "_save_pass_snapshot", side_effect=AssertionError):
            output = sampler_module.LTXLoopingDirectorSampler.execute(
                plan,
                [],
                model=object(),
                video_vae=object(),
                audio_vae=object(),
                pass_count=1,
                pass1_noise=object(),
                pass1_sampler=object(),
                pass1_sigmas=object(),
                pass1_guider=object(),
                pass1_latent=initial,
                checkpoint_policy="off",
            )
        self.assertIsNotNone(output)

    def test_combined_sampler_can_create_stage_one_video_latent(self):
        sampler_module = self._sampler_module()
        latent = sampler_module.LTXLoopingDirectorSampler._empty_video({
            "stage_1_width": 64,
            "stage_1_height": 64,
            "frame_count": 9,
        })
        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 2, 2, 2))

    def test_combined_sampler_keeps_tile_guides_through_spatial_extraction(self):
        sampler_module = sys.modules.get(SAMPLER_MODULE_NAME)
        if sampler_module is None:
            sampler_module = _load_module(
                SAMPLER_MODULE_NAME,
                ROOT / "ltx_looping_director_sampler.py",
            )

        class FakeSampler:
            _director_tile_guides = [
                [{"samples": torch.zeros(1, 4, 3, 4, 5), "strength": 0.75}],
                [],
            ]

            def _director_original_extract(self, latents, guiding, *args):
                tile = lambda value: value[:, :, :, :2, :3] if value is not None else None
                return (
                    {"samples": tile(latents["samples"])},
                    {"samples": tile(guiding["samples"])},
                    None,
                    None,
                    None,
                )

            @staticmethod
            def _extract_latent_spatial_tile(latent, v_start, v_end, h_start, h_end):
                return {
                    "samples": latent["samples"][:, :, :, v_start:v_end, h_start:h_end],
                }

        result = sampler_module._extract_spatial_tile(
            FakeSampler(),
            {"samples": torch.zeros(1, 4, 3, 4, 5)},
            {"samples": torch.zeros(1, 4, 3, 4, 5)},
            None,
            None,
            None,
            0,
            2,
            0,
            3,
            1,
            1,
        )
        self.assertEqual(len(result), 5)
        self.assertEqual(len(result[1]["_director_tile_guides"]), 2)
        self.assertEqual(len(result[1]["_director_tile_guides"][0]), 1)
        self.assertEqual(result[1]["_director_tile_guides"][0][0]["samples"].shape, (1, 4, 3, 2, 3))
        self.assertEqual(result[1]["_director_tile_guides"][0][0]["strength"], 0.75)
        self.assertEqual(result[1]["_director_tile_guides"][1], [])

    def test_seconds_schedule_matches_tile_helper_defaults(self):
        frame_count, tile_size, overlap, chunks, references = self.module._calculate_schedule(
            24, 48, 10, 2
        )
        self.assertEqual((frame_count, tile_size, overlap), (1145, 240, 48))
        self.assertEqual(len(chunks), 6)
        self.assertEqual(references, [0, 232, 424, 616, 808, 1000, 1144])

    def test_default_references_sit_on_each_tile_end(self):
        """Each tile is generated "to last image", so its keyframe terminates it."""
        frame_count, tile_size, overlap, chunks, references = self.module._calculate_schedule(
            24, 48, 10, 2
        )
        self.assertEqual(references[0], 0, "the first keyframe is the start image")
        final_index = ((frame_count - 1) // 8) * 8
        for tile_index, (latent_start, latent_end) in enumerate(chunks):
            expected = min(latent_end * 8 - 8, final_index)
            self.assertEqual(
                references[tile_index + 1],
                expected,
                f"tile {tile_index} keyframe must sit on its last aligned frame",
            )
            # The keyframe belongs to the tile it ends, not the one inheriting the overlap.
            self.assertTrue(latent_start * 8 <= references[tile_index + 1] < latent_end * 8)

    def test_tile_end_references_land_inside_the_next_tiles_leading_overlap(self):
        """The next tile inherits that frame as its start reference."""
        _, tile_size, overlap, chunks, references = self.module._calculate_schedule(
            24, 48, 10, 2
        )
        for tile_index in range(len(chunks) - 1):
            reference = references[tile_index + 1]
            next_start = chunks[tile_index + 1][0] * 8
            next_overlap_end = next_start + overlap
            self.assertTrue(
                next_start <= reference < next_overlap_end,
                f"tile {tile_index} end must fall in tile {tile_index + 1}'s leading overlap",
            )

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
        self.assertEqual(data["version"], 2)

    def test_version_one_timeline_migrates_without_media(self):
        data, prompts, keyframes = self.module._parse_timeline(
            '{"version":1,"tile_prompts":["tile"],"keyframes":[]}'
        )
        self.assertEqual(data["version"], 2)
        self.assertEqual(prompts, ["tile"])
        self.assertEqual(keyframes, [])
        self.assertEqual(data["video_segments"], [])
        self.assertEqual(data["audio_segments"], [])

    def test_reference_image_tooltip_describes_external_selection(self):
        schema = self.director.define_schema()
        reference = next(item for item in schema.outputs if item.display_name == "reference_image")
        self.assertIn("selected", reference.tooltip.lower())

    def test_schema_exposes_execution_plan_for_combined_sampler(self):
        schema = self.director.define_schema()
        output_names = [item.display_name for item in schema.outputs]
        self.assertIn("director_plan", output_names)
        self.assertNotIn("guiding_latents", output_names)

    def test_execution_plan_keeps_typed_runtime_values_out_of_custom_payload(self):
        data = {
            "version": 2,
            "video_segments": [],
            "ic_segments": [],
            "audio_segments": [],
            "retake_mode": False,
            "retake": None,
        }
        with mock.patch.object(
            self.director,
            "_validate",
            return_value=(data, ["tile"], [], [(0, 2)], 9, 8, 0),
        ), mock.patch.object(self.director, "_encode", return_value="conditioning"), mock.patch.object(
            self.director,
            "_build_conditionings",
            return_value=["conditioning"],
        ), mock.patch.object(
            self.director,
            "_build_keyframes_and_reference",
            return_value=(None, "", None, 32, 32, None),
        ):
            output = self.director.execute(object(), timeline_data="{}")

        plan = output.result[-1]
        self.assertNotIn("positive", plan)
        self.assertNotIn("per_tile_conditionings", plan)
        self.assertNotIn("cond_images", plan)
        self.assertNotIn("start_image", plan)

    def test_legacy_media_range_is_assigned_to_its_tile(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]
        timeline = {
            "version": 2,
            "tile_prompts": ["tile"] * 6,
            "keyframes": [],
            "video_segments": [{"imageFile": "guide.mp4", "start": 8, "length": 1137}],
        }
        with mock.patch.object(media_module, "_resolve_input_file", return_value="/tmp/guide.mp4"):
            data, _, _, chunks, _, _, _ = self.director._validate(timeline, 24, 48, 10, 2)
        self.assertEqual(data["video_segments"][0]["tile"], 0)
        self.assertEqual(chunks[0][0], 0)

    def test_legacy_media_pixel_range_maps_to_the_owned_tile(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]
        timeline = {
            "version": 2,
            "tile_prompts": ["tile"] * 6,
            "keyframes": [],
            "video_segments": [{"imageFile": "guide.mp4", "start": 240, "length": 900}],
        }
        with mock.patch.object(media_module, "_resolve_input_file", return_value="/tmp/guide.mp4"):
            data, _, _, chunks, _, _, _ = self.director._validate(timeline, 24, 48, 10, 2)
        self.assertEqual(data["video_segments"][0]["tile"], 1)
        self.assertEqual(chunks[1], (24, 54))

    def test_multiple_tile_guides_are_allowed(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]
        timeline = {
            "version": 2,
            "tile_prompts": ["tile"] * 6,
            "keyframes": [],
            "video_segments": [{"imageFile": "guide.mp4", "tile": 0}],
            "ic_segments": [{"imageFile": "ic.mp4", "tile": 1}],
        }
        with mock.patch.object(media_module, "_resolve_input_file", return_value="/tmp/guide.mp4"):
            data, _, _, _, _, _, _ = self.director._validate(timeline, 24, 48, 10, 2)
        self.assertEqual(data["video_segments"][0]["tile"], 0)
        self.assertEqual(data["ic_segments"][0]["tile"], 1)

    def test_tile_guide_is_encoded_at_its_tile_grid(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]

        class FakeVae:
            def encode(self, pixels):
                self.pixels = pixels
                return torch.zeros(1, 4, (pixels.shape[0] - 1) // 8 + 1, 2, 2)

        timeline = {
            "version": 2,
            "tile_prompts": ["tile"] * 6,
            "keyframes": [],
            "video_segments": [{"imageFile": "guide.mp4", "tile": 0}],
        }
        frames = torch.zeros(233, 32, 48, 3)
        with mock.patch.object(media_module, "_decode_video_frames", return_value=frames):
            guides = media_module.build_tile_guides(
                timeline,
                [(0, 30)],
                24,
                FakeVae(),
                64,
                64,
                2,
            )

        self.assertEqual(guides[0][0]["samples"].shape, (1, 4, 30, 2, 2))

    def test_multiple_ic_guides_are_kept_in_serialized_order(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]

        class FakeVae:
            def encode(self, pixels):
                return torch.zeros(1, 4, (pixels.shape[0] - 1) // 8 + 1, 2, 2)

        timeline = {
            "version": 2,
            "video_segments": [],
            "ic_segments": [
                {"imageFile": "ic-a.mp4", "tile": 0, "attentionStrength": 0.4},
                {"imageFile": "ic-b.mp4", "tile": 0, "attentionStrength": 0.8},
            ],
            "audio_segments": [],
        }
        with mock.patch.object(
            media_module,
            "_decode_video_frames",
            return_value=torch.zeros(233, 32, 48, 3),
        ):
            guides = media_module.build_tile_guides(
                timeline,
                [(0, 30)],
                24,
                FakeVae(),
                64,
                64,
            )
        self.assertEqual([guide["attention_strength"] for guide in guides[0]], [0.4, 0.8])

    def test_later_tile_guide_is_padded_before_the_overlap(self):
        sampler_module = self._sampler_module()
        source = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24, 1, 1)
        guide = {"samples": source, "tile": 1}
        fitted = sampler_module._fit_guide_frames(guide, 30, source.device, leading_frames=6)
        self.assertEqual(tuple(fitted["samples"].shape), (1, 1, 30, 1, 1))
        self.assertTrue(torch.equal(fitted["samples"][:, :, :6], source[:, :, :1].repeat(1, 1, 6, 1, 1)))
        self.assertTrue(torch.equal(fitted["samples"][:, :, 6:], source))

    def test_audio_mask_preserves_imported_latent_values(self):
        sampler_module = self._sampler_module()
        output = torch.full((1, 1, 4, 1), 9.0)
        initial = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
        mask = torch.tensor([[[[0.0], [0.0], [1.0], [0.5]]]])
        result = sampler_module._apply_audio_mask(output, initial, mask)
        self.assertTrue(torch.equal(result[:, :, :2], initial[:, :, :2]))
        self.assertEqual(float(result[0, 0, 2, 0]), 9.0)
        self.assertEqual(float(result[0, 0, 3, 0]), 6.0)

    def test_audio_mask_preserves_accumulated_prefix(self):
        sampler_module = self._sampler_module()
        output = torch.full((1, 1, 10, 1), 9.0)
        initial = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
        mask = torch.tensor([[[[0.0], [0.0], [1.0], [0.5]]]])
        result = sampler_module._apply_audio_mask(output, initial, mask, prefix_frames=6)
        self.assertEqual(tuple(result.shape), (1, 1, 10, 1))
        self.assertTrue(torch.equal(result[:, :, :6], output[:, :, :6]))
        self.assertTrue(torch.equal(result[:, :, 6:8], initial[:, :, :2]))
        self.assertEqual(float(result[0, 0, 8, 0]), 9.0)
        self.assertEqual(float(result[0, 0, 9, 0]), 6.0)

    def test_checkpoint_component_identity_tracks_sampler_and_guider_settings(self):
        sampler_module = self._sampler_module()

        def sampler_a(*args, **kwargs):
            return None

        def sampler_b(*args, **kwargs):
            return None

        class FakeSampler:
            extra_options = {}
            inpaint_options = {}

            def __init__(self, function):
                self.sampler_function = function

        class FakeGuider:
            def __init__(self, cfg):
                self.cfg = cfg

        self.assertNotEqual(
            sampler_module._component_identity(FakeSampler(sampler_a)),
            sampler_module._component_identity(FakeSampler(sampler_b)),
        )
        self.assertNotEqual(
            sampler_module._component_identity(FakeGuider(1.0)),
            sampler_module._component_identity(FakeGuider(2.0)),
        )

        class FakeUpsampler(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([value]))

        class FakePatcher:
            def __init__(self, value):
                self.model = FakeUpsampler(value)

        self.assertNotEqual(
            sampler_module._component_identity(FakePatcher(1.0)),
            sampler_module._component_identity(FakePatcher(2.0)),
        )

    def test_retake_audio_mask_uses_the_selected_tile(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]

        class AudioModel:
            latents_per_second = 1.0

        class FakeAudioVae:
            audio_sample_rate = 24
            first_stage_model = AudioModel()

            def encode(self, waveform):
                return torch.zeros(1, 1, 4, 1)

        timeline = {
            "version": 2,
            "video_segments": [],
            "ic_segments": [],
            "audio_segments": [],
            "retake_mode": True,
            "retake": {"imageFile": "base.mp4", "tile": 1, "strength": 0.75},
        }
        with mock.patch.object(
            media_module,
            "_segment_waveform",
            return_value=torch.ones(2, 25),
        ):
            _, audio = media_module.build_audio(
                timeline,
                73,
                24,
                FakeAudioVae(),
                [(0, 6), (4, 10)],
            )
        self.assertTrue(torch.equal(audio["noise_mask"][:, :, :3], torch.zeros(1, 1, 3, 1)))
        self.assertEqual(float(audio["noise_mask"][0, 0, 3, 0]), 0.75)

    def test_generated_workflow_matches_sampler_input_names(self):
        sampler_names = [item.id for item in self._sampler_module().LTXLoopingDirectorSampler.define_schema().inputs]
        with open(ROOT / "example_workflows" / "LTX-2.3_Director_Looping.json", encoding="utf-8") as stream:
            workflow = json.load(stream)
        sampler = next(node for node in workflow["nodes"] if node["type"] == "LTXLoopingDirectorSampler")
        self.assertEqual([item["name"] for item in sampler["inputs"]], sampler_names)
        self.assertTrue(any(link[1:5] == [30, 0, 301, sampler_names.index("pass1_latent")] for link in workflow["links"]))
        self.assertIn("MarkdownNote", [node["type"] for node in workflow["nodes"]])
        self.assertNotIn("Power Lora Loader (rgthree)", [node["type"] for node in workflow["nodes"]])

    def test_generated_sampler_widget_values_align_with_the_schema(self):
        """widgets_values must line up slot-for-slot with the node's widget inputs."""
        sampler_cls = self._sampler_module().LTXLoopingDirectorSampler
        v1 = sampler_cls.INPUT_TYPES()
        order = list(v1.get("required", {}).keys()) + list(v1.get("optional", {}).keys())
        specs = {**v1.get("required", {}), **v1.get("optional", {})}

        def is_widget(spec):
            kind = spec[0]
            return isinstance(kind, list) or kind in {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

        schema_widgets = [name for name in order if is_widget(specs[name])]
        with open(ROOT / "example_workflows" / "LTX-2.3_Director_Looping.json", encoding="utf-8") as stream:
            workflow = json.load(stream)
        sampler = next(node for node in workflow["nodes"] if node["type"] == "LTXLoopingDirectorSampler")
        self.assertEqual(
            len(sampler["widgets_values"]),
            len(schema_widgets),
            "widgets_values length must match the number of widget inputs",
        )
        by_name = dict(zip(schema_widgets, sampler["widgets_values"]))
        # Spot-check values whose type makes a shift unmistakable.
        self.assertEqual(by_name["pass_count"], 2)
        self.assertEqual(by_name["execution_mode"], "full")
        self.assertEqual(by_name["ic_lora_name"], "None")
        self.assertEqual(by_name["cond_image_crf"], 30)
        self.assertEqual(by_name["checkpoint_policy"], "off")
        self.assertEqual(by_name["checkpoint_prefix"], "ltx_looping_director")
        for name in ("pass1_seed_offsets", "pass2_seed_offsets"):
            self.assertEqual(by_name[name], "0")
        for name in ("pass1_guiding_start_step", "pass2_guiding_start_step"):
            self.assertEqual(by_name[name], 0)

    def test_every_sample_workflow_matches_the_sampler_schema(self):
        """Both samples, generated and hand-maintained, must track the schema."""
        sampler_cls = self._sampler_module().LTXLoopingDirectorSampler
        schema_inputs = [item.id for item in sampler_cls.define_schema().inputs]
        v1 = sampler_cls.INPUT_TYPES()
        order = list(v1.get("required", {}).keys()) + list(v1.get("optional", {}).keys())
        specs = {**v1.get("required", {}), **v1.get("optional", {})}
        widget_count = sum(
            1 for name in order
            if isinstance(specs[name][0], list)
            or specs[name][0] in {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
        )
        sys.path.insert(0, str(ROOT / "example_workflows"))
        try:
            transformer = importlib.import_module("transform_to_director_looping")
        finally:
            sys.path.pop(0)
        declared_type = {name: typ for name, typ, _ in transformer.SAMPLER_INPUTS}

        for filename in ("LTX-2.3_Director_Looping.json", "LTX-2.3_Director_Looping_Basic.json"):
            with self.subTest(workflow=filename):
                path = ROOT / "example_workflows" / filename
                if not path.is_file():
                    self.skipTest(f"{filename} is not present")
                with open(path, encoding="utf-8") as stream:
                    workflow = json.load(stream)
                sampler = next(
                    node for node in workflow["nodes"]
                    if node["type"] == "LTXLoopingDirectorSampler"
                )
                names = [item["name"] for item in sampler["inputs"]]
                self.assertEqual(names, schema_inputs)
                self.assertEqual(len(sampler["widgets_values"]), widget_count)
                # No link may land on an input of a different type.
                for link in workflow["links"]:
                    if link[3] != sampler["id"]:
                        continue
                    self.assertEqual(
                        link[5],
                        declared_type[names[link[4]]],
                        f"{filename}: {names[link[4]]} receives a {link[5]} link",
                    )

    def test_transformer_widget_order_matches_the_schema(self):
        """The transformer derives widget order rather than hand-maintaining it."""
        sys.path.insert(0, str(ROOT / "example_workflows"))
        try:
            transformer = importlib.import_module("transform_to_director_looping")
        finally:
            sys.path.pop(0)
        sampler_cls = self._sampler_module().LTXLoopingDirectorSampler
        v1 = sampler_cls.INPUT_TYPES()
        order = list(v1.get("required", {}).keys()) + list(v1.get("optional", {}).keys())
        specs = {**v1.get("required", {}), **v1.get("optional", {})}
        schema_widgets = [
            name for name in order
            if isinstance(specs[name][0], list)
            or specs[name][0] in {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
        ]
        self.assertEqual(transformer.sampler_widget_names(), schema_widgets)
        # The declared input list must also match the schema, name for name.
        self.assertEqual(
            [name for name, _, _ in transformer.SAMPLER_INPUTS],
            [item.id for item in sampler_cls.define_schema().inputs],
        )

    def test_generated_workflow_keeps_preprocess_on_the_conditioning_image_path(self):
        sampler_names = [item.id for item in self._sampler_module().LTXLoopingDirectorSampler.define_schema().inputs]
        with open(ROOT / "example_workflows" / "LTX-2.3_Director_Looping.json", encoding="utf-8") as stream:
            workflow = json.load(stream)
        preprocess = next(
            (node for node in workflow["nodes"] if node["type"] == "LTXVPreprocess"), None
        )
        self.assertIsNotNone(preprocess, "the source LTXVPreprocess node must be preserved")
        cond_images_slot = sampler_names.index("cond_images")
        # Director.cond_images -> LTXVPreprocess.image -> sampler.cond_images
        self.assertTrue(
            any(link[1:5] == [300, 2, preprocess["id"], 0] for link in workflow["links"]),
            "the Director conditioning images must feed LTXVPreprocess",
        )
        self.assertTrue(
            any(link[1:5] == [preprocess["id"], 0, 301, cond_images_slot] for link in workflow["links"]),
            "LTXVPreprocess must feed the sampler conditioning images",
        )

    def test_generated_workflow_note_matches_the_generated_timing(self):
        sys.path.insert(0, str(ROOT / "example_workflows"))
        try:
            transformer = importlib.import_module("transform_to_director_looping")
        finally:
            sys.path.pop(0)
        with open(ROOT / "example_workflows" / "LTX-2.3_Director_Looping.json", encoding="utf-8") as stream:
            workflow = json.load(stream)
        director = next(node for node in workflow["nodes"] if node["type"] == "LTXLoopingDirector")
        _, frame_rate, total_duration, tile_duration, overlap_duration = director["widgets_values"][:5]
        _, tile_size, overlap, tile_count = transformer.calculate_schedule(
            frame_rate, total_duration, tile_duration, overlap_duration
        )
        note = next(node for node in workflow["nodes"] if node["type"] == "MarkdownNote")
        self.assertIn(f"{tile_size / frame_rate:.2f}-second tiles", note["widgets_values"][0])
        self.assertIn(f"{overlap / frame_rate:.2f}-second overlap", note["widgets_values"][0])
        self.assertIn(f"{tile_count} tiles", note["widgets_values"][0])
        timeline = json.loads(director["widgets_values"][5])
        self.assertEqual(len(timeline["tile_prompts"]), tile_count)

    def test_retake_initialization_owns_only_the_selected_tile_region(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]

        class FakeVae:
            def encode(self, pixels):
                return torch.zeros(1, 4, 10, 2, 2)

        timeline = {
            "version": 2,
            "video_segments": [],
            "ic_segments": [],
            "audio_segments": [],
            "retake_mode": True,
            "retake": {"imageFile": "base.mp4", "tile": 1, "strength": 0.75},
        }
        with mock.patch.object(
            media_module,
            "_decode_video_frames",
            return_value=torch.zeros(73, 32, 32, 3),
        ):
            latent = media_module.build_retake_latent(
                timeline,
                [(0, 6), (4, 10)],
                73,
                24,
                FakeVae(),
                64,
                64,
            )

        self.assertEqual(latent["samples"].shape, (1, 4, 10, 2, 2))
        self.assertTrue(torch.all(latent["noise_mask"][:, :, :6] == 0))
        self.assertTrue(torch.all(latent["noise_mask"][:, :, 6:10] == 0.75))

    def test_retake_decodes_each_owned_tile_interval(self):
        media_module = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]

        class FakeVae:
            def encode(self, pixels):
                return torch.zeros(1, 4, (pixels.shape[0] - 1) // 8 + 1, 2, 2)

        calls = []

        def decode(filename, start, length, frame_rate, *args):
            calls.append((start, length))
            return torch.zeros(length, 32, 32, 3)

        timeline = {
            "version": 2,
            "video_segments": [],
            "ic_segments": [],
            "audio_segments": [],
            "retake_mode": True,
            "retake": {"imageFile": "base.mp4", "tile": 1, "trimStart": 3, "strength": 1.0},
        }
        with mock.patch.object(media_module, "_decode_video_frames", side_effect=decode):
            latent = media_module.build_retake_latent(
                timeline, [(0, 6), (4, 10)], 73, 24, FakeVae(), 64, 64
            )

        self.assertEqual(calls, [(3, 41), (51, 25)])
        self.assertEqual(latent["samples"].shape[2], 10)

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


class TestLoopingDirectorMediaSettings(TestCase):
    """IC settings and Retake tile selection must reach the encode/mask paths."""

    @classmethod
    def setUpClass(cls):
        _load_looping_director()
        TestLTXLoopingDirector._sampler_module()
        cls.media = sys.modules[f"{PACKAGE_NAME}.ltx_looping_media"]

    def test_ic_settings_default_to_center_crop_and_untiled_encode(self):
        self.assertEqual(
            self.media.resolve_ic_settings({}),
            {
                "crop": "center",
                "upscale_method": "bilinear",
                "use_tiled_encode": False,
                "tile_size": 256,
                "tile_overlap": 64,
            },
        )

    def test_invalid_ic_settings_fail_clearly(self):
        for settings, message in (
            ({"crop": "sideways"}, "crop"),
            ({"upscale_method": "magic"}, "upscale_method"),
            ({"use_tiled_encode": "yes"}, "use_tiled_encode"),
            ({"tile_size": 8}, "tile_size"),
            ({"tile_overlap": 4096}, "tile_overlap"),
        ):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, message):
                    self.media.resolve_ic_settings({"ic_settings": settings})

    def test_guide_encoding_consumes_crop_and_tiled_encode(self):
        calls = {}

        class FakeVae:
            def encode(self, pixels):
                calls["encode"] = pixels.shape
                return torch.zeros(1, 4, 2, 8, 8)

            def encode_tiled(self, pixels, tile_x, tile_y, overlap):
                calls["tiled"] = (pixels.shape, tile_x, tile_y, overlap)
                return torch.zeros(1, 4, 2, 8, 8)

        frames = torch.zeros(9, 32, 48, 3)
        settings = self.media.resolve_ic_settings({
            "ic_settings": {
                "crop": "disabled",
                "upscale_method": "bicubic",
                "use_tiled_encode": True,
                "tile_size": 128,
                "tile_overlap": 32,
            }
        })
        self.media._encode_guide_pixels(FakeVae(), frames, 64, 64, settings)
        self.assertNotIn("encode", calls)
        self.assertEqual(calls["tiled"][1:], (128, 128, 32))

        untiled = self.media.resolve_ic_settings({})
        self.media._encode_guide_pixels(FakeVae(), frames, 64, 64, untiled)

    def test_retake_tiles_accept_the_legacy_single_tile_form(self):
        self.assertEqual(self.media.retake_tiles({"tile": 2}, 4), [2])
        self.assertEqual(self.media.retake_tiles({"tiles": [2, 1, 1]}, 4), [1, 2])
        self.assertEqual(self.media.retake_tiles({}, 4), [])
        with self.assertRaisesRegex(ValueError, "outside the tile schedule"):
            self.media.retake_tiles({"tiles": [9]}, 4)
        with self.assertRaisesRegex(ValueError, "must be integers"):
            self.media.retake_tiles({"tiles": [1.5]}, 4)

    def test_adjacent_retake_tiles_regenerate_one_continuous_region(self):
        chunks = [(0, 4), (3, 7), (6, 10)]

        class FakeVae:
            def encode(self, pixels):
                return torch.zeros(1, 4, 4, 8, 8)

        with mock.patch.object(self.media, "_decode_video_frames", return_value=torch.zeros(25, 8, 8, 3)):
            latent = self.media.build_retake_latent(
                {
                    "retake_mode": True,
                    "retake": {"imageFile": "clip.mp4", "tiles": [1, 2], "strength": 1.0},
                    "video_segments": [],
                    "ic_segments": [],
                    "audio_segments": [],
                },
                chunks,
                73,
                24.0,
                FakeVae(),
                64,
                64,
            )
        mask = latent["noise_mask"][0, 0, :, 0, 0]
        # Tile 1 owns 4..6 (tile 0 keeps the 3..4 overlap) and tile 2 owns 7..9.
        self.assertEqual(mask[:4].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(value == 1.0 for value in mask[4:10].tolist()))

    def test_retake_audio_mask_covers_every_selected_tile(self):
        chunks = [(0, 4), (3, 7), (6, 10)]

        class FakeAudioVae:
            latent_channels = 1
            latent_frequency_bins = 1
            latents_per_second = 4.0

            def encode(self, waveform):
                return torch.zeros(1, 1, 12, 1)

        with mock.patch.object(self.media, "_decode_audio", return_value=torch.zeros(2, 44100)):
            _, audio_latent = self.media.build_audio(
                {
                    "retake_mode": True,
                    "retake": {"imageFile": "clip.mp4", "tiles": [1, 2], "strength": 1.0},
                    "video_segments": [],
                    "ic_segments": [],
                    "audio_segments": [],
                },
                73,
                24.0,
                FakeAudioVae(),
                chunks,
            )
        mask = audio_latent["noise_mask"][0, 0, :, 0]
        self.assertGreater(float(mask.sum()), 0.0)
        # The frozen head of the clip (tile 0's owned region) stays at 0.
        self.assertEqual(float(mask[0]), 0.0)


def _fake_ltx_runtime(calls):
    """Minimal stand-ins for the LTXVideo sampling primitives, recording their calls."""

    class SelectLatents:
        def select_latents(self, samples, start_index, end_index):
            selected = dict(samples)
            frames = selected["samples"].shape[2]
            start = frames + start_index if start_index < 0 else start_index
            end = frames + end_index if end_index < 0 else end_index
            start = max(0, min(start, frames - 1))
            end = max(0, min(end, frames - 1))
            selected["samples"] = selected["samples"][:, :, start:end + 1]
            if selected.get("noise_mask") is not None:
                selected["noise_mask"] = selected["noise_mask"][:, :, start:end + 1]
            return (selected,)

    class BaseSampler:
        def sample(self, **kwargs):
            calls.append(("base", kwargs))
            return ({"samples": kwargs["optional_initialization_latents"]["samples"].clone()},)

    class InContextSampler:
        def sample(self, **kwargs):
            calls.append(("incontext", kwargs))
            return ({"samples": kwargs["optional_initialization_latents"]["samples"].clone()},)

    class ExtendSampler:
        def sample(self, **kwargs):
            calls.append(("extend", kwargs))
            samples = kwargs["latents"]["samples"]
            grown = torch.cat(
                [samples, samples[:, :, -1:].repeat(1, 1, kwargs["num_new_frames"], 1, 1)],
                dim=2,
            )
            return ({"samples": grown},)

    return types.SimpleNamespace(
        LTXVSelectLatents=SelectLatents,
        LTXVBaseSampler=BaseSampler,
        LTXVInContextSampler=InContextSampler,
        LTXVExtendSampler=ExtendSampler,
    )


class TestLoopingDirectorSamplerGuides(TestCase):
    """The temporal loop must honour each pass's own legacy guiding latent."""

    @classmethod
    def setUpClass(cls):
        _load_looping_director()
        cls.sampler_module = TestLTXLoopingDirector._sampler_module()

    def _run_loop(self, *, legacy_guiding, tile_guides):
        module = self.sampler_module
        calls = []
        total_frames = 6
        overlap = 1

        guiding_payload = {"samples": torch.arange(
            1 * 4 * total_frames * 2 * 2, dtype=torch.float32
        ).reshape(1, 4, total_frames, 2, 2)}
        carrier = dict(guiding_payload)
        if tile_guides is not None:
            carrier["_director_tile_guides"] = tile_guides

        tile_config = types.SimpleNamespace(
            tile_latents={"samples": torch.zeros(1, 4, total_frames, 2, 2)},
            tile_guiding_latents=carrier,
            tile_normalizing_latents=None,
            tile_negative_index_latents=None,
            tile_keyframes=None,
            keyframe_per_tile_indices=[],
            first_seed=0,
            vertical_tiles=1,
            horizontal_tiles=1,
            v=0,
            h=0,
            tile_width=2,
            tile_height=2,
        )
        sampling_config = types.SimpleNamespace(
            temporal_tile_size=3,
            temporal_overlap=overlap,
            guiding_strength=0.5,
            time_scale_factor=1,
            width_scale_factor=1,
            height_scale_factor=1,
            cond_image_strength=1.0,
            optional_negative_index=-1,
            optional_negative_index_strength=1.0,
            guiding_start_step=0,
            guiding_end_step=1000,
            adain_factor=0.0,
            temporal_overlap_cond_strength=0.5,
            optional_positive_conditionings=None,
            per_tile_seed_offsets="0",
        )
        model_config = types.SimpleNamespace(
            model=object(),
            vae=object(),
            noise=types.SimpleNamespace(seed=0),
            sampler=object(),
            sigmas=object(),
            guider=object(),
        )

        fake = types.SimpleNamespace(
            _director_legacy_guiding=legacy_guiding,
            _director_audio_mask=None,
            _director_checkpoint_context={"resume_chunks": {}},
            _get_per_tile_value=lambda offsets, index: 0,
            _calculate_tile_seed=lambda *args: 1234,
            _prepare_guider_for_chunk=lambda guider, conds, index: guider,
        )

        with mock.patch.object(module, "_ltx_looping_module", return_value=_fake_ltx_runtime(calls)):
            module._process_temporal_chunks(fake, tile_config, sampling_config, model_config)
        return calls

    def test_legacy_guiding_latent_reaches_base_and_extend_paths(self):
        calls = self._run_loop(legacy_guiding=True, tile_guides=None)
        kinds = [kind for kind, _ in calls]
        self.assertEqual(kinds[0], "incontext")
        self.assertIn("extend", kinds)

        first = calls[0][1]
        self.assertIsNotNone(first["guiding_latents"])
        # The first chunk covers latent frames 0..2 of the connected guide.
        self.assertEqual(first["guiding_latents"]["samples"].shape[2], 3)
        self.assertEqual(first["guiding_strength"], 0.5)

        extend = next(kwargs for kind, kwargs in calls if kind == "extend")
        self.assertIsNotNone(
            extend["optional_guiding_latents"],
            "the connected guiding latent must reach LTXVExtendSampler",
        )
        self.assertEqual(extend["guiding_strength"], 0.5)

    def test_tile_guide_carrier_is_not_treated_as_a_legacy_guide(self):
        calls = self._run_loop(legacy_guiding=False, tile_guides=[[], [], []])
        kinds = [kind for kind, _ in calls]
        self.assertEqual(kinds[0], "base")
        self.assertNotIn("incontext", kinds)
        extend = next(kwargs for kind, kwargs in calls if kind == "extend")
        self.assertIsNone(extend["optional_guiding_latents"])

    def test_each_pass_receives_only_its_own_guide_controls(self):
        module = self.sampler_module
        plan = {
            "version": 1,
            "frame_count": 9,
            "frame_rate": 24,
            "chunks": [{"start": 0, "end": 2}],
            "temporal_tile_size": 2,
            "temporal_overlap": 0,
            "stage_1_width": 64,
            "stage_1_height": 64,
            "target_width": 128,
            "target_height": 128,
            "media": {"video_segments": [], "ic_segments": [], "audio_segments": []},
        }
        video = {"samples": torch.zeros(1, 4, 2, 2, 2)}
        audio = {"samples": torch.zeros(1, 1, 2, 1), "noise_mask": torch.ones(1, 1, 2, 1)}
        initial = module._concat_av(video, audio)
        pass1_guide = {"samples": torch.zeros(1, 4, 2, 2, 2)}
        pass2_guide = {"samples": torch.ones(1, 4, 2, 2, 2)}
        seen = []
        parameters = list(
            inspect.signature(
                module.LTXLoopingDirectorSampler.__dict__["_run_pass"].__func__
            ).parameters
        )[1:]

        def fake_run_pass(*args, **kwargs):
            bound = dict(zip(parameters, args))
            bound.update(kwargs)
            seen.append((bound["pass_index"], bound["legacy_guiding_latent"]))
            return initial

        with mock.patch.object(
            module.LTXLoopingDirectorSampler, "_run_pass", side_effect=fake_run_pass
        ), mock.patch.object(module.LTXLoopingDirectorSampler, "_upsample", side_effect=lambda v, *a: v):
            module.LTXLoopingDirectorSampler.execute(
                plan,
                [],
                model=object(),
                video_vae=object(),
                audio_vae=object(),
                latent_upscale_model=object(),
                pass_count=2,
                pass1_noise=object(), pass1_sampler=object(),
                pass1_sigmas=object(), pass1_guider=object(),
                pass2_noise=object(), pass2_sampler=object(),
                pass2_sigmas=object(), pass2_guider=object(),
                pass1_latent=initial,
                pass1_guiding_latent=pass1_guide,
                pass2_guiding_latent=pass2_guide,
                checkpoint_policy="off",
            )

        self.assertEqual(len(seen), 2)
        (pass1_index, pass1_legacy), (pass2_index, pass2_legacy) = seen
        self.assertEqual((pass1_index, pass2_index), (1, 2))
        self.assertIs(pass1_legacy, pass1_guide)
        self.assertIs(pass2_legacy, pass2_guide)


class TestLoopingDirectorCheckpoints(TestCase):
    """Checkpoints must commit as one immutable generation, manifest last."""

    @classmethod
    def setUpClass(cls):
        _load_looping_director()
        cls.sampler_module = TestLTXLoopingDirector._sampler_module()

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = mock.patch.object(
            self.sampler_module.folder_paths,
            "get_input_directory",
            return_value=self.directory.name,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _context(self, generation="aaaaaaaaaaaa", pass_index=1):
        return {
            "prefix": "ckpt",
            "generation": generation,
            "format_version": self.sampler_module.CHECKPOINT_FORMAT_VERSION,
            "pass_index": pass_index,
            "frame_count": 9,
            "fingerprint": "fp",
            "resume_chunks": {},
        }

    def _save_chunk(self, context, chunk_index):
        module = self.sampler_module
        fake = types.SimpleNamespace(_director_checkpoint_context=context)
        tile_config = types.SimpleNamespace(
            v=0, h=0, tile_latents={"samples": torch.zeros(1, 4, 2, 2, 2)}
        )
        module._save_checkpoint(
            fake,
            {"samples": torch.full((1, 4, 2 + chunk_index, 2, 2), float(chunk_index))},
            torch.zeros(1, 1, 2, 1),
            tile_config,
            chunk_index,
        )

    def _manifest(self):
        with open(Path(self.directory.name) / "ckpt_manifest.json", encoding="utf-8") as stream:
            return json.load(stream)

    def test_chunk_tensors_use_immutable_generation_filenames(self):
        context = self._context()
        self._save_chunk(context, 0)
        manifest = self._manifest()
        entry = manifest["completed_chunks"]["0:0"]
        self.assertEqual(entry["chunk"], 0)
        self.assertIn("_gaaaaaaaaaaaa_", entry["video"])
        self.assertIn("_c0_", entry["video"])
        self.assertTrue((Path(self.directory.name) / entry["video"]).is_file())

    def test_manifest_commit_precedes_removal_of_the_previous_chunk(self):
        context = self._context()
        self._save_chunk(context, 0)
        first = self._manifest()["completed_chunks"]["0:0"]["video"]
        self._save_chunk(context, 1)
        second = self._manifest()["completed_chunks"]["0:0"]["video"]

        self.assertNotEqual(first, second)
        self.assertEqual(self._manifest()["completed_chunks"]["0:0"]["chunk"], 1)
        self.assertTrue((Path(self.directory.name) / second).is_file())
        # Only after the manifest names the new chunk is the old one removed.
        self.assertFalse((Path(self.directory.name) / first).is_file())

    def test_a_failed_manifest_commit_leaves_the_previous_chunk_resumable(self):
        module = self.sampler_module
        context = self._context()
        self._save_chunk(context, 0)
        durable = self._manifest()
        durable_video = durable["completed_chunks"]["0:0"]["video"]

        with mock.patch.object(module, "_write_manifest", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self._save_chunk(context, 1)

        # The committed manifest still describes chunk 0, and chunk 0's tensors are
        # untouched, so resume redoes chunk 1 instead of double-generating it.
        self.assertEqual(self._manifest(), durable)
        self.assertTrue((Path(self.directory.name) / durable_video).is_file())

    def test_new_pass_one_progress_clears_a_completed_pass_one_snapshot(self):
        module = self.sampler_module
        generation = "bbbbbbbbbbbb"
        latent = module._concat_av(
            {"samples": torch.zeros(1, 4, 2, 2, 2)},
            {"samples": torch.zeros(1, 1, 2, 1)},
        )
        module._save_pass_snapshot("ckpt", generation, 1, latent, "fp")
        self.assertEqual(self._manifest()["completed_pass"], 1)

        self._save_chunk(self._context(generation=generation), 0)
        manifest = self._manifest()
        self.assertNotIn("completed_pass", manifest)
        self.assertNotIn("pass1_video", manifest)
        self.assertNotIn("pass1_audio", manifest)

    def test_a_new_generation_discards_the_previous_runs_manifest_keys(self):
        module = self.sampler_module
        latent = module._concat_av(
            {"samples": torch.zeros(1, 4, 2, 2, 2)},
            {"samples": torch.zeros(1, 1, 2, 1)},
        )
        module._save_pass_snapshot("ckpt", "cccccccccccc", 1, latent, "fp")
        stale_video = self._manifest()["pass1_video"]

        self._save_chunk(self._context(generation="dddddddddddd"), 0)
        manifest = self._manifest()
        self.assertEqual(manifest["generation"], "dddddddddddd")
        self.assertNotIn("pass1_video", manifest)
        # The superseded generation's tensors are pruned once the new manifest commits.
        self.assertFalse((Path(self.directory.name) / stale_video).is_file())

    def test_a_stale_generation_snapshot_is_not_loaded(self):
        module = self.sampler_module
        latent = module._concat_av(
            {"samples": torch.zeros(1, 4, 2, 2, 2)},
            {"samples": torch.zeros(1, 1, 2, 1)},
        )
        module._save_pass_snapshot("ckpt", "eeeeeeeeeeee", 1, latent, "fp")
        self.assertIsNone(module._load_pass_snapshot("ckpt", 1, "ffffffffffff"))
        self.assertIsNotNone(module._load_pass_snapshot("ckpt", 1, "eeeeeeeeeeee"))

    def test_a_missing_audio_stream_fails_instead_of_returning_video_only(self):
        module = self.sampler_module
        generation = "111111111111"
        latent = module._concat_av(
            {"samples": torch.zeros(1, 4, 2, 2, 2)},
            {"samples": torch.zeros(1, 1, 2, 1)},
        )
        module._save_pass_snapshot("ckpt", generation, 1, latent, "fp")
        audio_name = self._manifest()["pass1_audio"]
        self.assertIsNotNone(audio_name)
        (Path(self.directory.name) / audio_name).unlink()

        with self.assertRaisesRegex(ValueError, "missing or unreadable"):
            module._load_pass_snapshot("ckpt", 1, generation)

    def test_a_missing_video_stream_fails(self):
        module = self.sampler_module
        generation = "222222222222"
        latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
        module._save_pass_snapshot("ckpt", generation, 1, latent, "fp")
        video_name = self._manifest()["pass1_video"]
        (Path(self.directory.name) / video_name).unlink()

        with self.assertRaisesRegex(ValueError, "missing or unreadable"):
            module._load_pass_snapshot("ckpt", 1, generation)
