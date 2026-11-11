"""Behavior checks for split ownership, freeze gates and export denominators."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dataset_plan as plan


def metadata_for(case, success=False, accepted=False, status="completed"):
    policy = {"source_sha256": {"closed_loop_controller.py": "controller-a"},
              "simulation_parameters": {"speed": 6.0},
              "geometry_rule": {"stairs": case["parameters"]["stairs"], "depth_equals_height": True, "slope_deg": 45.0}}
    value = {"case": case["case_id"], "split": case["split"], "case_spec": case,
             "parameters": dict(case["parameters"], speed=6.0), "policy_snapshot": policy,
             "policy_sha256": plan.fingerprint(policy), "simulation_success": success,
             "dataset_accepted": accepted, "automated_checks_passed": accepted,
             "attempt_status": status}
    if case["split"] == "test":
        value["release_snapshot"] = plan.release_document(policy)
    return value


class SplitBehaviorTests(unittest.TestCase):
    def test_defaults_are_high_stairs_and_validation_uses_other_initial_states(self):
        train = plan.build_cases()
        validation = plan.build_cases("validation")
        self.assertEqual({case["parameters"]["height"] for case in train}, {0.15, 0.20})
        self.assertTrue(all(case["parameters"]["stairs"] == 15 for case in train + validation))
        self.assertTrue(all(case["parameters"]["height"] == case["parameters"]["depth"] for case in train + validation))
        self.assertFalse({c["group_sha256"] for c in train} & {c["group_sha256"] for c in validation})
        self.assertEqual({c["parameters"]["height"] for c in plan.build_cases("test")}, {0.17, 0.18})

    def test_reserved_heights_cannot_be_collected_as_training(self):
        for split, heights in [("train", [17]), ("validation", [18]), ("train", [4])]:
            with self.assertRaises(ValueError):
                plan.build_cases(split, heights=heights)
        with self.assertRaises(ValueError):
            plan.build_cases("validation", variants=["base"])

    def test_legacy_geometry_remains_twelve_stairs(self):
        old = {"case": "old_seed", "parameters": {"height": 0.04, "depth": 0.04, "stairs": 12}}
        spec = plan.assignment(old)
        self.assertEqual(spec["split"], "regression")
        self.assertEqual(spec["parameters"]["stairs"], 12)
        self.assertEqual(plan.assignment(dict(old, split="regression", case_spec=spec)), spec)
        self.assertEqual(plan.build_cases("regression")[0]["parameters"]["stairs"], 12)
        with self.assertRaises(ValueError):
            plan.build_cases("train", stairs=12)
        with self.assertRaises(ValueError):
            plan.build_cases("regression", stairs=15)

    def test_test_requires_frozen_unchanged_policy(self):
        policy = {"controller_sha256": "a", "simulation_parameters": {"speed": 6}, "source_sha256": {"x": "a"}}
        with self.assertRaises(ValueError):
            plan.require_release(None, policy)
        release = plan.release_document(policy)
        plan.require_release(release, policy)
        for field, replacement in [("controller_sha256", "b"), ("simulation_parameters", {"speed": 7}),
                                   ("source_sha256", {"x": "b"})]:
            changed = dict(policy, **{field: replacement})
            with self.assertRaises(ValueError):
                plan.require_release(release, changed)
        release["frozen"] = False
        with self.assertRaises(ValueError):
            plan.require_release(release, policy)

    def test_changed_plan_or_test_metadata_is_rejected(self):
        case = plan.build_cases("test")[0]
        metadata = metadata_for(case)
        plan.assignment(metadata)
        metadata["parameters"]["init_y"] = 0.01
        with self.assertRaises(ValueError):
            plan.assignment(metadata)
        metadata = metadata_for(case)
        metadata["release_snapshot"]["plan"]["cases"].pop()
        with self.assertRaises(ValueError):
            plan.assignment(metadata)

    def test_failure_and_incomplete_attempts_stay_in_test_denominator(self):
        cases = plan.build_cases("test")
        good = metadata_for(cases[0], success=True, accepted=True)
        bad = metadata_for(cases[1])
        interrupted = metadata_for(cases[2], status="error")
        attempts, accepted, unknown, summary = plan.split_exports(
            [("/good", good), ("/bad", bad), ("/interrupted", interrupted)],
            [{"metadata_path": "/good", "case": "good"}])
        self.assertEqual(summary["splits"]["test"]["attempts"], 3)
        self.assertEqual(summary["splits"]["test"]["accepted"], 1)
        self.assertEqual(summary["splits"]["test"]["incomplete_or_error"], 1)
        self.assertFalse(summary["test_reserved"])
        self.assertTrue(all(row["split"] == "test" for row in attempts["test"]))
        self.assertEqual(len(accepted["train"]), 0)

    def test_outcome_changes_do_not_change_assignment(self):
        case = plan.build_cases("validation")[0]
        failed = metadata_for(case)
        passed = dict(failed, simulation_success=True, dataset_accepted=True)
        self.assertEqual(plan.assignment(failed), plan.assignment(passed))

    def test_unknown_legacy_attempt_is_preserved_without_guessing_split(self):
        old = {"case": "old_5cm", "parameters": {"height": 0.05, "depth": 0.05, "stairs": 12}}
        _, _, unknown, summary = plan.split_exports([("/old", old)], [])
        self.assertEqual(len(unknown), 1)
        self.assertEqual(summary["unassigned_legacy_attempts"], 1)
        with self.assertRaises(ValueError):
            plan.split_exports([("/old", old)], [{"metadata_path": "/old"}])

    def test_release_creation_is_explicit_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "release.json"
            plan.write_release(path, {"policy": "a"})
            plan.require_release(json.loads(path.read_text()), {"policy": "a"})
            with self.assertRaises(FileExistsError):
                plan.write_release(path, {"policy": "b"})

    def test_cli_test_dry_run_never_starts_simulation_or_creates_output(self):
        import generate_dataset as generator
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "not-created"
            with patch.object(generator.course, "simulate", side_effect=AssertionError("simulation called")), \
                 patch.object(generator, "policy_snapshot", return_value={"policy": "a"}), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(generator.main(["--split", "test", "--dry-run", "--dataset-root", str(root)]), 0)
            value = json.loads(output.getvalue())
            self.assertFalse(value["selected_split_executable"])
            self.assertFalse(value["test_release_valid"])
            self.assertEqual(len(value["cases"]), 2)
            self.assertFalse(value["simulation_started"])
            self.assertFalse(root.exists())

    def test_cli_test_without_freeze_rejects_before_creating_attempt(self):
        import generate_dataset as generator
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "not-created"
            with patch.object(generator.course, "simulate", side_effect=AssertionError("simulation called")), \
                 patch.object(generator, "policy_snapshot", return_value={"policy": "a"}), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    generator.main(["--split", "test", "--dataset-root", str(root)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(root.exists())

    def test_publisher_exports_all_four_groups_without_touching_compatibility_indexes(self):
        import publish_dataset as publisher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "episodes" / "failed"
            folder.mkdir(parents=True)
            metadata = metadata_for(plan.build_cases("train")[0])
            (folder / "metadata.json").write_text(json.dumps(metadata))
            (root / "accepted.jsonl").write_text("")
            result = publisher.refresh_splits(root)
            self.assertEqual(result["summary"]["splits"]["train"]["attempts"], 1)
            self.assertEqual((root / "accepted.jsonl").read_text(), "")
            for split in plan.SPLITS:
                self.assertTrue((root / f"{split}.jsonl").exists())
                self.assertTrue((root / f"{split}_attempts.jsonl").exists())
            self.assertEqual(len((root / "train_attempts.jsonl").read_text().splitlines()), 1)
            self.assertFalse((root / "test_attempts.jsonl").read_text())

    def test_generator_error_keeps_attempt_and_concrete_split(self):
        import generate_dataset as generator
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(generator, "BASE", root), \
                 patch.object(generator.course, "simulate", side_effect=RuntimeError("injected test error")):
                with self.assertRaisesRegex(RuntimeError, "injected test error"):
                    generator.generate("failed_case", 0.15, {"speed": 6.0})
            metadata = json.loads((root / "episodes/failed_case/metadata.json").read_text())
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(metadata["parameters"]["stairs"], 15)
            self.assertEqual(metadata["attempt_status"], "error")
            self.assertFalse(metadata["dataset_accepted"])
            self.assertEqual(len((root / "attempts.jsonl").read_text().splitlines()), 1)
            self.assertEqual(plan.assignment(metadata)["split"], "train")

    def test_publication_preserves_old_seed_and_web_while_exporting_new_train(self):
        import hashlib
        import publish_dataset as publisher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            videos = Path(temp) / "videos"
            videos.mkdir()
            paths = []
            new = metadata_for(plan.build_cases("train")[0], success=True)
            old = {"case": "old_seed", "parameters": {"height": 0.04, "depth": 0.04, "stairs": 12},
                   "simulation_success": True}
            for metadata in [old, new]:
                case = metadata["case"]
                folder = root / "episodes" / case
                folder.mkdir(parents=True)
                trajectory = folder / "trajectory.npz"
                scene = folder / "scene.xml"
                image = folder / "review.jpg"
                video = videos / f"{case}.mp4"
                for file in [trajectory, scene, image, video]:
                    file.write_bytes(b"fixture")
                metadata.update(trajectory_path=str(trajectory), scene_path=str(scene), out=str(video),
                                automated_checks_passed=True, dataset_accepted=False)
                path = folder / "metadata.json"
                path.write_text(json.dumps(metadata))
                review = {"status": "full_video_pass", "reviewer": "unit-test", "notes": "fixture",
                          "reviewed_images": [str(image)], "trajectory_sha256": hashlib.sha256(b"fixture").hexdigest(),
                          "video_sha256": hashlib.sha256(b"fixture").hexdigest()}
                (folder / "visual_review.json").write_text(json.dumps(review))
                paths.append(path)
            verify = lambda _: {"verify_ok": True}
            probe = lambda _: {"ffprobe_ok": True}
            result = publisher.publish(paths, dataset_root=root, videos_dir=videos, verifier=verify, video_probe=probe)
            self.assertEqual(result["accepted_count"], 2)
            self.assertEqual(len((root / "accepted.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((videos / "runs.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((root / "train.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((root / "regression.jsonl").read_text().splitlines()), 1)
            reread = json.loads(paths[0].read_text())
            self.assertEqual(reread["parameters"]["stairs"], 12)
            self.assertEqual(reread["split"], "regression")
            result = publisher.publish([paths[1]], dataset_root=root, videos_dir=videos, verifier=verify, video_probe=probe)
            self.assertEqual(result["accepted_count"], 2)



if __name__ == "__main__":
    unittest.main()
