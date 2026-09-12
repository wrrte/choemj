"""CPU regressions for shared Retrieval and sequential post-warmup branches.

Run with a Python environment containing torch and einops::

    python -m unittest discover -s tests -v

The training entry points are intentionally not imported: doing so requires the
Atari/DMControl runtime and GPU-only model extensions.
"""

import ast
import argparse
import copy
import importlib.util
import json
import random
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from collections import deque

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import FastHashBucket, RetrievalContextManager
from training_branches import (
    capture_rng_state,
    launch_training_branches,
    parse_retrieval_mode,
    restore_rng_state,
    save_final_models,
    split_retrieval_override,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_function(name, path=None, extra_namespace=None):
    """Execute one real training function without importing GPU extensions."""
    path = path or ROOT / "Drama" / "train.py"
    module = ast.parse(path.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args + function.args.kwonlyargs:
        argument.annotation = None
    extracted = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"np": np, "torch": torch, "is_logging_enabled": lambda logger: False}
    namespace.update(extra_namespace or {})
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace[name]


DramaReplayBuffer = load_module(
    "drama_replay_buffer_for_tests", ROOT / "Drama" / "replay_buffer.py"
).ReplayBuffer
checkpoint_helpers = load_module("drama_checkpoint_for_tests", ROOT / "Drama" / "training_checkpoint.py")


class TinyTrainable(torch.nn.Module):
    """Includes optimizer and auxiliary statistics that state_dict omits."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.03)
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=.5)
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(1., (step + 1) / 3))
        self.lowerbound_ema = SimpleNamespace(scalar=torch.tensor(-2.5))
        self.upperbound_ema = SimpleNamespace(scalar=4.5)
        self.normalizer = torch.nn.Identity()
        self.normalizer.ob_rms = SimpleNamespace(
            mean=torch.tensor([1., 2.]), var=torch.tensor([3., 4.]), count=torch.tensor(9.)
        )
        self.device = "cpu"

    def update_once(self):
        self.optimizer.zero_grad()
        self.weight.square().sum().backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.warmup_scheduler.step()


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def tree_equal(testcase, expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual)
    elif isinstance(expected, dict):
        testcase.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            tree_equal(testcase, expected[key], actual[key])
    elif isinstance(expected, (tuple, list, deque)):
        testcase.assertEqual(len(expected), len(actual))
        for first, second in zip(expected, actual):
            tree_equal(testcase, first, second)
    elif isinstance(expected, FastHashBucket):
        tree_equal(testcase, expected.__dict__, actual.__dict__)
    else:
        testcase.assertEqual(expected, actual)


def make_drama_buffer(store_tensors=False, continuous=True, capacity=8):
    config = SimpleNamespace(
        BasicSettings=SimpleNamespace(
            ReplayBufferOnGPU=store_tensors, ImageSize=(2, 2), ImageChannel=1
        ),
        JointTrainAgent=SimpleNamespace(
            BufferMaxLength=capacity,
            WorldModelWarmUp=0,
            BehaviourWarmUp=0,
            Tau=1.0,
            ImaginationTau=1.0,
            Alpha=0.1,
            Beta=0.1,
            ImagineBatchSize=2,
            BatchSize=2,
        ),
    )
    return DramaReplayBuffer(
        config, device="cpu", action_dim=2 if continuous else 1,
        is_discrete=not continuous,
    )


def fill_buffer(buffer, count):
    for step in range(count):
        action = step if buffer.is_discrete else np.array([step, -step], np.float32)
        buffer.append(np.full((2, 2, 1), step, np.uint8), action, step / 10, False)


class ConstantEncoder:
    def __init__(self):
        self.seen = []

    def encode_obs(self, observations, sample_mode):
        assert observations.device.type == "cpu"
        assert observations.ndim == 5
        self.seen.append(observations.clone())
        return torch.ones((*observations.shape[:2], 4), device=observations.device)


def make_manager(context_length=3, enabled=True):
    manager = RetrievalContextManager(
        num_envs=1,
        config={
            "enable": enabled,
            "context_length": context_length,
            "hash_bits": 2,
            "use_pca": False,
            "anchor_weight": 0.7,
        },
        latent_dim=4,
        device="cpu",
    )
    manager.hash_proj.fill_(1)
    return manager


def retrieve_at(manager, buffer, pointer, *, target=1, return_indices=True):
    manager._insert_into_bucket(pointer, 0, 3)
    manager.active_anchors.append(((pointer, 0), 3))
    return manager.retrieve_contexts(
        buffer, ConstantEncoder(), max_anchors=1,
        target=target, multiplier=5, return_indices=return_indices,
    )


class RetrievalModeTests(unittest.TestCase):
    def test_storm_yaml_loader_accepts_modes_with_yacs(self):
        from yacs.config import CfgNode
        load_config = load_train_function("load_config", ROOT / "STORM" / "utils.py", {"CN": CfgNode})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for literal, expected in [("true", True), ("false", False), ('"False"', False), ("Both", "Both")]:
                with self.subTest(literal=literal):
                    path.write_text(f"JointTrainAgent:\n  Retrieval:\n    enable: {literal}\n")
                    self.assertEqual(parse_retrieval_mode(load_config(path).JointTrainAgent.Retrieval.enable), expected)

    def test_drama_cli_preserves_both_and_false_modes(self):
        parse_config = load_train_function("parse_args_and_update_config", extra_namespace={
            "argparse": argparse, "ast": ast, "parse_retrieval_mode": parse_retrieval_mode,
        })
        for raw, expected in [("True", True), ("False", False), ("Both", "Both")]:
            config = {"JointTrainAgent": {"Retrieval": {"enable": False, "warmup_steps": 10}}}
            result = parse_config(config, argv=["--JointTrainAgent.Retrieval.enable", raw])
            self.assertEqual(result["JointTrainAgent"]["Retrieval"]["enable"], expected)

    def test_normalizes_modes_without_string_truthiness(self):
        for raw, expected in [
            (True, True), (False, False), ("True", True),
            ("False", False), ("both", "Both"), ("BOTH", "Both"),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_retrieval_mode(raw), expected)
        self.assertFalse(make_manager(enabled="False").enabled)

    def test_rejects_ambiguous_values(self):
        for value in [None, 0, 1, [], {}, "", "yes", "off", "truu"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_retrieval_mode(value)

    def test_yacs_override_keeps_other_options_and_last_mode(self):
        remaining, mode = split_retrieval_override([
            "JointTrainAgent.BatchSize", "12",
            "JointTrainAgent.Retrieval.enable", "False",
            "JointTrainAgent.Retrieval.warmup_steps", "100",
            "JointTrainAgent.Retrieval.enable", "Both",
        ], "JointTrainAgent.Retrieval.enable")
        self.assertEqual(mode, "Both")
        self.assertEqual(remaining, [
            "JointTrainAgent.BatchSize", "12",
            "JointTrainAgent.Retrieval.warmup_steps", "100",
        ])
        with self.assertRaises(ValueError):
            split_retrieval_override(
                ["JointTrainAgent.Retrieval.enable"],
                "JointTrainAgent.Retrieval.enable",
            )


class RetrievalBufferTests(unittest.TestCase):
    def test_cpu_view_preserves_continuous_actions_and_storage(self):
        for tensors in [False, True]:
            with self.subTest(tensors=tensors):
                buffer = make_drama_buffer(tensors)
                fill_buffer(buffer, 5)
                view = buffer.retrieval_view()
                self.assertEqual(tuple(view.obs_buffer.shape), (8, 1, 2, 2, 1))
                self.assertEqual(tuple(view.action_buffer.shape), (8, 1, 2))
                if tensors:
                    self.assertEqual(view.obs_buffer.data_ptr(), buffer.obs_buffer.data_ptr())
                else:
                    self.assertTrue(np.shares_memory(view.obs_buffer, buffer.obs_buffer))
                result = retrieve_at(make_manager(), buffer, 4)
                obs, action, _, _, weights, groups, indexes = result
                self.assertEqual(tuple(obs.shape), (1, 3, 1, 2, 2))
                self.assertEqual(tuple(action.shape), (1, 3, 2))
                torch.testing.assert_close(action[0], torch.tensor([[2., -2.], [3., -3.], [4., -4.]]))
                torch.testing.assert_close(obs[0, :, 0, 0, 0], torch.tensor([2., 3., 4.]) / 255)
                self.assertEqual(weights, [1.0])
                self.assertEqual(groups, 1)
                self.assertEqual(indexes, [(4, 0)])

    def test_discrete_actions_and_legacy_six_value_result(self):
        buffer = make_drama_buffer(continuous=False)
        fill_buffer(buffer, 5)
        result = retrieve_at(make_manager(), buffer, 4, return_indices=False)
        self.assertEqual(len(result), 6)
        self.assertEqual(tuple(result[1].shape), (1, 3))
        for return_indices in [False, True]:
            result = make_manager(enabled=False).retrieve_contexts(
                buffer, ConstantEncoder(), 1, return_indices=return_indices
            )
            self.assertEqual(len(result), 7 if return_indices else 6)
            self.assertIsNone(result[0])

    def test_partial_buffer_rejects_unwritten_or_wrapped_history(self):
        buffer = make_drama_buffer()
        fill_buffer(buffer, 4)
        for pointer in [0, 1, 4, 7]:
            with self.subTest(pointer=pointer):
                self.assertIsNone(retrieve_at(make_manager(), buffer, pointer)[0])
        for pointer in [2, 3]:
            with self.subTest(pointer=pointer):
                self.assertIsNotNone(retrieve_at(make_manager(), buffer, pointer)[0])

    def test_ring_buffer_accepts_real_wrap_and_rejects_newest_to_oldest_jump(self):
        buffer = make_drama_buffer()
        fill_buffer(buffer, 10)  # newest slot 1; oldest slot 2
        for pointer in [2, 3]:
            self.assertIsNone(retrieve_at(make_manager(), buffer, pointer)[0])
        obs = retrieve_at(make_manager(), buffer, 0)[0]
        torch.testing.assert_close(obs[0, :, 0, 0, 0], torch.tensor([6., 7., 8.]) / 255)

    def test_context_may_end_at_terminal_but_cannot_cross_terminal(self):
        buffer = make_drama_buffer()
        fill_buffer(buffer, 6)
        buffer.termination_buffer[3] = 1
        self.assertIsNotNone(retrieve_at(make_manager(), buffer, 3)[0])
        self.assertIsNone(retrieve_at(make_manager(), buffer, 4)[0])
        self.assertIsNone(retrieve_at(make_manager(), buffer, 5)[0])

    def test_retrieval_cannot_cross_truncation_without_environment_terminal(self):
        buffer = make_drama_buffer()
        fill_buffer(buffer, 4)
        buffer.append(np.full((2, 2, 1), 4, np.uint8), np.array([4., -4.]), 0., False, episode_end=True)
        buffer.append(np.full((2, 2, 1), 5, np.uint8), np.array([5., -5.]), 0., False)
        self.assertEqual(float(buffer.termination_buffer[4]), 0.)
        self.assertIsNotNone(retrieve_at(make_manager(), buffer, 4)[0])
        self.assertIsNone(retrieve_at(make_manager(), buffer, 5)[0])

    def test_lazy_rehash_and_group_weights_work_on_cpu(self):
        buffer = make_drama_buffer()
        fill_buffer(buffer, 6)
        manager = make_manager()
        manager._insert_into_bucket(2, 0, 3)
        manager._insert_into_bucket(3, 0, 3)
        result = retrieve_at(manager, buffer, 5, target=3)
        self.assertEqual(result[0].shape[0], 3)
        self.assertAlmostEqual(result[3], 1.0)
        self.assertEqual(result[-1][0], (5, 0))
        self.assertEqual(set(result[-1]), {(2, 0), (3, 0), (5, 0)})
        self.assertAlmostEqual(result[4][0], 0.7)
        self.assertAlmostEqual(sum(result[4]), 1.0)

    def test_rebuild_uses_cpu_view_and_only_written_slots(self):
        buffer = make_drama_buffer()
        fill_buffer(buffer, 4)
        manager = make_manager()
        manager.rebuild_all_hash_buckets(buffer, ConstantEncoder(), chunk_size=2)
        self.assertTrue(manager.index_to_bucket)
        self.assertTrue(all(0 <= pointer < 4 and env == 0 for pointer, env in manager.index_to_bucket))
        self.assertIsNotNone(retrieve_at(manager, buffer, 3)[0])

    def test_sample_indices_identify_returned_sequences(self):
        for tensors in [False, True]:
            with self.subTest(tensors=tensors):
                buffer = make_drama_buffer(tensors)
                fill_buffer(buffer, 6)
                self.assertEqual(len(buffer.sample(2, 3)), 4)
                result = buffer.sample(2, 3, return_indices=True)
                obs, actions, rewards, done, starts, envs = result
                self.assertEqual(tuple(obs.shape), (2, 3, 1, 2, 2))
                np.testing.assert_array_equal(envs, [0, 0])
                for row, start in enumerate(starts):
                    expected = torch.arange(int(start), int(start) + 3, dtype=torch.float32)
                    torch.testing.assert_close(actions[row, :, 0], expected)
                    torch.testing.assert_close(rewards[row], expected / 10)

    def test_sample_handles_small_and_empty_batches_and_chronological_wrap(self):
        for tensors in [False, True]:
            with self.subTest(tensors=tensors):
                buffer = make_drama_buffer(tensors)
                fill_buffer(buffer, 10)
                empty = buffer.sample(0, 3, return_indices=True)
                self.assertEqual(tuple(empty[0].shape), (0, 3, 1, 2, 2))
                self.assertEqual(tuple(empty[1].shape), (0, 3, 2))
                result = buffer.sample(20, 3, return_indices=True)
                self.assertEqual(len(result[0]), 20)
                for actions in result[1]:
                    torch.testing.assert_close(actions[1:, 0] - actions[:-1, 0], torch.ones(2))
                    self.assertGreaterEqual(actions[0, 0].item(), 2)
                    self.assertLessEqual(actions[-1, 0].item(), 9)


class RandomStateTests(unittest.TestCase):
    def test_branches_start_from_identical_python_numpy_and_torch_rng(self):
        random.seed(93)
        np.random.seed(42)
        torch.manual_seed(37)
        state = capture_rng_state()
        first = (random.random(), np.random.random(4), torch.rand(4))
        restore_rng_state(state)
        second = (random.random(), np.random.random(4), torch.rand(4))
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


class DramaTrainingContractTests(unittest.TestCase):
    def test_world_model_updates_collect_warmup_statistics_then_trigger_anchors(self):
        for log_metrics in [False, True]:
            with self.subTest(log_metrics=log_metrics):
                train_step = load_train_function("train_world_model_step", extra_namespace={
                    "is_logging_enabled": lambda logger: log_metrics,
                })
                buffer = make_drama_buffer(capacity=5)
                fill_buffer(buffer, 5)
                buffer.reward_buffer[:] = 10.
                manager = make_manager(context_length=2)
                manager.trigger_mode = "z_score"
                manager.ema_alpha = .5
                manager.z_score_threshold = .01
                for pointer in range(5):
                    manager._insert_into_bucket(pointer, 0, 3)

                def update(obs, action, reward, terminal, **kwargs):
                    self.assertTrue(kwargs["return_latent"])
                    self.assertEqual(kwargs["return_metrics"], log_metrics)
                    # Match the AMP feature dtype returned by the real world model.
                    features = action[..., :1].expand(-1, -1, 4).to(torch.bfloat16)
                    metrics = tuple(float(index) for index in range(8)) if log_metrics else None
                    return metrics, features

                def value(features):
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertEqual(features.dtype, torch.float32)
                    return features[..., :1]

                model = SimpleNamespace(device="cpu", update=mock.Mock(side_effect=update))
                agent = SimpleNamespace(gamma=.9, use_amp=False, value=mock.Mock(side_effect=value))
                logger = mock.Mock()
                train_step(buffer, model, 2, 5, logger, epoch=2, global_step=9,
                           agent=agent, retrieval_manager=manager, imagine_context_length=2,
                           is_warmup=True)
                self.assertEqual(len(manager.active_anchors), 0)
                # Values t=2,3 produce TD errors 10.7,10.6; two EMA updates use alpha=.5.
                np.testing.assert_allclose(manager.ema_mean, [10.65 * .75], rtol=1e-6)
                np.testing.assert_allclose(manager.ema_vd_mean, [.75])
                logger.log.assert_any_call("Retrieval/triggered_anchors_step", 0, global_step=9)
                train_step(buffer, model, 2, 5, logger, epoch=2, global_step=10,
                           agent=agent, retrieval_manager=manager, imagine_context_length=2,
                           is_warmup=False)
                self.assertEqual(list(manager.active_anchors), [((1, 0), 3)] * 4)
                self.assertEqual(agent.value.call_count, 4)
                self.assertEqual(model.update.call_count, 4)
                self.assertEqual([call.kwargs["epoch_step"] for call in model.update.call_args_list], [0, 1, 0, 1])
                logger.log.assert_any_call("Retrieval/triggered_anchors_step", 4, global_step=10)
                if not log_metrics:
                    self.assertFalse(any(call.args[0].startswith("WorldModel/") for call in logger.log.call_args_list))

    def test_world_model_updates_with_retrieval_disabled_skip_features_and_value_calls(self):
        train_step = load_train_function("train_world_model_step")
        for manager in [None, make_manager(enabled=False)]:
            with self.subTest(manager=manager):
                buffer = make_drama_buffer()
                fill_buffer(buffer, 6)
                buffer.sample = mock.Mock(wraps=buffer.sample)
                model = SimpleNamespace(device="cpu", update=mock.Mock(return_value=None))
                agent = SimpleNamespace(value=mock.Mock())
                logger = mock.Mock()
                train_step(buffer, model, 2, 5, logger, epoch=3, global_step=10,
                           agent=agent, retrieval_manager=manager, imagine_context_length=2)
                self.assertEqual(model.update.call_count, 3)
                for call in model.update.call_args_list:
                    self.assertFalse(call.kwargs["return_latent"])
                    self.assertFalse(call.kwargs["return_metrics"])
                for call in buffer.sample.call_args_list:
                    self.assertFalse(call.kwargs["return_indices"])
                agent.value.assert_not_called()
                logger.log.assert_not_called()

    def test_anchor_trigger_short_sequences_and_external_only_batches_are_noops(self):
        manager = make_manager(context_length=2)
        for sequence_length in [1, 2, 3]:
            with self.subTest(sequence_length=sequence_length):
                result = manager.add_batch_transitions(
                    torch.zeros(1, sequence_length), torch.ones(1, sequence_length),
                    torch.zeros(1, sequence_length), .99,
                    np.array([0]), np.array([0]), 8, skip_len=2,
                )
                self.assertEqual(result, 0)
        self.assertEqual(manager.add_batch_transitions(
            torch.zeros(1, 5), torch.ones(1, 5), torch.zeros(1, 5), .99,
            np.array([-1]), np.array([-1]), 8, skip_len=2,
        ), 0)
        self.assertFalse(manager.active_anchors)

    def test_shared_loop_runs_warmup_once_and_children_resume_at_next_step(self):
        class Environment:
            def __init__(self):
                self.action_space = SimpleNamespace(n=2, seed=lambda seed: None, sample=lambda: 1)
                self.steps = 0
                self.closed = False

            def reset(self):
                return np.zeros((2, 2, 1), np.uint8), {}

            def step(self, action):
                self.steps += 1
                return np.full((2, 2, 1), self.steps, np.uint8), 0., False, {"is_terminal": False}

            def close(self):
                self.closed = True

        config = AttrDict(
            BasicSettings=AttrDict(Env_name="memory_test", ImageSize=(2, 2), Seed=1),
            JointTrainAgent=AttrDict(
                Retrieval={"enable": "Both", "warmup_steps": 3, "hash_bits": 2, "use_pca": False},
                RealityContextLength=3, ImagineContextLength=3, SampleMaxSteps=5,
                SaveModels=False,
            ),
            Models=AttrDict(WorldModel=AttrDict(CategoricalDim=2, ClassDim=2)),
            Evaluate=AttrDict(DuringTraining=False),
        )
        environments = []
        def make_environment(*args, **kwargs):
            env = Environment()
            environments.append(env)
            return env
        train = load_train_function("joint_train_world_model_agent", extra_namespace={
            "os": os, "Path": Path, "deque": deque,
            "MemoryMaze": make_environment,
            "colorama": SimpleNamespace(Fore=SimpleNamespace(YELLOW=""), Style=SimpleNamespace(RESET_ALL="")),
            "tqdm": lambda values, **kwargs: values,
            "parse_retrieval_mode": parse_retrieval_mode,
            "RetrievalContextManager": RetrievalContextManager,
            "retrieval_warmup": load_train_function("retrieval_warmup"),
            "save_branch_checkpoint": checkpoint_helpers.save_branch_checkpoint,
            "save_final_models": save_final_models,
            "capture_rng_state": capture_rng_state,
            "restore_rng_state": restore_rng_state,
        })
        buffer = make_drama_buffer(continuous=False)
        buffer.world_model_warmup_length = 100
        buffer.behaviour_warmup_length = 100
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = train(config, directory, buffer, TinyTrainable(), TinyTrainable(), mock.Mock())
            self.assertEqual(environments[0].steps, 3)
            self.assertTrue(environments[0].closed)
            self.assertEqual(buffer.length, 3)
            self.assertEqual(buffer.termination_buffer[2], 0)
            self.assertTrue(buffer.episode_end_buffer[2])
            final_weights = {}
            for mode in [False, True]:
                branch_buffer = make_drama_buffer(continuous=False)
                branch_buffer.world_model_warmup_length = 100
                branch_buffer.behaviour_warmup_length = 100
                world, agent = TinyTrainable(), TinyTrainable()
                state, rng = checkpoint_helpers.load_branch_checkpoint(checkpoint, world, agent, branch_buffer)
                self.assertEqual(state["next_step"], 3)
                branch_config = copy.deepcopy(config)
                branch_config.JointTrainAgent.Retrieval["enable"] = mode
                logdir = Path(directory) / ("on" if mode else "off")
                train(branch_config, logdir, branch_buffer, world, agent, mock.Mock(), resume_state=state, resume_rng=rng)
                self.assertEqual(environments[-1].steps, 2)
                self.assertEqual(branch_buffer.length, 5)
                final_weights[mode] = (logdir / "ckpt" / "world_model.pth").read_bytes()
                saved = torch.load(logdir / "ckpt" / "world_model.pth", weights_only=True)
                tree_equal(self, world.state_dict(), saved)
                completion = json.loads((logdir / "ckpt" / "training_complete.json").read_text())
                self.assertEqual(completion["next_step"], 5)
            self.assertEqual((Path(directory) / "off" / "ckpt" / "world_model.pth").read_bytes(), final_weights[False])

    def test_fixed_warmup_switches_at_requested_step_and_dynamic_state_resumes(self):
        warmup = load_train_function("retrieval_warmup")
        self.assertTrue(warmup({"warmup_steps": 100}, 99, [], {}))
        self.assertFalse(warmup({"warmup_steps": 100}, 100, [], {}))
        self.assertFalse(warmup({"warmup_steps": 0}, 0, [], {}))
        with self.assertRaises(ValueError):
            warmup({"warmup_steps": -2}, 0, [], {})
        state = {"dynamic_warmup_met_step": 30}
        config = {"warmup_steps": -1, "min_warmup_steps": 5, "dynamic_warmup_target_steps": 100}
        self.assertTrue(warmup(config, 99, [], state))
        self.assertFalse(warmup(config, 100, [], state))
        self.assertTrue(state["warmup_finished"])
        self.assertFalse(warmup(config, 101, [], state))
        self.assertFalse(warmup({"warmup_steps": -1, "max_warmup_steps": 10}, 10, [], {}))

    def test_imagination_aligns_retrieved_rewards_terminals_and_continuous_actions(self):
        imagine = load_train_function("world_model_imagine_data")
        for tensors in [False, True]:
            for model_type in ["Transformer", "Mamba2"]:
                with self.subTest(tensors=tensors, model_type=model_type):
                    buffer = make_drama_buffer(tensors)
                    fill_buffer(buffer, 6)
                    buffer.termination_buffer[5] = 1
                    manager = make_manager()
                    manager.config.update(target=2, batch_size_reduction="retrieved")
                    manager._insert_into_bucket(3, 0, 3)
                    manager._insert_into_bucket(5, 0, 3)
                    manager.active_anchors.append(((5, 0), 3))
                    model = ConstantEncoder()
                    model.model = model_type
                    model.device = "cpu"
                    model.eval = lambda: None
                    outputs = tuple(torch.zeros(2, 2) for _ in range(6))
                    model.imagine_data = mock.Mock(return_value=outputs)
                    model.imagine_data2 = mock.Mock(return_value=outputs)
                    result = imagine(
                        buffer, model, SimpleNamespace(eval=lambda: None),
                        2, 3, 2, False, mock.Mock(), 100,
                        retrieval_manager=manager,
                    )
                    called = model.imagine_data if model_type == "Transformer" else model.imagine_data2
                    self.assertEqual(called.call_args.kwargs["imagine_batch_size"], 2)
                    actions = called.call_args.args[2]
                    self.assertEqual(tuple(actions.shape), (2, 3, 2))
                    torch.testing.assert_close(actions[:, :, 0], torch.tensor([[3., 4., 5.], [1., 2., 3.]]))
                    torch.testing.assert_close(result[4], torch.tensor([[.3, .4, .5], [.1, .2, .3]]))
                    torch.testing.assert_close(result[5], torch.tensor([[0., 0., 1.], [0., 0., 0.]]))
                    torch.testing.assert_close(result[8], torch.tensor([.7, .3]))
                    self.assertEqual(int(buffer.imagined_counter[1]), 1)
                    self.assertEqual(int(buffer.imagined_counter[3]), 1)

    def test_imagination_during_warmup_leaves_retrieval_queue_untouched(self):
        imagine = load_train_function("world_model_imagine_data")
        buffer = make_drama_buffer()
        fill_buffer(buffer, 6)
        manager = make_manager()
        manager.active_anchors.append(((5, 0), 3))
        model = SimpleNamespace(
            model="Transformer", device="cpu", eval=lambda: None,
            imagine_data=mock.Mock(return_value=tuple(torch.zeros(1, 2) for _ in range(6))),
        )
        result = imagine(
            buffer, model, SimpleNamespace(eval=lambda: None),
            1, 3, 2, False, mock.Mock(), 99,
            retrieval_manager=manager, is_warmup=True,
        )
        self.assertEqual(len(manager.active_anchors), 1)
        self.assertIsNone(result[8])
        self.assertEqual(model.imagine_data.call_args.kwargs["imagine_batch_size"], 1)


class DramaCheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_optimizer_schedulers_normalizers_replay_and_next_step(self):
        for tensor_storage in [False, True]:
            with self.subTest(tensor_storage=tensor_storage), tempfile.TemporaryDirectory() as directory:
                world, agent = TinyTrainable(), TinyTrainable()
                world.update_once()
                agent.update_once()
                buffer = make_drama_buffer(tensor_storage)
                fill_buffer(buffer, 10)
                buffer.sample(2, 3)
                buffer.sample(2, 3, imagine=True)
                manager = make_manager()
                manager._insert_into_bucket(7, 0, 3)
                manager.active_anchors.append(((7, 0), 3))
                manager.ema_mean[0] = 1.2
                loop = {"next_step": 100, "last_rebuild_step": 90, "warmup_finished": True,
                        "retrieval": manager.state_dict(), "episode_rewards": [1., 2.]}
                expected_world = checkpoint_helpers.model_training_state(world)
                expected_agent = checkpoint_helpers.model_training_state(agent)
                checkpoint_helpers.save_branch_checkpoint(directory, {"enable": "Both"}, world, agent, buffer, loop)
                expected_random = (random.random(), np.random.random(3), torch.rand(3))
                world.update_once()
                world.normalizer.ob_rms.mean.add_(100)
                self.assertTrue(torch.all(expected_world["normalizers"]["normalizer"]["mean"] < 100))
                first_states = []
                for _ in range(2):
                    resumed_world, resumed_agent = TinyTrainable(), TinyTrainable()
                    resumed_buffer = make_drama_buffer(not tensor_storage)
                    state, rng = checkpoint_helpers.load_branch_checkpoint(directory, resumed_world, resumed_agent, resumed_buffer)
                    tree_equal(self, expected_world, checkpoint_helpers.model_training_state(resumed_world))
                    tree_equal(self, expected_agent, checkpoint_helpers.model_training_state(resumed_agent))
                    self.assertEqual(state["next_step"], 100)
                    self.assertEqual(resumed_buffer.length, 8)
                    self.assertEqual(resumed_buffer.last_pointer, 1)
                    for name in ["obs_buffer", "action_buffer", "reward_buffer", "termination_buffer", "episode_end_buffer", "sampled_counter", "imagined_counter"]:
                        np.testing.assert_array_equal(np.asarray(getattr(buffer, name)), np.asarray(getattr(resumed_buffer, name)))
                    restored_manager = make_manager()
                    restored_manager.load_state_dict(state["retrieval"])
                    tree_equal(self, manager.state_dict(), restored_manager.state_dict())
                    restore_rng_state(rng)
                    actual_random = (random.random(), np.random.random(3), torch.rand(3))
                    tree_equal(self, expected_random, actual_random)
                    resumed_world.update_once()
                    first_states.append(checkpoint_helpers.model_training_state(resumed_world))
                tree_equal(self, first_states[0], first_states[1])


class StormTrainingContractTests(unittest.TestCase):
    def test_both_branches_before_collecting_boundary_transition_including_zero_warmup(self):
        from yacs.config import CfgNode
        StormReplayBuffer = load_module("storm_replay_for_tests", ROOT / "STORM" / "replay_buffer.py").ReplayBuffer

        class SupervisorStarted(Exception):
            pass

        for warmup in [0, 3]:
            with self.subTest(warmup=warmup):
                env = mock.Mock()
                env.action_space.sample.return_value = np.array([1])
                env.reset.return_value = (np.zeros((1, 2, 2, 1), np.uint8), {})
                env.step.return_value = (
                    np.zeros((1, 2, 2, 1), np.uint8), np.array([0.]),
                    np.array([False]), np.array([False]), {"life_loss": np.array([False])},
                )
                config = CfgNode({"JointTrainAgent": {"Retrieval": {
                    "enable": "Both", "warmup_steps": warmup, "hash_bits": 2, "use_pca": False,
                }}})
                saved, launched = mock.Mock(), mock.Mock(side_effect=SupervisorStarted)
                train = load_train_function("joint_train_world_model_agent", ROOT / "STORM" / "train.py", {
                    "os": SimpleNamespace(path=os.path, makedirs=mock.Mock()),
                    "args": SimpleNamespace(n="unit_test_Both"), "conf": config,
                    "build_vec_env": lambda *args, **kwargs: env,
                    "colorama": SimpleNamespace(Fore=SimpleNamespace(YELLOW="", GREEN=""), Style=SimpleNamespace(RESET_ALL="")),
                    "deque": deque, "tqdm": lambda values: values,
                    "RetrievalContextManager": lambda **kwargs: RetrievalContextManager(**kwargs, device="cpu"),
                    "save_full_checkpoint": saved, "launch_training_branches": launched,
                })
                buffer = StormReplayBuffer((2, 2, 1), 1, max_length=8, warmup_length=100, store_on_gpu=False)
                with mock.patch.object(torch, "save"), mock.patch.dict(sys.modules, {"wandb": SimpleNamespace(finish=lambda: None)}):
                    with self.assertRaises(SupervisorStarted):
                        train(
                            env_name="test", max_steps=5, num_envs=1, image_size=(2, 2),
                            replay_buffer=buffer, world_model=TinyTrainable(), agent=TinyTrainable(),
                            train_dynamics_every_steps=1, train_agent_every_steps=1,
                            batch_size=1, demonstration_batch_size=0, batch_length=3,
                            imagine_batch_size=1, imagine_demonstration_batch_size=0,
                            imagine_context_length=3, imagine_batch_length=2,
                            save_every_steps=100, seed=1, logger=mock.Mock(),
                            branch_commands=([sys.executable, "--mode", "True"], [sys.executable, "--mode", "False"]),
                        )
                self.assertEqual(env.step.call_count, warmup)
                self.assertEqual(buffer.length, warmup)
                self.assertEqual(saved.call_args.args[4], warmup)
                self.assertTrue(saved.call_args.kwargs["shared_warmup"])
                env.close.assert_called_once()
                checkpoint, enabled, disabled = launched.call_args.args
                self.assertEqual(enabled[-2:], ["--resume_from", checkpoint])
                self.assertEqual(disabled[-2:], ["--resume_from", checkpoint])
                if warmup:
                    np.testing.assert_array_equal(buffer.termination_buffer[buffer.last_pointer], [1])


class BranchSupervisorTests(unittest.TestCase):
    def write_manifest(self, directory, commands):
        manifest = Path(directory) / "branches.json"
        manifest.write_text(json.dumps({"cwd": directory, "commands": commands}))
        return manifest

    def run_supervisor(self, manifest):
        return subprocess.run([
            sys.executable, str(ROOT / "training_branches.py"),
            "--supervise", str(manifest),
        ], capture_output=True, text=True, timeout=10)

    def test_launcher_replaces_cuda_parent_with_lightweight_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            enabled = [sys.executable, "train.py", "--retrieval", "True"]
            disabled = [sys.executable, "train.py", "--retrieval", "False"]
            with mock.patch("training_branches.os.execv") as execv:
                launch_training_branches(directory, enabled, disabled)
            execv.assert_called_once()
            executable, args = execv.call_args.args
            self.assertEqual(executable, sys.executable)
            self.assertEqual(args[:3], [sys.executable, str(ROOT / "training_branches.py"), "--supervise"])
            manifest = json.loads(Path(args[3]).read_text())
            self.assertEqual(manifest["commands"], {"retrieval_on": enabled, "retrieval_off": disabled})
            self.assertEqual(manifest["execution_order"], ["retrieval_off", "retrieval_on"])

    def test_children_run_false_then_true_from_same_unchanged_saved_state(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory) / "shared.json"
            original = json.dumps({"next_step": 101, "optimizer_updates": 17})
            shared.write_text(original)
            child = "\n".join([
                "import json, pathlib, sys, time",
                "name = sys.argv[1]",
                "state = json.loads(pathlib.Path('shared.json').read_text())",
                "if name == 'on':",
                "    assert pathlib.Path('off.json').exists()",
                "    assert not pathlib.Path('off.running').exists()",
                "    progress = json.loads(pathlib.Path('branch_results.json').read_text())",
                "    assert progress['exit_codes']['retrieval_off'] == 0",
                "    assert progress['active_branch'] == 'retrieval_on'",
                "else:",
                "    assert not pathlib.Path('on.json').exists()",
                "pathlib.Path(name + '.running').touch()",
                "time.sleep(0.05)",
                "state['initial_step'] = state['next_step']",
                "state['next_step'] += 100",
                "state['optimizer_updates'] += 1",
                "pathlib.Path(name + '.json').write_text(json.dumps(state))",
                "pathlib.Path(name + '.running').unlink()",
            ])
            manifest = self.write_manifest(directory, {
                "retrieval_on": [sys.executable, "-c", child, "on"],
                "retrieval_off": [sys.executable, "-c", child, "off"],
            })
            result = self.run_supervisor(manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            on = json.loads((Path(directory) / "on.json").read_text())
            off = json.loads((Path(directory) / "off.json").read_text())
            self.assertEqual(on, {"initial_step": 101, "next_step": 201, "optimizer_updates": 18})
            self.assertEqual(on, off)
            self.assertEqual(shared.read_text(), original)
            summary = json.loads((Path(directory) / "branch_results.json").read_text())
            self.assertEqual(summary["exit_codes"], {"retrieval_on": 0, "retrieval_off": 0})

    def test_failed_false_does_not_start_true(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory, {
                "retrieval_on": [sys.executable, "-c", "from pathlib import Path; Path('on.started').touch()"],
                "retrieval_off": [sys.executable, "-c", "raise SystemExit(7)"],
            })
            result = self.run_supervisor(manifest)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertFalse((Path(directory) / "on.started").exists())
            summary = json.loads((Path(directory) / "branch_results.json").read_text())
            self.assertEqual(summary["supervisor_exit_code"], 7)
            self.assertEqual(summary["exit_codes"], {"retrieval_off": 7})

    def test_true_failure_preserves_completed_false(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory, {
                "retrieval_off": [sys.executable, "-c", "from pathlib import Path; Path('off.weights').write_bytes(b'final-false')"],
                "retrieval_on": [sys.executable, "-c", "raise SystemExit(9)"],
            })
            result = self.run_supervisor(manifest)
            self.assertEqual(result.returncode, 9, result.stderr)
            self.assertEqual((Path(directory) / "off.weights").read_bytes(), b"final-false")
            summary = json.loads((Path(directory) / "branch_results.json").read_text())
            self.assertEqual(summary["exit_codes"], {"retrieval_off": 0, "retrieval_on": 9})

    def test_interrupt_during_true_preserves_false_and_shared_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shared.weights").write_bytes(b"warmup-state")
            manifest = self.write_manifest(directory, {
                "retrieval_off": [sys.executable, "-c", "from pathlib import Path; Path('off.weights').write_bytes(b'final-false')"],
                "retrieval_on": [sys.executable, "-c", "from pathlib import Path; import time; assert Path('off.weights').exists(); Path('on.started').touch(); time.sleep(30)"],
            })
            process = subprocess.Popen([sys.executable, str(ROOT / "training_branches.py"), "--supervise", str(manifest)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / "on.started").exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / "on.started").exists())
                progress = json.loads((root / "branch_results.json").read_text())
                self.assertEqual(progress["exit_codes"], {"retrieval_off": 0})
                process.terminate()
                process.communicate(timeout=5)
                self.assertEqual(process.returncode, 128 + signal.SIGTERM)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=5)
            self.assertEqual((root / "off.weights").read_bytes(), b"final-false")
            self.assertEqual((root / "shared.weights").read_bytes(), b"warmup-state")
            summary = json.loads((root / "branch_results.json").read_text())
            self.assertEqual(summary["exit_codes"]["retrieval_off"], 0)
            self.assertEqual(summary["exit_codes"]["retrieval_on"], -signal.SIGTERM)


class FinalModelTests(unittest.TestCase):
    def test_final_weights_and_completion_marker_are_saved_for_both_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            world, agent = TinyTrainable(), TinyTrainable()
            world.update_once()
            for name, filenames in [("storm", ("world_model_final.pth", "agent_final.pth")),
                                    ("drama", ("world_model.pth", "agent.pth"))]:
                target = Path(directory) / name
                save_final_models(target, world, agent, 103, filenames=filenames)
                tree_equal(self, world.state_dict(), torch.load(target / filenames[0], weights_only=True))
                tree_equal(self, agent.state_dict(), torch.load(target / filenames[1], weights_only=True))
                self.assertEqual(json.loads((target / "training_complete.json").read_text())["next_step"], 103)

    def test_interrupted_true_save_does_not_modify_false_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            world, agent = TinyTrainable(), TinyTrainable()
            target = Path(directory) / "off"
            save_final_models(target, world, agent, 103)
            before = {path.name: path.read_bytes() for path in target.iterdir()}
            with mock.patch.object(torch, "save", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    save_final_models(Path(directory) / "on", world, agent, 104)
            self.assertEqual({path.name: path.read_bytes() for path in target.iterdir()}, before)
            self.assertFalse((Path(directory) / "on" / "training_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
