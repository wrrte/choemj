"""Real TWISTER network/optimizer regressions with a deterministic toy environment.

Run in an environment with TWISTER's torch, torchvision, and logging dependencies:
    python -m unittest discover -s tests -p 'test_twister_retrieval.py' -v
"""

import copy
import importlib
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "207843cb645fb4b6471bb3deb1c94ec18921fede"
sys.path.insert(0, str(ROOT / "TWISTER"))
sys.path.insert(0, str(ROOT))

# Only the simulator package is substituted; all model, distribution, replay,
# attention, loss, and optimizer implementations below are the real modules.
env_module = types.ModuleType("nnet.envs")
env_module.__path__ = [str(ROOT / "TWISTER/nnet/envs")]
sys.modules["nnet.envs"] = env_module
import nnet
from nnet.structs import AttrDict
from nnet.datasets.retrieval_replay import RetrievalReplay
from nnet.models.twister_retrieval import TWISTERRetrieval
from retrieval import RetrievalContextManager

env_module.wrappers = importlib.import_module("nnet.envs.wrappers")
env_module.wrappers.BatchEnv = importlib.import_module("nnet.envs.wrappers.batch_env").BatchEnv


class ToyEnv:
    def __init__(self, **kwargs):
        self.num_actions = 3
        self.action_repeat = kwargs.get("action_repeat", 4)
        self.clip_low, self.clip_high = -1, 1
        self.t = 0

    def obs_space(self):
        return [(3, 64, 64)]

    def reset(self):
        self.t = 0
        return self.observation(first=True)

    def observation(self, first=False):
        return AttrDict(state=torch.full((3, 64, 64), 20 + self.t, dtype=torch.uint8),
                        reward=torch.tensor(float(self.t % 3)), done=torch.tensor(float(self.t == 9)),
                        is_first=torch.tensor(float(first)), is_last=torch.tensor(float(self.t == 9)),
                        error=torch.tensor(False))

    def step(self, action):
        self.t += 1
        return self.observation()

    def sample(self):
        return torch.nn.functional.one_hot(torch.randint(3, ()), 3).float()


env_module.atari = types.SimpleNamespace(AtariEnv=ToyEnv)
env_module.dm_control = types.SimpleNamespace(dm_control_dict={"Toy": ToyEnv})


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def assert_tree(test, a, b):
    if isinstance(a, torch.nn.Module):
        assert_tree(test, a.state_dict(), b.state_dict())
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_tree(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree(test, x, y)
    else:
        test.assertEqual(a, b)


def make_model(root, enabled=False, cls=None, **overrides):
    config = dict(
        batch_size=2, L=8, H=2, num_envs=2, eval_episodes=0,
        dim_cnn=2, model_stoch_size=2, model_discrete=4,
        model_hidden_size=16, action_hidden_size=16, value_hidden_size=16,
        reward_hidden_size=16, discount_hidden_size=16, action_layers=1,
        value_layers=1, reward_layers=1, discount_layers=1,
        att_context_left=3, num_blocks_trans=2, num_heads_trans=2,
        contrastive_steps=2, contrastive_hidden_size=16, contrastive_out_size=16,
        contrastive_layers=1, pre_fill_steps=0, env_step_period=10**12,
        precision="float32", buffer_capacity=30,
    )
    if cls is None:
        cls = nnet.models.TWISTER
        config.update(retrieval_enabled=enabled, retrieval=dict(
            context_length=3, warmup_steps=0, hash_bits=1, use_pca=False,
            threshold=0.0, anchor_offset=0, target=2, max_anchors=2,
            max_contexts=4, global_rebuild_enable=False, chunk_size=4))
    config.update(overrides)
    model = cls("atari100k-toy", override_config=config)
    model.compile()
    replay = nnet.datasets.ReplayBuffer(2, root, 30, 1, config["L"], save_trajectories=True)
    model.set_replay_buffer(replay)
    return model


def fill(model, count=13):
    for _ in range(count):
        model.env_step()


def train(model, inputs):
    return model.train_step(copy.deepcopy(inputs), [], torch.float32,
                            torch.cuda.amp.GradScaler(enabled=False), 1, 0, False)


def baseline_class():
    source = subprocess.check_output(
        ["git", "show", BASELINE_REVISION + ":nnet/models/twister.py"], cwd=ROOT / "TWISTER", text=True)
    namespace = {"__name__": "nnet.models._baseline_twister"}
    exec(compile(source, "baseline_twister.py", "exec"), namespace)
    return namespace["TWISTER"]


class TWISTERRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disabled_matches_original_initialization_rng_losses_and_adam_updates(self):
        with tempfile.TemporaryDirectory() as root:
            seed(42)
            original = make_model(root, cls=baseline_class())
            fill(original)
            original_rng = rng_state()
            seed(42)
            disabled = make_model(root)
            fill(disabled)
            assert_tree(self, original_rng, rng_state())
            assert_tree(self, original.state_dict(), disabled.state_dict())
            self.assertIsNone(disabled.retrieval)
            self.assertIsNone(disabled.replay_buffer.retrieval)
            for _ in range(2):
                inputs = original.replay_buffer.collate_fn([original.replay_buffer.sample() for _ in range(2)])["inputs"]
                state = rng_state()
                expected = train(original, inputs)
                expected_rng = rng_state()
                restore_rng(state)
                actual = train(disabled, inputs)
                assert_tree(self, expected, actual)
                assert_tree(self, original.state_dict(), disabled.state_dict())
                assert_tree(self, original.optimizer["world_model"].state_dict(), disabled.optimizer["world_model"].state_dict())
                assert_tree(self, original.optimizer["actor_model"].state_dict(), disabled.optimizer["actor_model"].state_dict())
                assert_tree(self, original.optimizer["critic_model"].state_dict(), disabled.optimizer["critic_model"].state_dict())
                assert_tree(self, expected_rng, rng_state())

    def test_enabled_real_training_adds_contexts_without_changing_world_update(self):
        with tempfile.TemporaryDirectory() as root:
            seed(3)
            baseline = make_model(root)
            fill(baseline)
            seed(3)
            model = make_model(root, enabled=True)
            fill(model)
            sample = model.replay_buffer.collate_fn([model.replay_buffer.sample() for _ in range(2)])["inputs"]
            state = rng_state()
            expected = train(baseline, sample[:6])
            restore_rng(state)
            actual = train(model, sample)
            for key in expected[0]:
                if key.startswith("world_model_"):
                    assert_tree(self, expected[0][key], actual[0][key])
            assert_tree(self, baseline.world_model.state_dict(), model.world_model.state_dict())
            self.assertGreater(model.infos["retrieval_contexts"], 0)
            self.assertEqual(model.detached_feats.shape[0], 16 + model.infos["retrieval_contexts"])
            self.assertAlmostEqual(model.retrieval.sample_weights.sum().item(), 16 + model.infos["retrieval_anchors"])
            self.assertTrue(all(torch.isfinite(value["loss"]) for value in model.actor_model.added_losses.values()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            train(model, sample)

    def test_hash_probs_are_deterministic_and_rng_neutral(self):
        with tempfile.TemporaryDirectory() as root:
            model = make_model(root, enabled=True)
            obs = torch.rand(2, 3, 3, 64, 64)
            state = rng_state()
            with model.retrieval.evaluation():
                first = model.encode_obs(obs)
                second = model.encode_obs(obs)
            assert_tree(self, state, rng_state())
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(first.reshape(2, 3, 2, 4).sum(-1), torch.ones(2, 3, 2))

    def test_sparse_replay_deduplicates_windows_rejects_boundaries_and_prunes(self):
        with tempfile.TemporaryDirectory() as root:
            replay = nnet.datasets.ReplayBuffer(1, root, 2, 1, 4, save_trajectories=False)
            replay.enable_retrieval(2)
            for env in range(2):
                for t in range(7):
                    replay.append_step([
                        torch.full((3, 2, 2), t + env * 10, dtype=torch.uint8),
                        torch.tensor([t, env], dtype=torch.float32), torch.tensor(float(t)),
                        torch.tensor(float(t == 4)), torch.tensor(float(t in (0, 5))), torch.tensor(0)], env)
            view = replay.retrieval_view()
            self.assertEqual(len(view.steps), 5)
            self.assertFalse(view.is_valid_context(3, 0, 2))
            self.assertTrue(view.is_valid_context(4, 1, 3))
            self.assertFalse(view.is_valid_context(5, 1, 3))
            self.assertTrue(view.is_valid_context(6, 1, 2))
            actions = view.context_field([(6, 1)], 2, 1, "cpu")
            torch.testing.assert_close(actions, torch.tensor([[[5., 1.], [6., 1.]]]))
            manager = RetrievalContextManager(2, dict(enable=True, hash_bits=1, context_length=2), 1, device="cpu")
            manager._insert_into_bucket(1, 0, 0)
            world = types.SimpleNamespace(encode_obs=lambda obs, **kwargs: torch.ones(obs.shape[0], 1, 1))
            view.update_hashes(manager, world, 2)
            self.assertNotIn((1, 0), manager.index_to_bucket)
            self.assertTrue(set(manager.index_to_bucket) <= set(view.steps))

    def test_weighted_loss_normalizes_groups_and_zero_weight_gradients(self):
        retrieval = object.__new__(TWISTERRetrieval)
        retrieval.sample_weights = torch.tensor([1., .5, .25, .25, 0.])
        losses = torch.tensor([[2., 4.], [4., 6.], [6., 8.], [8., 10.], [100., 100.]], requires_grad=True)
        result = retrieval.loss_mean(losses)
        torch.testing.assert_close(result, torch.tensor((3 + .5 * 5 + .25 * 7 + .25 * 9) / 2))
        result.backward()
        torch.testing.assert_close(losses.grad[-1], torch.zeros(2))
        retrieval.sample_weights = None
        torch.testing.assert_close(retrieval.loss_mean(losses), losses.mean(), rtol=0, atol=0)

    def test_arrival_rewards_and_reset_edges_are_aligned_before_td_trigger(self):
        with tempfile.TemporaryDirectory() as root:
            model = make_model(root, enabled=True)
            fill(model)
            sample = model.replay_buffer.collate_fn([model.replay_buffer.sample() for _ in range(2)])["inputs"]
            model.retrieval.ensure_manager()
            inputs = model.preprocess_inputs(copy.deepcopy(sample[:6]), True)
            model.world_model(inputs)
            inputs[2] = torch.arange(16, dtype=torch.float32).reshape(2, 8)
            inputs[3].zero_()
            inputs[4].zero_()
            inputs[4][0, 5] = 1
            with model.retrieval.evaluation(), mock.patch.object(model.retrieval.manager, "add_batch_transitions", return_value=0) as trigger:
                model.retrieval._trigger(inputs, sample[6], False)
            args, kwargs = trigger.call_args
            torch.testing.assert_close(args[1][:, :-1], inputs[2][:, 1:])
            torch.testing.assert_close(args[2][:, :-1], inputs[3][:, 1:])
            self.assertFalse(kwargs["transition_mask"][0, 4])
            self.assertFalse(kwargs["transition_mask"][0, 5])

    def test_checkpoint_restores_index_statistics_and_rebuilds_with_new_stream_gap(self):
        with tempfile.TemporaryDirectory() as root:
            model = make_model(root, enabled=True)
            fill(model)
            sample = model.replay_buffer.collate_fn([model.replay_buffer.sample() for _ in range(2)])["inputs"]
            train(model, sample)
            model.retrieval.manager.ema_mean[:] = 7
            path = str(Path(root) / "checkpoint.ckpt")
            model.save(path)
            resumed = make_model(root, enabled=True)
            resumed.load(path, verbose=False)
            assert_tree(self, model.state_dict(), resumed.state_dict())
            self.assertEqual(model.replay_buffer.retrieval.windows, resumed.replay_buffer.retrieval.windows)
            self.assertTrue((resumed.retrieval.manager.ema_mean == 7).all())
            self.assertFalse(resumed.retrieval.hash_built)
            self.assertFalse(resumed.replay_buffer.streams)
            train(resumed, sample)
            self.assertTrue(resumed.retrieval.hash_built)
            self.assertTrue(set(resumed.retrieval.manager.index_to_bucket) <= set(resumed.replay_buffer.retrieval.steps))

    def test_warmup_and_empty_retrieval_keep_original_imagination_size(self):
        with tempfile.TemporaryDirectory() as root:
            for config in (dict(warmup_steps=100000), dict(warmup_steps=0, max_contexts=0)):
                model = make_model(root, enabled=True, retrieval=dict(context_length=3, hash_bits=1, use_pca=False, **config))
                fill(model)
                sample = model.replay_buffer.collate_fn([model.replay_buffer.sample() for _ in range(2)])["inputs"]
                train(model, sample)
                self.assertIsNone(model.retrieval.sample_weights)
                self.assertEqual(model.detached_feats.shape[0], 16)

    def test_retrieved_final_cache_masks_and_terminal_match_world_model_flattening(self):
        with tempfile.TemporaryDirectory() as root:
            for length in (1, 3, 5):
                with self.subTest(context_length=length):
                    seed(7)
                    model = make_model(root, enabled=True)
                    fill(model)
                    model.retrieval.config["context_length"] = length
                    view = model.replay_buffer.retrieval_view()
                    indices = [(9, 0)]  # A terminal observation is a valid endpoint.
                    self.assertTrue(view.is_valid_context(9, 0, length))
                    fields = [view.context_field(indices, length, field, "cpu") for field in range(6)]
                    actions_before = fields[1].clone()
                    inputs = model.replay_buffer.collate_fn([model.replay_buffer.sample() for _ in range(2)])["inputs"][:6]
                    inputs = model.preprocess_inputs(inputs, True)
                    with model.retrieval.evaluation():
                        model.world_model(inputs)
                        original_posts = copy.deepcopy(model.detached_posts)
                        original_firsts = model.detached_is_firsts.clone()
                        original_hidden_firsts = model.detached_is_firsts_hidden.clone()
                        # Compare to the existing flattening implementation on
                        # exactly this context, including its RNG sequence.
                        state = rng_state()
                        model.config.L = length
                        model.config.contrastive_steps = 1
                        model.world_model(model.preprocess_inputs(copy.deepcopy(fields), True))
                        expected_posts = model.detached_posts
                        expected_first = model.detached_is_firsts[-1:]
                        expected_hidden_firsts = model.detached_is_firsts_hidden[-1:]
                        model.config.L = 8
                        model.detached_posts = original_posts
                        model.detached_is_firsts = original_firsts
                        model.detached_is_firsts_hidden = original_hidden_firsts
                        restore_rng(state)
                        model.retrieval._append_contexts(fields[0].float() / 255, fields[1], indices, [1.], inputs[3])
                    for key in expected_posts:
                        if key == "hidden":
                            for actual_block, expected_block in zip(model.detached_posts[key], expected_posts[key]):
                                for actual, expected in zip(actual_block, expected_block):
                                    torch.testing.assert_close(actual[-1:], expected[-1:], rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(model.detached_posts[key][-1:], expected_posts[key][-1:], rtol=0, atol=0)
                    torch.testing.assert_close(model.detached_is_firsts[-1:], expected_first, rtol=0, atol=0)
                    torch.testing.assert_close(model.detached_is_firsts_hidden[-1:], expected_hidden_firsts, rtol=0, atol=0)
                    torch.testing.assert_close(fields[1], actions_before, rtol=0, atol=0)
                    self.assertEqual(model.retrieval.initial_dones[-1].item(), 1.)
                    model.rssm.eval()
                    model.actor_model(inputs)
                    self.assertEqual(torch.count_nonzero(model.detached_weights[-1]).item(), 0)

    def test_continuous_dynamics_actor_training_and_pca_rebuild(self):
        with tempfile.TemporaryDirectory() as root:
            seed(8)
            model = make_model(root, enabled=True, policy_discrete=False, actor_grad="dynamics")
            fill(model)
            model.retrieval.config.update(use_pca=True, max_pca_samples=12, global_rebuild_enable=True,
                                          global_rebuild_cooldown=0, global_rebuild_threshold=1.1)
            sample = model.replay_buffer.collate_fn([model.replay_buffer.sample() for _ in range(2)])["inputs"]
            train(model, sample)
            self.assertGreater(model.infos["retrieval_contexts"], 0)
            self.assertIsNotNone(model.retrieval.manager.hash_mean)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertFalse(model.retrieval.manager.active_anchors)

    def test_legacy_checkpoint_can_enable_retrieval_without_cross_window_history(self):
        with tempfile.TemporaryDirectory() as root:
            original = make_model(root)
            fill(original)
            path = str(Path(root) / "legacy.ckpt")
            original.save(path)
            resumed = make_model(root, enabled=True)
            resumed.load(path, verbose=False)
            view = resumed.replay_buffer.retrieval_view()
            self.assertEqual(len(view.steps), len(resumed.replay_buffer.ram_buffer) * 8)
            bases = sorted(base for base, env in view.windows.values())
            self.assertFalse(view.is_valid_context(bases[1], 0, 2))
            sample = resumed.replay_buffer.collate_fn([resumed.replay_buffer.sample() for _ in range(2)])["inputs"]
            train(resumed, sample)
            self.assertTrue(resumed.retrieval.hash_built)

    def test_transition_mask_excludes_invalid_maxima_and_ema_samples(self):
        manager = RetrievalContextManager(1, dict(enable=True, hash_bits=1, threshold=1.,
                                                 context_length=1, anchor_offset=0), 1, device="cpu")
        for pointer in range(6):
            manager._insert_into_bucket(pointer, 0, 0)
        reward = torch.tensor([[0., 100., 2., 0., 0., 0.]])
        mask = torch.tensor([[True, False, True, True, True]])
        triggered = manager.add_batch_transitions(torch.zeros(1, 6), reward, torch.zeros(1, 6),
                                                 .9, np.array([0]), np.array([0]), 100,
                                                 skip_len=1, transition_mask=mask)
        self.assertEqual(triggered, 1)
        self.assertEqual(manager.active_anchors[0][0], (3, 0))
        manager.trigger_mode = "z_score"
        manager.active_anchors.clear()
        manager.add_batch_transitions(torch.zeros(1, 6), reward, torch.zeros(1, 6),
                                      .9, np.array([0]), np.array([0]), 100,
                                      skip_len=1, is_warmup=True, transition_mask=mask)
        self.assertAlmostEqual(manager.ema_mean[0], manager.ema_alpha * 2 / 3, places=8)
        self.assertFalse(manager.active_anchors)
        previous_mean = manager.ema_mean.copy()
        manager.add_batch_transitions(torch.zeros(1, 6), reward, torch.zeros(1, 6),
                                      .9, np.array([0]), np.array([0]), 100,
                                      skip_len=1, transition_mask=torch.zeros_like(mask))
        np.testing.assert_array_equal(manager.ema_mean, previous_mean)


if __name__ == "__main__":
    unittest.main()
