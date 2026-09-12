"""CPU checks for Drama retrieval losses and model interfaces without Triton imports."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from einops import rearrange


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, namespace, names=None):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and (names is None or node.name in names)]
    exec(compile(tree, str(path), "exec"), namespace)


loss_spec = importlib.util.spec_from_file_location(
    "drama_retrieval_losses", ROOT / "Drama/sub_models/functions_losses.py")
losses = importlib.util.module_from_spec(loss_spec)
loss_spec.loader.exec_module(losses)

namespace = {
    "torch": torch, "nn": nn, "np": np, "copy": copy,
    "F": torch.nn.functional, "distributions": torch.distributions,
    "Normal": torch.distributions.Normal, "OneHotCategorical": torch.distributions.OneHotCategorical,
    "profile": lambda function: function, "rearrange": rearrange,
    "SymLogTwoHotLoss": losses.SymLogTwoHotLoss, "weighted_mean": losses.weighted_mean,
    "is_logging_enabled": lambda logger: False,
}
load_definitions(ROOT / "Drama/utils.py", namespace, {"EMAScalar"})
load_definitions(ROOT / "Drama/agents.py", namespace)
namespace["agents"] = SimpleNamespace(ActorCriticAgent=namespace["ActorCriticAgent"])
load_definitions(ROOT / "Drama/sub_models/world_models.py", namespace, {"WorldModel"})


class NoopScheduler:
    def step(self):
        pass

    def dampen(self):
        pass


class FakeInferenceParams:
    def __init__(self, **kwargs):
        self.seqlen_offset = 0


namespace["InferenceParams"] = FakeInferenceParams


def make_agent(name, continuous=False):
    cls = namespace[name]
    agent = cls.__new__(cls)
    nn.Module.__init__(agent)
    agent.use_amp = False
    agent.action_dim = 3
    agent.is_discrete = not continuous
    agent.gamma = 0.99
    agent.lambd = 0.95
    agent.entropy_coef = 0.01
    agent.max_grad_norm = 100
    agent.unimix_ratio = 0
    if continuous:
        agent.actor_mean = nn.Linear(8, 3)
        agent.actor_log_std = nn.Parameter(torch.zeros(1, 3))
    else:
        agent.actor = nn.Linear(8, 3)
    agent.critic = nn.Linear(8, 255)
    agent.slow_critic = copy.deepcopy(agent.critic)
    agent.symlog_twohot_loss = losses.SymLogTwoHotLoss(255, -20, 20)
    agent.lowerbound_ema = namespace["EMAScalar"](0.99)
    agent.upperbound_ema = namespace["EMAScalar"](0.99)
    agent.optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    agent.scaler = torch.amp.GradScaler("cuda", enabled=False)
    agent.lr_scheduler = NoopScheduler()
    agent.warmup_scheduler = NoopScheduler()
    if name == "PPOAgent":
        agent.K_epochs = 2
        agent.minibatch_size = 4
        agent.eps_clip = 0.2
        agent.c1 = 1.0
        agent.c2 = 0.01
    return agent


class RetrievalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_weighted_twohot_loss_matches_context_mixture(self):
        torch.manual_seed(1)
        output = torch.randn(3, 4, 255, requires_grad=True)
        target = torch.randn(3, 4)
        weights = torch.tensor([0.75, 0.25, 0.0])
        loss = losses.SymLogTwoHotLoss(255, -20, 20)
        actual = loss(output, target, weights=weights)
        expected = 0.75 * loss(output[:1], target[:1]) + 0.25 * loss(output[1:2], target[1:2])
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(torch.count_nonzero(output.grad[2]).item(), 0)
        torch.testing.assert_close(loss(output, target), loss(output, target, weights=torch.ones(3)))

    def test_weighted_actor_critic_and_ppo_updates(self):
        for name, continuous in (("ActorCriticAgent", False), ("PPOAgent", False), ("PPOAgent", True)):
            with self.subTest(name=name, continuous=continuous):
                torch.manual_seed(4)
                agent = make_agent(name, continuous)
                latent = torch.randn(4, 7, 8)
                action, old_logits = agent.sample(latent[:, :-1])
                reward = torch.randn(4, 6)
                termination = torch.zeros(4, 6)
                actor = agent.actor_mean if continuous else agent.actor
                old_actor_weight = actor.weight.detach().clone()
                agent.update(latent, action, old_logits, None, None, None,
                             reward, termination, None, 0,
                             weights=torch.tensor([0.5, 0.25, 0.25, 1.0]))
                self.assertFalse(torch.equal(actor.weight, old_actor_weight))
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in agent.parameters()))

    def test_continuous_env_action_removes_singleton_time_axis(self):
        agent = make_agent("PPOAgent", continuous=True)
        env_action, device_action = agent.sample_as_env_action(
            torch.zeros(2, 1, 8), return_device_action=True)
        self.assertEqual(env_action.shape, (2, 3))
        self.assertEqual(tuple(device_action.shape), (2, 3))

    def test_encode_obs_probabilities_are_deterministic(self):
        cls = namespace["WorldModel"]
        world_model = cls.__new__(cls)
        nn.Module.__init__(world_model)
        world_model.use_amp = False
        world_model.encoder = nn.Flatten(start_dim=2)
        world_model.dist_head = SimpleNamespace(
            forward_post=lambda embedding: embedding.reshape(*embedding.shape[:2], 2, 3))
        obs = torch.randn(2, 3, 1, 2, 3)
        actual = world_model.encode_obs(obs, sample_mode="probs")
        expected = obs.reshape(2, 3, 2, 3).softmax(dim=-1).reshape(2, 3, 6)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, world_model.encode_obs(obs, sample_mode="probs"))
        with self.assertRaises(ValueError):
            world_model.encode_obs(obs, sample_mode="invalid")

    def test_imagination_buffers_resize_for_discrete_and_continuous_actions(self):
        cls = namespace["WorldModel"]
        for discrete in (True, False):
            with self.subTest(discrete=discrete):
                world_model = cls.__new__(cls)
                nn.Module.__init__(world_model)
                world_model.is_discrete = discrete
                world_model.action_dim = 3
                world_model.stoch_flattened_dim = 6
                world_model.hidden_state_dim = 4
                world_model.imagine_batch_size = -1
                world_model.imagine_batch_length = -1
                for batch_size in (8, 3, 11):
                    world_model.init_imagine_buffer(batch_size, 5, torch.float32, "cpu")
                    expected = (batch_size, 5) if discrete else (batch_size, 5, 3)
                    self.assertEqual(tuple(world_model.action_buffer.shape), expected)
                    self.assertEqual(tuple(world_model.sample_buffer.shape), (batch_size, 6, 6))

    def test_imagination_video_supports_small_and_uneven_retrieval_batches(self):
        cls = namespace["WorldModel"]
        for method in ("imagine_data", "imagine_data2"):
            for batch_size in (1, 3, 5):
                with self.subTest(method=method, batch_size=batch_size):
                    world_model = cls.__new__(cls)
                    nn.Module.__init__(world_model)
                    world_model.is_discrete = True
                    world_model.action_dim = 3
                    world_model.stoch_flattened_dim = 6
                    world_model.hidden_state_dim = 4
                    world_model.imagine_batch_size = world_model.imagine_batch_length = -1
                    world_model.tensor_dtype = torch.float32
                    world_model.device = "cpu"
                    world_model.use_amp = world_model.use_cg = False
                    world_model.encode_obs = lambda obs: torch.zeros(*obs.shape[:2], 6)
                    world_model.dist_head = SimpleNamespace(
                        forward_prior=lambda feat: torch.zeros(*feat.shape[:2], 2, 3))
                    world_model.image_decoder = lambda sample: torch.zeros(*sample.shape[:2], 1, 2, 3)
                    world_model.reward_decoder = lambda feat: torch.zeros(*feat.shape[:2], 255)
                    world_model.termination_decoder = lambda feat: torch.zeros(*feat.shape[:2])
                    world_model.symlog_twohot_loss_func = losses.SymLogTwoHotLoss(255, -20, 20)
                    if method == "imagine_data":
                        world_model.sequence_model = SimpleNamespace(reset_kv_cache_list=lambda *args, **kwargs: None)
                        world_model.predict_next = lambda sample, action, log_video: (
                            torch.zeros(*sample.shape[:2], 1, 2, 3),
                            torch.zeros(*sample.shape[:2]), torch.zeros(*sample.shape[:2]),
                            sample, torch.zeros(*sample.shape[:2], 4))
                    else:
                        world_model.sequence_model = lambda sample, action, **kwargs: torch.zeros(*sample.shape[:2], 4)
                    agent = SimpleNamespace(sample=lambda latent: (
                        torch.zeros(*latent.shape[:2]), torch.zeros(*latent.shape[:2], 3)))
                    videos = []
                    logger = SimpleNamespace(log=lambda name, value, **kwargs: videos.append(value))
                    with torch.no_grad():
                        getattr(world_model, method)(
                            agent, torch.zeros(batch_size, 2, 1, 2, 3), torch.zeros(batch_size, 2),
                            batch_size, 2, True, logger, 0)
                    time_steps = 2 if method == "imagine_data" else 3
                    self.assertEqual(videos[0].shape, (time_steps, 1, 2, 3 * min(batch_size, 4)))


if __name__ == "__main__":
    unittest.main()
