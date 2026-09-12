"""Retrieval context boundaries must include time-limit resets."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drama_episode_replay", ROOT / "Drama/replay_buffer.py")
replay_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_module)


class EpisodeBoundaryTests(unittest.TestCase):
    def test_retrieval_rejects_truncations_without_changing_bootstrap_mask(self):
        for store_on_gpu in (False, True):
            with self.subTest(store_on_gpu=store_on_gpu):
                config = SimpleNamespace(
                    BasicSettings=SimpleNamespace(ReplayBufferOnGPU=store_on_gpu, ImageSize=(2, 2), ImageChannel=1),
                    JointTrainAgent=SimpleNamespace(
                        BufferMaxLength=6, WorldModelWarmUp=0, BehaviourWarmUp=0,
                        Tau=1., ImaginationTau=1., Alpha=1., Beta=1., ImagineBatchSize=2, BatchSize=2))
                replay = replay_module.ReplayBuffer(config, device="cpu")
                obs = np.zeros((2, 2, 1), dtype=np.uint8)
                replay.append(obs, 0, 0., False)
                replay.append(obs, 0, 0., False, episode_end=True)
                replay.append(obs, 0, 0., False)
                self.assertTrue(replay.is_valid_context(1, 0, 2))
                self.assertFalse(replay.is_valid_context(2, 0, 2))
                self.assertFalse(bool(replay.termination_buffer[1]))
                self.assertTrue(bool(replay.episode_end_buffer[1]))
                self.assertEqual(replay.retrieval_view().episode_end_buffer.shape, (6, 1))
                *_, terminations = replay.sample(4, 2)
                self.assertFalse(bool(terminations.any()))

                # Default calls still derive context boundaries from termination.
                replay.append(obs, 0, 0., True)
                replay.append(obs, 0, 0., False)
                self.assertTrue(bool(replay.episode_end_buffer[3]))
                self.assertFalse(replay.is_valid_context(4, 0, 2))

                # An explicit false episode flag cannot hide a true terminal.
                replay.append(obs, 0, 0., True, episode_end=False)
                replay.append(obs, 0, 0., False)
                self.assertFalse(replay.is_valid_context(0, 0, 2))


if __name__ == "__main__":
    unittest.main()
