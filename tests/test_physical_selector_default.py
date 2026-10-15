import unittest

from iprune.config import ExperimentConfig
from iprune.experiments.protocol import PaperProtocolConfig


class PhysicalSelectorDefaultTests(unittest.TestCase):
    def test_shared_configs_default_to_physical_q_one(self):
        self.assertEqual(ExperimentConfig().qbeta, 0.0)
        self.assertEqual(PaperProtocolConfig().qbeta, 0.0)


if __name__ == "__main__":
    unittest.main()
