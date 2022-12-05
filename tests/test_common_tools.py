import os
import unittest
from unittest.mock import patch

import numpy as np
import torch


class TestConfigSources(unittest.TestCase):
    def setUp(self):
        self.global_rc_file = os.path.expanduser("~/.hippynnrc")
        self.rc_file = ".local_hippynn_rc"
        # test environmental variables
        os.environ["HIPPYNN_LOCAL_RC_FILE"] = self.rc_file
        # test if local rc is read
        with open(self.rc_file, "w") as rc:
            print("[GLOBALS]\nDEBUG_GRAPH_EXECUTION = True", file=rc)
        # test if 1 is correctly parsed as True
        with open(self.global_rc_file, "w") as rc:
            print("[GLOBALS]\nTRANSPARENT_PLOT = 1", file=rc)
        from hippynn import _settings_setup

        self.config_sources = _settings_setup.config_sources
        self.settings = _settings_setup.settings

    def test_global_rc(self):

        self.assertIsNotNone(self.config_sources.get("~/.hippynnrc"))
        self.assertTrue(self.settings.TRANSPARENT_PLOT)

    def test_local_rc(self):
        # passing this means hippynn can correct parse the environmental variables as well
        self.assertIsNotNone(self.config_sources.get("LOCAL_RC_FILE"))
        self.assertTrue(self.settings.DEBUG_GRAPH_EXECUTION)

    def tearDown(self):
        os.remove(self.global_rc_file)
        os.remove(self.rc_file)
        os.environ.pop("HIPPYNN_LOCAL_RC_FILE")


class TestTools(unittest.TestCase):
    def test_torch_dtype(self):
        from hippynn.tools import np_of_torchdefaultdtype

        torch.set_default_dtype(torch.float16)
        self.assertEqual(np_of_torchdefaultdtype(), np.float16)
        torch.set_default_dtype(torch.float64)
        self.assertEqual(np_of_torchdefaultdtype(), np.float64)

    def test_array_padding(self):
        from hippynn.tools import pad_np_array_to_length_with_zeros

        # get a random shape
        i, j = np.random.randint(5, high=15, size=2)
        # generate an empty array
        test_array = np.empty((i, j))
        # if ask to pad to a negative value
        with self.assertRaises(ValueError) as cm:
            pad_np_array_to_length_with_zeros(test_array, i - 1)
        self.assertEqual(
            f"Cannot pad array to negative length! Array length: {i}, Total length requested: {i-1}", str(cm.exception)
        )
        # test padding the second dimension to +2
        padded_array = pad_np_array_to_length_with_zeros(test_array, j + 2, axis=1)
        self.assertEqual(padded_array.shape, (i, j + 2))

    @patch("builtins.print")
    def test_print_lr(self, mock_print):
        from hippynn.tools import print_lr

        test_layer = torch.nn.Linear(10, 10, bias=False)
        optimizers = [
            "Adadelta",
            "Adagrad",
            "Adam",
            "AdamW",
            "SparseAdam",
            "Adamax",
            "ASGD",
            "NAdam",
            "RAdam",
            "RMSprop",
            "Rprop",
            "SGD",
        ]
        lr = np.random.random()
        # test for all available optimizers except BFGS which we probably will never use
        for i in optimizers:
            opt_algo = getattr(torch.optim, i)
            opt = opt_algo(test_layer.parameters(), lr=lr)
            print_lr(opt)
            mock_print.assert_called_with(f"Learning rate:{lr:>10.5g}")


if __name__ == "__main__":
    unittest.main()
