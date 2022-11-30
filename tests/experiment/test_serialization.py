import contextlib
import os
import shutil
import unittest
from copy import copy
from pathlib import Path

import torch

CWD = os.getcwd()
NULL = open(os.devnull, "w")
TESTS_ROOT_DIR = Path(__file__).parent.parent.absolute()
if not hasattr(torch._C, "_cuda_getDeviceCount"):
    # hardware
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    # software
    CUDA_AVAILABLE = False
    # Error message is different for CPU only PyTorch
    CUDA_ERROR = AssertionError
    CUDA_NOT_AVAILABLE_MESSAGE = "Torch not compiled with CUDA enabled"
else:
    GPU_COUNT = torch._C._cuda_getDeviceCount()
    if GPU_COUNT == 0:
        GPU_AVAILABLE = False
    else:
        GPU_AVAILABLE = True
    # software
    CUDA_AVAILABLE = True
    # if PyTorch is compiled with CUDA and CUDA driver is available
    if shutil.which("nvidia-smi"):
        CUDA_ERROR = RuntimeError
        CUDA_NOT_AVAILABLE_MESSAGE = "No CUDA GPUs are available"
    else:
        CUDA_ERROR = AssertionError
        CUDA_NOT_AVAILABLE_MESSAGE = "Found no NVIDIA driver on your system"

MAP_LOCATION_MESSAGE = "Attempting to deserialize object"
INVALID_DEVICE_MESSAGE = "CUDA error: invalid device ordinal"

# find the first non-scalar tensor and check its device
def check_state_dict(d: dict, target_device):
    for _, v in d.items():
        if isinstance(v, torch.Tensor):
            # ignore scalar values
            # they are on CPU by default
            if v.dim() == 0:
                continue
            assert v.device == target_device
            return
        elif isinstance(v, dict):
            return check_state_dict(v, target_device)
        else:
            raise ValueError("Not sure if this ever happens")


def check_tensor_devices(checkpoint: dict, device: torch.device):
    training_modules = checkpoint["training_modules"]
    controller = checkpoint["controller"]

    # model should be on `device`
    assert next(training_modules.model.parameters()).device == device
    # evaluator.model should be on `device`
    assert next(training_modules.evaluator.model.parameters()).device == device
    # loss should be on `device`
    check_state_dict(training_modules.loss.state_dict(), device)
    # evaluator.loss should always be on "cpu"
    check_state_dict(training_modules.evaluator.loss.state_dict(), torch.device("cpu"))
    # training_modules.evaluator.model_device should be set properly
    # otherwise some tensor will go to the original device assigned before restarting
    assert training_modules.evaluator.model_device == device
    # optimizer should be on `device`
    check_state_dict(controller.optimizer.state_dict(), device)


def load_checkpoint_with_redirect(kwargs):
    from hippynn.experiment.serialization import load_checkpoint_from_cwd

    with contextlib.redirect_stdout(NULL), contextlib.redirect_stderr(NULL):
        checkpoint = load_checkpoint_from_cwd(**kwargs)
    # checkpoint = load_checkpoint_from_cwd(**kwargs)
    return checkpoint


# TODO: add load model tests without duplicating too much code


class TestRestarting(unittest.TestCase):
    # options that should be tested in all cases
    common_test_options = ["simple", "map_to_cpu", "both", "auto", "cpu", "gpu_0"]

    def setUp(self):
        self.possible_options = {
            "simple": {"model_device": None, "map_location": None},
            "map_to_cpu": {"model_device": None, "map_location": "cpu"},
            "wrong_map_to_gpu": {"model_device": None, "map_location": "cuda:0"},
            "right_map_to_gpu": {"model_device": None, "map_location": None},
            "both": {"model_device": "auto", "map_location": "cpu"},
            "auto": {"model_device": "auto", "map_location": None},
            "cpu": {"model_device": "cpu", "map_location": None},
            "gpu_0": {"model_device": 0, "map_location": None},
            "gpu_1": {"model_device": 1, "map_location": None},
        }
        # None values need to be determined at runtime
        self.expected_results = {
            "simple": None,
            "map_to_cpu": torch.device("cpu"),
            "wrong_map_to_gpu": (TypeError, "RNG state must be a torch.ByteTensor"),
            "right_map_to_gpu": None,
            "both": (TypeError, "Passing map_location explicitly and the model device are incompatible"),
            "auto": None,
            "cpu": torch.device("cpu"),
            "gpu_0": None,
            "gpu_1": None,
        }

    def test_gpu_0_checkpoint(self):
        test_options = copy(self.common_test_options)
        if GPU_AVAILABLE:
            test_options.append("gpu_1")
            test_options.append("wrong_map_to_gpu")
            self.expected_results.update(
                {
                    "simple": torch.device(0),
                    "auto": torch.device(0),
                    "gpu_0": torch.device(0),
                }
            )
            if GPU_COUNT > 1:
                self.expected_results["gpu_1"] = torch.device(1)
                self.possible_options["right_map_to_gpu"]["map_location"] = {"cuda:0": "cuda:1"}
                self.expected_results["right_map_to_gpu"] = torch.device(1)
                test_options.append("right_map_to_gpu")
            else:
                self.expected_results["gpu_1"] = (RuntimeError, INVALID_DEVICE_MESSAGE)
        else:
            self.expected_results.update(
                {
                    "simple": (RuntimeError, MAP_LOCATION_MESSAGE),
                    "auto": torch.device("cpu"),
                    "gpu_0": (CUDA_ERROR, CUDA_NOT_AVAILABLE_MESSAGE),
                }
            )
        self._run_tests(test_options, "gpu0")

    def test_gpu_1_checkpoint(self):
        test_options = copy(self.common_test_options)
        if GPU_AVAILABLE:
            test_options.append("wrong_map_to_gpu")
            test_options.append("right_map_to_gpu")
            if GPU_COUNT > 1:
                self.expected_results["simple"] = torch.device(1)
            else:
                self.expected_results["simple"] = (RuntimeError, MAP_LOCATION_MESSAGE)
            self.possible_options["right_map_to_gpu"]["map_location"] = {"cuda:1": "cuda:0"}
            self.expected_results["right_map_to_gpu"] = torch.device(0)
            self.expected_results["auto"] = torch.device(0)
            self.expected_results["gpu_0"] = torch.device(0)
        else:
            self.expected_results.update(
                {
                    "simple": (RuntimeError, MAP_LOCATION_MESSAGE),
                    "auto": torch.device("cpu"),
                    "gpu_0": (CUDA_ERROR, CUDA_NOT_AVAILABLE_MESSAGE),
                }
            )
        self._run_tests(test_options, "gpu1")

    def test_cpu_checkpoint(self):
        test_options = copy(self.common_test_options)
        self.expected_results["simple"] = torch.device("cpu")
        if GPU_AVAILABLE:
            test_options.append("wrong_map_to_gpu")
            self.expected_results["auto"] = torch.device(0)
            self.expected_results["gpu_0"] = torch.device(0)
        else:
            self.expected_results.update(
                {
                    "auto": torch.device("cpu"),
                    "gpu_0": (CUDA_ERROR, CUDA_NOT_AVAILABLE_MESSAGE),
                }
            )
        self._run_tests(test_options, "cpu")

    def _run_tests(self, test_options, folder):
        os.chdir(f"{TESTS_ROOT_DIR}/assets/models/{folder}")
        for option in test_options:
            kwargs = self.possible_options[option]
            expected_result = self.expected_results[option]
            if type(expected_result) == torch.device:
                checkpoint = load_checkpoint_with_redirect(kwargs)
                check_tensor_devices(checkpoint, expected_result)
                del checkpoint
            else:
                error, message = expected_result
                with self.assertRaises(error) as cm:
                    checkpoint = load_checkpoint_with_redirect(kwargs)
                self.assertIn(message, str(cm.exception))
