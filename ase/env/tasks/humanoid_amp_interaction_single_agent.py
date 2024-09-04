import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from env.tasks.humanoid_amp import HumanoidAMP


class HumanoidAMPInteractionSingleAgent(HumanoidAMP):
    def __init__(self, cfg)


