import os
import os.path as osp
from improve.config import resolver

ROOT = osp.dirname(osp.dirname(__file__))
CONFIG = osp.join(ROOT, "config")
WEIGHTS = osp.join(ROOT, "weights")
RESULTS = osp.join(ROOT, "results")
