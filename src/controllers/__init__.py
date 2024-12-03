REGISTRY = {}

from .basic_controller import BasicMAC
from .fop_basic_controller import BasicMAC as fopBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["fop_mac"] = fopBasicMAC