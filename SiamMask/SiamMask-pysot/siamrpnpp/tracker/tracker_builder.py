# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamrpnpp.core.config import cfg
from siamrpnpp.tracker.siamrpn_tracker import SiamRPNTracker
from siamrpnpp.tracker.siammask_tracker import SiamMaskTracker
from siamrpnpp.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }

def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
