from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from trtr.core.config import cfg
from trtr.tracker.siamban_tracker import SiamBANTracker

TRACKS = {
          'SiamBANTracker': SiamBANTracker
         }

def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
