from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nanotrack.core.config import cfg
from nanotrack.tracker.nano_tracker import NanoTracker

TRACKS = {
          'NanoTracker': NanoTracker 
         } 

def build_tracker(model): 
    return TRACKS[cfg.TRACK.TYPE](model) 
