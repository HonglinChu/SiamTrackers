from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# # NanoTrackV1
# from nanotrack.models.head.ban_v1 import UPChannelBAN, DepthwiseBAN

# NanoTrackV2
from nanotrack.models.head.ban_v2 import UPChannelBAN, DepthwiseBAN

BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN
       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)

