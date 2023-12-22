from gypsum.units import Hertz
from gypsum.units import MetersPerSecond
from gypsum.units import PrnChipCount
from gypsum.units import Seconds

# Length of each PRN, in chips.
PRN_CHIP_COUNT: PrnChipCount = 1023

# How many times the GPS satellites repeat the PRN codes per second.
PRN_REPETITIONS_PER_SECOND: Hertz = 1000

# Center frequency that GPS signals are emitted at.
# PT: The SDR must be set to this center frequency.
GPS_L1_FREQUENCY: Hertz = 1575.42e6

# One satellite gives a sphere around that satellite.
# Two satellites give the intersection of two spheres: a circle.
# Three satellites give two points along that circle.
# Four satellites give a single point.
MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX = 4

# The navigation message is transmitted at 50 bits per second.
BITS_PER_SECOND = 50
# This means that there are 20 PRN correlations / 20 'pseudosymbols' per navigation message bit.
PSEUDOSYMBOLS_PER_NAVIGATION_BIT = 20
PSEUDOSYMBOLS_PER_SECOND = PSEUDOSYMBOLS_PER_NAVIGATION_BIT * BITS_PER_SECOND

# The Unix epoch is 1970/01/01.
# The GPS epoch is 1980/01/06.
# Therefore, the Unix timestamp of the GPS epoch is exactly 10 years and 7 days worth of seconds.
UNIX_TIMESTAMP_OF_GPS_EPOCH: Seconds = (60 * 60 * 24) * ((365 * 10) + 7)
SECONDS_PER_WEEK: Seconds = 60 * 60 * 24 * 7

# WSG84 speed of light in a vacuum, specified by GPS 30.3.4.3.
SPEED_OF_LIGHT: MetersPerSecond = 2.99792458e8
