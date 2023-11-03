# Length of each PRN, in chips
PRN_CHIP_COUNT = 1023

# How many times the GPS satellites repeat the PRN codes per second
PRN_REPETITIONS_PER_SECOND = 1000

# Assumes Nyquist sample rate
SAMPLES_PER_PRN_TRANSMISSION = 2 * PRN_CHIP_COUNT

# Assumes Nyquist sample rate
SAMPLES_PER_SECOND = SAMPLES_PER_PRN_TRANSMISSION * PRN_REPETITIONS_PER_SECOND

# Center frequency that GPS signals are emitted at.
# PT: The SDR must be set to this center frequency
GPS_L1_FREQUENCY = 1575.42e6

# One satellite gives a sphere around that satellite.
# Two satellites give the intersection of two spheres: a circle.
# Three satellites give two points along that circle.
# Four satellites give a single point.
MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX = 4
