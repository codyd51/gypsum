from gypsum.units import Degrees, Hertz, Percent, Seconds

# Controls how many milliseconds of antenna data we'll integrate when searching for satellite PRNs.
# Don't go above 20 as we might be subject to navigation message bit flips
ACQUISITION_INTEGRATION_PERIOD_MS = 20
# PT: Chosen experimentally
# PT: It doesn't make much sense for this to be related to the acquisition integration period...
# Instead, it should be selected dynamically based on the signal and noise levels in the data.
# ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD = ACQUISITION_INTEGRATION_PERIOD_MS * 3
ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD = 70

DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND = -6000
DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND = 6000
DOPPLER_SHIFT_SEARCH_INTERVAL = 500

# The 'week number' encoded in subframe #1 is the number of weeks since the GPS epoch, mod 1024.
# This means that this field rolls over every 19.6 years, and receivers must have prior knowledge to know which
# base week to add to this value.
# At time of writing, there's 15 years until this rolls over.
# I think that should give me just enough time to deduce the base week automatically.
GPS_EPOCH_BASE_WEEK_NUMBER = 2048

# The GPS timeframe does not add leap seconds to its time frame, while UTC occasionally does.
# GPS receivers need to maintain an awareness of the current number of leap seconds that have been added to UTC, to
# synchronize the two time frames.
UTC_LEAP_SECONDS_COUNT = 27

# The tracker will look at signal heuristics from the last N milliseconds of tracker data to decide whether the
# tracker is currently locked onto the signal.
MILLISECONDS_TO_CONSIDER_FOR_TRACKER_LOCK_STATE = 250
# If the variance of the phase error in the consideration period is too high, we won't consider the signal locked.
MAXIMUM_PHASE_ERROR_VARIANCE_FOR_LOCK_STATE = 900
# The tracker has a periodic job that looks at the rotation of the constellation plot and performs a frequency
# correction if the observed rotation exceeds a threshold. This value controls how often this job runs.
CONSTELLATION_BASED_FREQUENCY_ADJUSTMENT_PERIOD: Seconds = 4
# This defines the maximum rotation that may be observed in the constellation plot, after which a
# frequency correction will be performed.
CONSTELLATION_BASED_FREQUENCY_ADJUSTMENT_MAXIMUM_ALLOWED_ROTATION: Degrees = 3
# When we decide to make a frequency adjustment based on an observed rotation in the constellation plot,
# this value controls the magnitude of the adjustment.
CONSTELLATION_BASED_FREQUENCY_ADJUSTMENT_MAGNITUDE: Hertz = 5

# The navigation bit integrator will recalculate the optimal pseudosymbol phase in response to a variety of conditions,
# one of which is the firing of a periodic timer.
RECALCULATE_PSEUDOSYMBOL_PHASE_PERIOD: Seconds = 1
# The navigation bit integrator will also recalculate the pseudosymbol phase if too many bits in the last memory window
# were unresolvable.
RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_MEMORY_SIZE = 10
# If a certain proportion of bits were unresolvable, trigger a pseudosymbol phase recalculation.
RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_THRESHOLD: Percent = 50
