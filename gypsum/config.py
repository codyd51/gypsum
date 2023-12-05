# Controls how many milliseconds of antenna data we'll integrate when searching for satellite PRNs.
# Don't go above 20 as we might be subject to navigation message bit flips
ACQUISITION_INTEGRATION_PERIOD_MS = 20
# PT: Chosen experimentally
# PT: It doesn't make much sense for this to be related to the acquisition integration period...
# Instead, it should be selected dynamically based on the signal and noise levels in the data.
#ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD = ACQUISITION_INTEGRATION_PERIOD_MS * 3
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
