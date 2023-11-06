# Controls how many milliseconds of antenna data we'll integrate when searching for satellite PRNs.
# Don't go above 20 as we might be subject to navigation message bit flips
ACQUISITION_INTEGRATION_PERIOD_MS = 20
# PT: Chosen experimentally
ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD = ACQUISITION_INTEGRATION_PERIOD_MS * 6

DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND = -6000
DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND = 6000
DOPPLER_SHIFT_SEARCH_INTERVAL = 500
