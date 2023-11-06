# Controls how many milliseconds of antenna data we'll integrate when searching for satellite PRNs.
# Don't go above 20 as we might be subject to navigation message bit flips
ACQUISITION_INTEGRATION_PERIOD_MS = 20
# PT: Chosen experimentally
# PT: It doesn't make much sense for this to be related to the acquisition integration period...
# Instead, it should be selected dynamically based on the signal and noise levels in the data.
ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD = ACQUISITION_INTEGRATION_PERIOD_MS * 6

DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND = -6000
DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND = 6000
DOPPLER_SHIFT_SEARCH_INTERVAL = 500
