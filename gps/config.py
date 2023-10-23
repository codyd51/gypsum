
# PT: This controls how many times we'll 'stack' the PRN signal on top of the received signal, while searching for the
# PRN in the spectrum.
# Increasing this value will cause the PRN to be identified within the signal more strongly/with higher accuracy,
# but at the cost of increased computation.
# I chose 32 because it makes correlations really clear and strong when plotting the data.
# Also called the integration period?
PRN_CORRELATION_CYCLE_COUNT = 32

# PT: Chosen through manually inspecting correlation graphs
PRN_CORRELATION_MAGNITUDE_THRESHOLD = 80

DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND = -6000
DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND = 6000
DOPPLER_SHIFT_SEARCH_INTERVAL = 500
