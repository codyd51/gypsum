
# PT: This controls how many times we'll 'stack' the PRN signal on top of the received signal, while searching for the
# PRN in the spectrum.
# Increasing this value will cause the PRN to be identified within the signal more strongly/with higher accuracy,
# but at the cost of increased computation.
# I chose 32 because it makes correlations really clear and strong when plotting the data.
PRN_CORRELATION_CYCLE_COUNT = 32
