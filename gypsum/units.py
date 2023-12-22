import numpy as np

# TODO(PT): Rename this (or introduce a new type) to reflect that it represents 'seconds since startup'?
Seconds = float
WallClockSeconds = float
ReceiverDataSeconds = float
GpsSatelliteSeconds = float
GpsSatelliteSecondsIntoWeek = float

MetersPerSecond = float
RadiansPerSecond = float
Radians = float
Degrees = float
Percent = float

AntennaSamplesSpanningAcquisitionIntegrationPeriodMs = np.ndarray
AntennaSamplesSpanningOneMs = np.ndarray
PrnReplicaCodeSamplesSpanningOneMs = np.ndarray

CorrelationProfile = np.ndarray
CoherentCorrelationProfile = CorrelationProfile
NonCoherentCorrelationProfile = CorrelationProfile
# Represents the power of a correlation peak relative to the mean power of the correlation profile
CorrelationStrengthRatio = float
CoherentCorrelationPeak = complex

Hertz = float
DopplerShiftHz = float
CarrierWavePhaseInRadians = float
PrnCodePhaseInSamples = int

SampleCount = int
SampleRateHz = Hertz
