from enum import Enum
from enum import auto
from typing import Any
from typing import TypeVar, Collection, Iterator

import math
import numpy as np

from gypsum.constants import SAMPLES_PER_PRN_TRANSMISSION
from gypsum.constants import SAMPLES_PER_SECOND

_IterType = TypeVar("_IterType")

AntennaSamplesSpanningAcquisitionIntegrationPeriodMs = np.ndarray
AntennaSamplesSpanningOneMs = np.ndarray
PrnReplicaCodeSamplesSpanningOneMs = np.ndarray

CorrelationProfile = np.ndarray
CorrelationStrength = float

DopplerShiftHz = float
CarrierWavePhaseInRadians = float
PrnCodePhaseInSamples = int


class IntegrationType(Enum):
    Coherent = auto()
    NonCoherent = auto()


def chunks(li: Collection[_IterType], chunk_size: int, step: int | None = None) -> Iterator[_IterType]:
    chunk_step = chunk_size
    if step:
        if step <= chunk_size:
            raise ValueError(f"Expected the custom step to be at least a chunk size")
        chunk_step = step
    for i in range(0, len(li), chunk_step):
        # Don't return a final truncated chunk
        if len(li) - i < chunk_size:
            # print(f'breaking because were on the last chunk, len={len(li)}, chunk_size={chunk_size}, i={i}')
            break
        yield li[i : i + chunk_size]


def round_to_previous_multiple_of(val: int, multiple: int) -> int:
    return val - (val % multiple)


def get_indexes_of_sublist(li: list[Any], sub: list[Any]) -> list[int]:
    index_to_is_sublist_match = [li[pos: pos + len(sub)] == sub for pos in range(0, len(li) - len(sub) + 1)]
    indexes_of_sublist_matches = [match[0] for match in np.argwhere(index_to_is_sublist_match == True)]
    return indexes_of_sublist_matches


def does_list_contain_sublist(li: list[Any], sub: list[Any]) -> bool:
    indexes_of_sublist = get_indexes_of_sublist(li, sub)
    return len(indexes_of_sublist) > 0


def frequency_domain_correlation(
    antenna_samples: AntennaSamplesSpanningOneMs, prn_replica: PrnReplicaCodeSamplesSpanningOneMs
) -> CorrelationProfile:
    # Perform correlation in the frequency domain.
    # This is much more efficient than attempting to perform correlation in the time domain, as we don't need to try
    # every possible phase shift of the PRN to identify the correlation peak.
    antenna_samples_fft = np.fft.fft(antenna_samples)
    prn_replica_fft = np.fft.fft(prn_replica)
    # Multiply by the complex conjugate of the PRN replica.
    # This aligns the phases of the antenna data and replica, and performs the cross-correlation.
    correlation_in_frequency_domain = antenna_samples_fft * np.conj(prn_replica_fft)
    # Convert the correlation result back to the time domain.
    # Each value gives the correlation of the antenna data with the PRN at different phase offsets.
    # Therefore, the offset of the peak will give the phase shift of the PRN that gives maximum correlation.
    return np.fft.ifft(correlation_in_frequency_domain)


def integrate_correlation_with_doppler_shifted_prn(
    integration_type: IntegrationType,
    antenna_data: AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    doppler_shift: DopplerShiftHz,
    prn_as_complex: PrnReplicaCodeSamplesSpanningOneMs,
) -> CorrelationProfile:
    correlation_data_type = {
        IntegrationType.Coherent: complex,
        IntegrationType.NonCoherent: np.float64,
    }[integration_type]
    integrated_correlation_result = np.zeros(SAMPLES_PER_PRN_TRANSMISSION, dtype=correlation_data_type)
    for i, chunk_that_may_contain_one_prn in enumerate(chunks(antenna_data, SAMPLES_PER_PRN_TRANSMISSION)):
        sample_index = i * SAMPLES_PER_PRN_TRANSMISSION
        integration_time_domain = (np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND) + (
            sample_index / SAMPLES_PER_SECOND
        )
        doppler_shift_carrier = np.exp(-1j * math.tau * doppler_shift * integration_time_domain)
        doppler_shifted_antenna_data_chunk = chunk_that_may_contain_one_prn * doppler_shift_carrier

        correlation_result = frequency_domain_correlation(doppler_shifted_antenna_data_chunk, prn_as_complex)

        if integration_type == IntegrationType.Coherent:
            integrated_correlation_result += correlation_result
        elif integration_type == IntegrationType.NonCoherent:
            integrated_correlation_result += np.abs(correlation_result)
        else:
            raise ValueError("Unexpected integration type")

    return integrated_correlation_result
