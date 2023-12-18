import math
from enum import Enum, auto
from typing import Any, Collection, Iterator, TypeVar

import numpy as np

from gypsum.antenna_sample_provider import SampleProviderAttributes
from gypsum.units import (
    AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    AntennaSamplesSpanningOneMs,
    CorrelationProfile,
    DopplerShiftHz,
    PrnReplicaCodeSamplesSpanningOneMs,
)
from gypsum.units import CorrelationStrength
from gypsum.units import NonCoherentCorrelationProfile

_IterType = TypeVar("_IterType")


class IntegrationType(Enum):
    Coherent = auto()
    NonCoherent = auto()


def chunks(li: Collection[_IterType], chunk_size: int, step: int | None = None) -> Iterator[Collection[_IterType]]:
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
        yield li[i : i + chunk_size]  # type: ignore


def round_to_previous_multiple_of(val: int, multiple: int) -> int:
    return val - (val % multiple)


def get_indexes_of_sublist(li: list[Any], sub: list[Any]) -> list[int]:
    index_to_is_sublist_match = [li[pos : pos + len(sub)] == sub for pos in range(0, len(li) - len(sub) + 1)]
    indexes_of_sublist_matches = [match[0] for match in np.argwhere(np.array(index_to_is_sublist_match) == True)]
    return indexes_of_sublist_matches


def does_list_contain_sublist(li: list[Any], sub: list[Any]) -> bool:
    indexes_of_sublist = get_indexes_of_sublist(li, sub)
    return len(indexes_of_sublist) > 0


DEBUG = False


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
    # I notice that the samples returned by this function can directly be used as the input to a Costas tracking loop. I don't have a good intuition for why this works, as my understanding is that the samples returned by this function represent an abstract correlation magnitude.


def integrate_correlation_with_doppler_shifted_prn(
    integration_type: IntegrationType,
    antenna_data: AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    stream_attributes: SampleProviderAttributes,
    doppler_shift: DopplerShiftHz,
    prn_as_complex: PrnReplicaCodeSamplesSpanningOneMs,
) -> CorrelationProfile:
    correlation_data_type = {
        IntegrationType.Coherent: complex,
        IntegrationType.NonCoherent: np.float64,
    }[integration_type]
    samples_per_second = stream_attributes.samples_per_second
    samples_per_prn_transmission = stream_attributes.samples_per_prn_transmission
    integrated_correlation_result: np.ndarray = np.zeros(samples_per_prn_transmission, dtype=correlation_data_type)
    for i, chunk_that_may_contain_one_prn in enumerate(chunks(antenna_data, samples_per_prn_transmission)):
        sample_index = i * samples_per_prn_transmission
        integration_time_domain = (np.arange(samples_per_prn_transmission) / samples_per_second) + (
            sample_index / samples_per_second
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


def get_normalized_correlation_peak_strength(profile: NonCoherentCorrelationProfile) -> CorrelationStrength:
    correlation_peak_magnitude = np.max(profile)
    correlation_profile_excluding_peak = profile[profile != correlation_peak_magnitude]
    mean_magnitude_excluding_peak = np.mean(correlation_profile_excluding_peak)
    correlation_strength = correlation_peak_magnitude / mean_magnitude_excluding_peak
    return correlation_strength
