from collections import defaultdict
from enum import Enum, auto
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

from scipy.signal import resample_poly

from gps.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId
from gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source
from gps.constants import SAMPLES_PER_SECOND, SAMPLES_PER_PRN_TRANSMISSION
from gps.satellite import GpsSatellite, ALL_SATELLITE_IDS
from gps.utils import chunks


class Samples:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.i = 0

    def peek_samples(self, n: int) -> np.ndarray:
        return self.data[self.i:self.i + n]

    def get_samples(self, n: int) -> np.ndarray:
        out = self.peek_samples(n)
        self.i += n
        return out


def get_satellites_info_and_antenna_samples() -> Tuple[dict[GpsSatelliteId, GpsSatellite], Samples]:
    satellites_to_replica_prn_signals = generate_replica_prn_signals()
    satellites_by_id = {
        satellite_id: GpsSatellite(
            satellite_id=satellite_id,
            prn_code=code
        )
        for satellite_id, code in satellites_to_replica_prn_signals.items()
    }
    input_source = INPUT_SOURCES[6]
    print(input_source.path.as_posix())
    sample_rate = input_source.sdr_sample_rate
    antenna_data = Samples(get_samples_from_radio_input_source(input_source, sample_rate))
    return satellites_by_id, antenna_data


def frequency_domain_correlation(segment, prn_replica):
    R = np.fft.fft(segment)
    L = np.fft.fft(prn_replica)
    product = R * np.conj(L)
    return np.fft.ifft(product)


_CorrelationProfile = np.ndarray
_CorrelationStrength = float

_DopplerShiftHz = float
_CarrierWavePhaseInRadians = float
_PrnCodePhaseInSamples = int


class ResampledPrnProvider:
    def __init__(self, satellites: dict[GpsSatelliteId, GpsSatellite]):
        self.satellites = satellites
        self.sv_id_to_resampled_prn_cache = defaultdict(dict)

    def get_resampled_prn(self, sv_id: GpsSatelliteId, sample_count: int) -> np.ndarray:
        if sample_count not in self.sv_id_to_resampled_prn_cache[sv_id]:
            prn = self.satellites[sv_id].prn_as_complex
            resampled_prn = resample_poly(prn, sample_count, len(prn))
            resampled_prn = np.array([complex(1, 0) if x.real >= 0.5 else complex(-1, 0) for x in resampled_prn][:sample_count])
            self.sv_id_to_resampled_prn_cache[sv_id][sample_count] = resampled_prn
        return self.sv_id_to_resampled_prn_cache[sv_id][sample_count]


def contains(subseq, inseq):
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))


@dataclass
class BestNonCoherentCorrelationProfile:
    doppler_shift: _DopplerShiftHz
    non_coherent_correlation_profile: _CorrelationProfile
    # Just convenience accessors that can be derived from the correlation profile
    sample_offset_of_correlation_peak: int
    correlation_strength: float


@dataclass
class DetectedSatelliteInfo:
    satellite_id: GpsSatelliteId
    doppler_shift: _DopplerShiftHz
    carrier_wave_phase_shift: _CarrierWavePhaseInRadians
    prn_phase_shift: _PrnCodePhaseInSamples


class IntegrationType(Enum):
    Coherent = auto()
    NonCoherent = auto()


def integrate_correlation_with_doppler_shifted_prn(
    integration_type: IntegrationType,
    antenna_data: Samples,
    sv_id: GpsSatelliteId,
    doppler_shift: _DopplerShiftHz,
    prn_provider: ResampledPrnProvider,
) -> _CorrelationProfile:
    # Calculate the PRN length, accounting for this Doppler shift
    resampled_prn = prn_provider.satellites[sv_id].prn_as_complex

    integration_period_ms = 20
    samples_in_window = integration_period_ms * SAMPLES_PER_PRN_TRANSMISSION
    antenna_data_snippet = antenna_data.data[:samples_in_window]
    integration_time_domain = np.arange(integration_period_ms * SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND
    doppler_shift_carrier = np.exp(-1j * 2 * np.pi * doppler_shift * integration_time_domain)
    doppler_shifted_antenna_data_snippet = antenna_data_snippet * doppler_shift_carrier

    correlation_data_type = {
        IntegrationType.Coherent: complex,
        IntegrationType.NonCoherent: np.float64,
    }[integration_type]
    coherent_integration_result = np.zeros(SAMPLES_PER_PRN_TRANSMISSION, dtype=correlation_data_type)
    for i, chunk_that_may_contain_one_prn in enumerate(chunks(doppler_shifted_antenna_data_snippet, SAMPLES_PER_PRN_TRANSMISSION)):
        correlation_result = frequency_domain_correlation(chunk_that_may_contain_one_prn, resampled_prn)

        if integration_type == IntegrationType.Coherent:
            coherent_integration_result += correlation_result
        elif integration_type == IntegrationType.NonCoherent:
            coherent_integration_result += np.abs(correlation_result)
        else:
            raise ValueError('Unexpected integration type')

    return coherent_integration_result


def compute_best_doppler_shift_estimation(
    center: float,
    spread: float,
    antenna_data: Samples,
    sv_id: GpsSatelliteId,
    prn_provider: ResampledPrnProvider,
) -> BestNonCoherentCorrelationProfile:
    doppler_shift_to_correlation_profile = {}
    for doppler_shift in range(
        int(center - spread),
        int(center + spread),
        int(spread / 10),
    ):
        correlation_profile = integrate_correlation_with_doppler_shifted_prn(
            # Always use non-coherent integration when searching for the best Doppler peaks.
            # This will give us the strongest SNR possible to detect peaks.
            IntegrationType.NonCoherent,
            antenna_data,
            sv_id,
            doppler_shift,
            prn_provider,
        )
        doppler_shift_to_correlation_profile[doppler_shift] = correlation_profile

    # Find the best correlation result
    best_doppler_shift = max(doppler_shift_to_correlation_profile, key=lambda key: np.max(doppler_shift_to_correlation_profile[key]))
    best_correlation_profile = doppler_shift_to_correlation_profile[best_doppler_shift]
    sample_offset_of_correlation_peak = np.argmax(best_correlation_profile)
    correlation_strength = best_correlation_profile[sample_offset_of_correlation_peak]
    return BestNonCoherentCorrelationProfile(
        doppler_shift=best_doppler_shift,
        non_coherent_correlation_profile=best_correlation_profile,
        sample_offset_of_correlation_peak=int(sample_offset_of_correlation_peak),
        correlation_strength=correlation_strength,
    )


def test_acquire(
    satellites_by_id: dict[GpsSatelliteId, GpsSatellite],
    antenna_data: Samples,
    prn_provider: ResampledPrnProvider,
) -> dict[GpsSatellite, DetectedSatelliteInfo]:
    detected_satellites_by_id = {}
    for sv_id in ALL_SATELLITE_IDS:
        # Detection
        doppler_frequency_estimation_spread = 5000
        # This must be 10 as the search factor divides the spread by 10
        best_non_coherent_correlation_profile_across_all_search_space = None
        doppler_estimation = 0
        while doppler_frequency_estimation_spread >= 10:
            best_non_coherent_correlation_profile_in_this_search_space = compute_best_doppler_shift_estimation(
                doppler_estimation,
                doppler_frequency_estimation_spread,
                antenna_data,
                sv_id,
                prn_provider,
            )
            doppler_frequency_estimation_spread /= 2
            if (
                # Base case
                not best_non_coherent_correlation_profile_across_all_search_space
                # Found a better candidate
                or best_non_coherent_correlation_profile_in_this_search_space.correlation_strength > best_non_coherent_correlation_profile_across_all_search_space.correlation_strength
            ):
                best_non_coherent_correlation_profile_across_all_search_space = best_non_coherent_correlation_profile_in_this_search_space
                doppler_estimation = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
                print(
                    f'Found a better candidate Doppler for SV({sv_id}): '
                    f'(Found in [{doppler_estimation - doppler_frequency_estimation_spread:.2f} | '
                    f'{doppler_estimation:.2f} | {doppler_estimation + doppler_frequency_estimation_spread:.2f}], '
                    f'Strength: '
                    f'{best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}'
                )

        print(f'Best correlation for SV({sv_id}) at Doppler {doppler_estimation:.2f} corr {best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}')
        if best_non_coherent_correlation_profile_across_all_search_space.correlation_strength > 37.0:
            # Detection, Doppler frequency over threshold
            print('Non-coherent correlation strength was above threshold, continuing to coherent integration...')
            # Now, compute the coherent correlation so that we can determine (an estimate) of the phase of the carrier wave
            best_doppler_shift = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
            coherent_correlation_profile = integrate_correlation_with_doppler_shifted_prn(
                IntegrationType.Coherent,
                antenna_data,
                sv_id,
                best_doppler_shift,
                prn_provider,
            )

            # Rely on the correlation peak index that comes from non-coherent integration, since it'll be stronger and
            # therefore has less chance of being overridden by noise. Coherent integration may have selected a noise
            # peak.
            sample_offset_of_correlation_peak = best_non_coherent_correlation_profile_across_all_search_space.sample_offset_of_correlation_peak
            carrier_wave_phase_shift = np.angle(coherent_correlation_profile[sample_offset_of_correlation_peak])
            # The sample offset where the best correlation occurs gives us (an estimate) of the phase shift of the PRN
            prn_phase_shift = sample_offset_of_correlation_peak
            print(f'Detected SV({sv_id}):')
            print(f'\tDoppler {best_doppler_shift:.2f}')
            print(f'\tCarrier phase {carrier_wave_phase_shift}')
            print(f'\tPRN phase {prn_phase_shift:.2f}')
            detected_satellites_by_id[satellites_by_id[sv_id]] = DetectedSatelliteInfo(
                satellite_id=sv_id,
                doppler_shift=best_doppler_shift,
                carrier_wave_phase_shift=carrier_wave_phase_shift,
                prn_phase_shift=prn_phase_shift,
            )

    return detected_satellites_by_id


def main_new():
    satellites_by_id, antenna_data = get_satellites_info_and_antenna_samples()
    prn_provider = ResampledPrnProvider(satellites_by_id)

    if False:
        detected_satellites = test_acquire(
            satellites_by_id,
            antenna_data,
            prn_provider
        )

        sat_25 = satellites_by_id[GpsSatelliteId(id=25)]
        detected_satellite = detected_satellites[sat_25]
    else:
        detected_satellite = DetectedSatelliteInfo(
            satellite_id=GpsSatelliteId(id=32),
            #doppler_shift=2323.00,
            doppler_shift=2333.23,
            carrier_wave_phase_shift=-0.7562601181399523,
            #prn_phase_shift=1064,
            prn_phase_shift=982,
        )

    # Tracking loop
    sv_id = detected_satellite.satellite_id
    doppler_shift = detected_satellite.doppler_shift
    carrier_wave_phase = detected_satellite.carrier_wave_phase_shift
    prn_code_phase = detected_satellite.prn_phase_shift
    sample_index = 0

    # Also called 'alpha'
    #loop_gain_phase = 0.00005
    # Also called 'beta'
    #loop_gain_freq = 0.0001
    loop_gain_phase = 0.0005
    # Also called 'beta'
    loop_gain_freq = 0.00003

    correlations = []
    navigation_bit_pseudosymbols = []
    carrier_wave_phase_errors = []
    carrier_wave_phases = []
    doppler_shifts = []
    while True:
        antenna_data_snippet = antenna_data.data[sample_index:sample_index + SAMPLES_PER_PRN_TRANSMISSION]
        if len(antenna_data_snippet) < SAMPLES_PER_PRN_TRANSMISSION:
            print(f'Ran out of antenna data at sample index {sample_index}')
            break
        sample_index += SAMPLES_PER_PRN_TRANSMISSION

        # Generate Doppler-shifted and phase-shifted carrier wave
        time_domain = (np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND) + (sample_index / SAMPLES_PER_SECOND)
        doppler_shift_carrier = np.exp(-1j * ((2 * np.pi * doppler_shift * time_domain) + carrier_wave_phase))
        doppler_shifted_antenna_data_snippet = antenna_data_snippet * doppler_shift_carrier

        # Correlate early, prompt, and late phase versions of the PRN
        unslid_prn = prn_provider.satellites[sv_id].prn_as_complex
        prompt_prn = np.roll(unslid_prn, -prn_code_phase)

        coherent_prompt_correlation = frequency_domain_correlation(doppler_shifted_antenna_data_snippet, prompt_prn)
        non_coherent_prompt_correlation = np.abs(coherent_prompt_correlation)
        non_coherent_prompt_peak_offset = np.argmax(non_coherent_prompt_correlation)
        non_coherent_prompt_peak = non_coherent_prompt_correlation[non_coherent_prompt_peak_offset]

        # Try to detect and ignore low-quality samples
        if non_coherent_prompt_peak < 7:
            print(f'Skipping bad sample with a low peak of {non_coherent_prompt_peak}')
            # Need to add an empty correlation array/empty bit to keep all our numbers on track
            correlations.append(np.zeros(SAMPLES_PER_PRN_TRANSMISSION))
            navigation_bit_pseudosymbols.append(0)
            continue

        if False:
            # 2 samples per chip, so 1 sample offset is half a chip
            early_prn = np.roll(unslid_prn, -prn_code_phase - 1)
            late_prn = np.roll(unslid_prn, -prn_code_phase + 1)

            coherent_early_correlation = frequency_domain_correlation(doppler_shifted_antenna_data_snippet, early_prn)
            coherent_late_correlation = frequency_domain_correlation(doppler_shifted_antenna_data_snippet, late_prn)
            non_coherent_early_correlation = np.abs(coherent_early_correlation)
            non_coherent_late_correlation = np.abs(coherent_late_correlation)
            # We'll always use the non-coherent peak offset to index into the coherent result
            coherent_prompt_prn_correlation_peak = coherent_prompt_correlation[non_coherent_prompt_peak_offset]
            did_shift = False
            # First, try to adjust the PRN code phase via correlation strength
            if non_coherent_early_peak_offset > max(non_coherent_prompt_peak_offset, non_coherent_late_peak_offset):
                print(f'select early peak')
                prn_code_phase += 1
                did_shift = True

            elif non_coherent_late_peak_offset > max(non_coherent_prompt_peak_offset, non_coherent_early_peak_offset):
                print(f'select late peak')
                prn_code_phase -= 1
                did_shift = True

            if not did_shift:
                # If we didn't observe any difference in correlation strength, try to adjust via
                # the correlation peak offset
                if non_coherent_early_peak_offset < min(non_coherent_prompt_peak_offset, non_coherent_late_peak_offset):
                    print(f'shifting due to early peak (E {non_coherent_early_peak_offset}) (P {non_coherent_prompt_peak_offset}) (L {non_coherent_late_peak_offset})')
                    prn_code_phase -= 1
                    #distance = non_coherent_early_peak_offset
                    #prn_code_phase -= int(distance * prn_code_phase_loop_gain)
                elif non_coherent_late_peak_offset < min(non_coherent_prompt_peak_offset, non_coherent_early_peak_offset):
                    print(f'shifting due to late peak (E {non_coherent_early_peak_offset}) (P {non_coherent_prompt_peak_offset}) (L {non_coherent_late_peak_offset})')
                    prn_code_phase += 1
                    #distance = non_coherent_late_peak_offset
                    #prn_code_phase -= (distance * prn_code_phase_loop_gain)

        if True:
            if non_coherent_prompt_peak_offset <= SAMPLES_PER_PRN_TRANSMISSION / 2:
                centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset
            else:
                centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset - SAMPLES_PER_PRN_TRANSMISSION
            print(f'Peak offset {non_coherent_prompt_peak_offset}, centered offset {centered_non_coherent_prompt_peak_offset}')
            if centered_non_coherent_prompt_peak_offset > 0:
                prn_code_phase -= centered_non_coherent_prompt_peak_offset
            else:
                prn_code_phase += centered_non_coherent_prompt_peak_offset

        if False:
            plt.plot(non_coherent_early_correlation, label="early")
            plt.plot(non_coherent_prompt_correlation, label="prompt")
            plt.plot(non_coherent_late_correlation, label="late")
            plt.legend()
            plt.show()

        # Finally, ensure we're always sliding within one PRN transmission
        prn_code_phase = int(prn_code_phase)
        prn_code_phase %= SAMPLES_PER_PRN_TRANSMISSION

        # Ensure carrier wave alignment
        new_prompt_prn = np.roll(unslid_prn, -prn_code_phase)
        new_coherent_prompt_correlation = frequency_domain_correlation(doppler_shifted_antenna_data_snippet, new_prompt_prn)
        new_non_coherent_prompt_correlation = np.abs(new_coherent_prompt_correlation)
        new_non_coherent_prompt_peak_offset = np.argmax(new_non_coherent_prompt_correlation)
        new_coherent_prompt_prn_correlation_peak = new_coherent_prompt_correlation[new_non_coherent_prompt_peak_offset]

        #carrier_wave_phase_error = np.angle(new_coherent_prompt_prn_correlation_peak)
        I = np.real(new_coherent_prompt_prn_correlation_peak)
        Q = np.imag(new_coherent_prompt_prn_correlation_peak)
        carrier_wave_phase_error = I * Q
        doppler_shift += loop_gain_freq * carrier_wave_phase_error
        carrier_wave_phase += loop_gain_phase * carrier_wave_phase_error
        carrier_wave_phase %= (2. * np.pi)

        navigation_bit_pseudosymbol_value = int(np.sign(new_coherent_prompt_prn_correlation_peak))
        navigation_bit_pseudosymbols.append(navigation_bit_pseudosymbol_value)

        correlations.append(coherent_prompt_correlation)

        print(f'Doppler shift {doppler_shift:.2f}')
        print(f'Carrier phase {carrier_wave_phase:.8f}')
        print(f'Code phase {prn_code_phase}')

        doppler_shifts.append(doppler_shift)
        carrier_wave_phase_errors.append(carrier_wave_phase_error)
        carrier_wave_phases.append(carrier_wave_phase)

        if len(correlations) > 120000:
            break

    if False:
        all = np.concatenate(correlations)
        plt.figure(figsize=(17,4))
        plt.plot(all)
        plt.title(f'α={loop_gain_phase} β={loop_gain_freq}')
        plt.tight_layout()
        plt.show()

    plt.plot(doppler_shifts)
    plt.title(f'Doppler shift')
    plt.show()
    plt.plot(carrier_wave_phases)
    plt.title(f'Carrier wave phase')
    plt.show()
    plt.plot(carrier_wave_phase_errors)
    plt.title(f'Carrier wave phase error')
    plt.show()

    if False:
        plt.plot(navigation_bit_pseudosymbols)
        plt.show()

    confidence_scores = []
    for roll in range(0, 20):
        print(f'Try roll {roll}')
        phase_shifted_bits = navigation_bit_pseudosymbols[roll:]

        confidences = []
        for twenty_pseudosymbols in chunks(phase_shifted_bits, 20):
            #print(twenty_pseudosymbols)
            integrated_value = sum(twenty_pseudosymbols)
            confidences.append(abs(integrated_value))
        # Compute an overall confidence score for this offset
        confidence_scores.append(np.mean(confidences))

    print(f'Confidence scores: {confidence_scores}')
    best_offset = np.argmax(confidence_scores)
    print(f"Best Offset: {best_offset} ({confidence_scores[best_offset]})")

    bit_phase = best_offset
    phase_shifted_bits = navigation_bit_pseudosymbols[bit_phase:]
    bits = []
    for twenty_pseudosymbols in chunks(phase_shifted_bits, 20):
        integrated_value = sum(twenty_pseudosymbols)
        bit_value = np.sign(integrated_value)
        bits.append(bit_value)

    digital_bits = [1 if b == 1.0 else 0 for b in bits]
    inverted_bits = [0 if b == 1.0 else 1 for b in bits]
    print(f'Bit count: {len(digital_bits)}')
    print(f'Bits:          {digital_bits}')
    print(f'Inverted bits: {inverted_bits}')

    preamble = [1, 0, 0, 0, 1, 0, 1, 1]
    print(f'Preamble {preamble} found in bits? {contains(preamble, digital_bits)}')
    print(f'Preamble {preamble} found in inverted bits? {contains(preamble, inverted_bits)}')

    def get_matches(l, sub):
        return [l[pos:pos + len(sub)] == sub for pos in range(0, len(l) - len(sub) + 1)]

    preamble_starts_in_digital_bits = (
        [x[0] for x in (np.argwhere(np.array(get_matches(digital_bits, preamble)) == True))])
    print(f'Preamble starts in bits:          {preamble_starts_in_digital_bits}')
    from itertools import pairwise
    for (i, j) in pairwise(preamble_starts_in_digital_bits):
        diff = j - i
        print(f'\tDiff from {j} to {i}: {diff}')
    #plt.plot([1 if x in preamble_starts_in_digital_bits else 0 for x in range(len(digital_bits))],
    #         label="Preambles in upright bits")

    preamble_starts_in_inverted_bits = (
        [x[0] for x in (np.argwhere(np.array(get_matches(inverted_bits, preamble)) == True))])
    print(f'Preamble starts in inverted bits: {preamble_starts_in_inverted_bits}')
    for (i, j) in pairwise(preamble_starts_in_inverted_bits):
        diff = j - i
        print(f'\tDiff from {j} to {i}: {diff}')


@dataclass
class TelemetryWord:
    # TODO(PT): Introduce a dedicated type? Ints are nicer than bools because we'll probably do bit manipulation?
    telemetry_message: list[int]
    integrity_status_flag: int
    spare_bit: int
    parity_bits: list[int]


@dataclass
class HandoverWord:
    time_of_week: list[int]
    alert_flag: int
    anti_spoof_flag: int
    subframe_id_codes: list[int]
    to_be_solved: list[int]
    parity_bits: list[int]


class BitParser:
    def __init__(self, bits: list[int]) -> None:
        self.bits = bits
        self.cursor = 0

    def peek_bit_count(self, n: int) -> list[int]:
        return self.bits[self.cursor:self.cursor + n]

    def get_bit_count(self, n: int) -> list[int]:
        out = self.peek_bit_count(n)
        self.cursor += n
        return out

    def get_bit(self) -> int:
        return self.get_bit_count(1)[0]

    def match_bits(self, expected_bits: list[int]) -> None:
        actual_bits = self.get_bit_count(len(expected_bits))
        if actual_bits != expected_bits:
            raise ValueError(
                f'Expected to read {"".join([str(b) for b in expected_bits])}, '
                f'but read {"".join([str(b) for b in actual_bits])}'
            )

    def parse_telemetry_word(self) -> TelemetryWord:
        # Ref: IS-GPS-200L, Figure 20-2
        tlm_prelude = [1, 0, 0, 0, 1, 0, 1, 1]
        self.match_bits(tlm_prelude)
        telemetry_message = self.get_bit_count(14)
        integrity_status_flag = self.get_bit()
        spare_bit = self.get_bit()
        parity_bits = self.get_bit_count(6)
        return TelemetryWord(
            telemetry_message=telemetry_message,
            integrity_status_flag=integrity_status_flag,
            spare_bit=spare_bit,
            parity_bits=parity_bits,
        )

    def parse_handover_word(self) -> HandoverWord:
        time_of_week = self.get_bit_count(17)
        alert_flag = self.get_bit()
        anti_spoof_flag = self.get_bit()
        subframe_id_codes = self.get_bit_count(3)
        to_be_solved = self.get_bit_count(2)
        parity_bits = self.get_bit_count(6)
        return HandoverWord(
            time_of_week=time_of_week,
            alert_flag=alert_flag,
            anti_spoof_flag=anti_spoof_flag,
            subframe_id_codes=subframe_id_codes,
            to_be_solved=to_be_solved,
            parity_bits=parity_bits,
        )


def main_decode_bits():
    bits = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    # TODO find dnyamically
    first_prelude_index = 155

    parser = BitParser(bits)
    parser.cursor = first_prelude_index

    while True:
        print(f'bit index {parser.cursor}')
        telemetry_word = parser.parse_telemetry_word()
        print(telemetry_word)
        handover_word = parser.parse_handover_word()
        print(handover_word)
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
        time_of_week_bits = handover_word.time_of_week
        # Each bit represents (1.5*(2^(bit_position+2))) seconds
        # Start with the LSB
        time_of_week_accumulator = 0
        for i, bit in enumerate(reversed(time_of_week_bits)):
            bit_granularity = 1.5 * (2 ** (i + 2))
            if bit == 1:
                time_of_week_accumulator += bit_granularity
        print(f'TOW accumulator: {time_of_week_accumulator}')
        minutes = time_of_week_accumulator / 60
        hours = minutes / 60
        days = hours / 24
        print(f'\t{minutes:.2f} minutes')
        print(f'\t{hours:.2f} hours')
        print(f'\t{days:.2f} days')

        parser.cursor += 240
        # Skip to the next one...
        # Gonna be 200



if __name__ == '__main__':
    #main_new()
    main_decode_bits()

    #test_acquire()
    #main()
