import collections
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from gypsum.acquisition import GpsSatelliteDetector
from gypsum.gps_ca_prn_codes import generate_replica_prn_signals
from gypsum.constants import SAMPLES_PER_PRN_TRANSMISSION, MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX
from gypsum.navigation_bit_intergrator import NavigationBitIntegrator
from gypsum.navigation_message_decoder import NavigationMessageDecoder
from gypsum.satellite import ALL_SATELLITE_IDS
from gypsum.satellite import GpsSatellite
from gypsum.antenna_sample_provider import AntennaSampleProvider
from gypsum.config import ACQUISITION_INTEGRATION_PERIOD_MS
from gypsum.tracker import GpsSatelliteTracker
from gypsum.tracker import GpsSatelliteTrackingParameters
from gypsum.utils import AntennaSamplesSpanningOneMs
from gypsum.utils import chunks
from gypsum.utils import does_list_contain_sublist


_logger = logging.getLogger(__name__)


class UnknownEventError(Exception):
    pass


class TrackingState(Enum):
    PROVISIONAL_PROBE = auto()
    LOCKED = auto()


class GpsSatelliteSignalProcessingPipeline:
    satellite: GpsSatellite
    state: TrackingState

    # Tracks PRN code phase shift, carrier wave Doppler shift, and carrier wave phase
    tracker: GpsSatelliteTracker

    pseudosymbol_integrator: NavigationBitIntegrator
    navigation_message_decoder: NavigationMessageDecoder

    def __init__(self, satellite: GpsSatellite, acquisition_result: SatelliteAcquisitionAttemptResult) -> None:
        self.satellite = satellite
        self.state = TrackingState.PROVISIONAL_PROBE
        tracking_params = GpsSatelliteTrackingParameters(
            satellite=satellite,
            current_doppler_shift=acquisition_result.doppler_shift,
            current_carrier_wave_phase_shift=acquisition_result.carrier_wave_phase_shift,
            current_prn_code_phase_shift=acquisition_result.prn_phase_shift,
            doppler_shifts=[],
            carrier_wave_phases=[],
            carrier_wave_phase_errors=[],
            navigation_bit_pseudosymbols=[],
        )
        self.tracker = GpsSatelliteTracker(tracking_params)
        self.pseudosymbol_integrator = NavigationBitIntegrator()
        self.navigation_message_decoder = NavigationMessageDecoder()

    def process_samples(self, samples: AntennaSamplesSpanningOneMs, sample_index: int):
        pseudosymbol = self.tracker.process_samples(samples, sample_index)
        integrator_events = self.pseudosymbol_integrator.process_pseudosymbol(pseudosymbol)

        for event in integrator_events:
            if isinstance(event, DeterminedBitPhaseEvent):
                _logger.info(
                    f'Integrator for SV({self.satellite.satellite_id.id}) has determined bit phase {event.bit_phase}'
                )

            elif isinstance(event, EmitNavigationBitEvent):
                _logger.info(
                    f'Integrator for SV({self.satellite.satellite_id.id}) emitted bit {event.bit_value}'
                )
                self.navigation_message_decoder.process_bit_from_satellite(self.satellite, event.bit_value)

            else:
                raise UnknownEventError(type(event))

        # The pseudosymbol integrator will only emit a bit every 20 pseudosymbols
        #if not maybe_navigation_bit:
        #    return


class GpsReceiver:
    def __init__(self, antenna_samples_provider: AntennaSampleProvider) -> None:
        self.antenna_samples_provider = antenna_samples_provider

        # Generate the replica signals that we'll use to correlate against the received antenna signals upfront
        satellites_to_replica_prn_signals = generate_replica_prn_signals()
        self.satellites_by_id = {
            satellite_id: GpsSatellite(satellite_id=satellite_id, prn_code=code)
            for satellite_id, code in satellites_to_replica_prn_signals.items()
        }

        self.satellite_trackers: list[GpsSatelliteTracker] = []
        self.satellite_ids_eligible_for_acquisition = deepcopy(ALL_SATELLITE_IDS)

        self.satellite_detector = GpsSatelliteDetector(self.satellites_by_id)
        # Used during acquisition to integrate correlation over a longer period than a millisecond.
        self.rolling_samples_buffer = collections.deque(maxlen=ACQUISITION_INTEGRATION_PERIOD_MS)
        # The main loop could always peel off two milliseconds, then pass whatever it can to the trackers
        # Even better: the trackers can return a "would block" / needs more samples

        self.navigation_bit_integrator = NavigationBitIntegrator()
        self.navigation_message_decoder = NavigationMessageDecoder()

    def step(self):
        """Run one 'iteration' of the GPS receiver. This consumes one millisecond of antenna data."""
        sample_index = self.antenna_samples_provider.cursor
        samples: AntennaSamplesSpanningOneMs = self.antenna_samples_provider.get_samples(SAMPLES_PER_PRN_TRANSMISSION)
        # PT: Instead of trying to find a roll that works across all time, dynamically adjust where we consider the start to be?

        if (
            len(self.satellite_trackers)
            and len(self.satellite_trackers[-1].tracking_params.carrier_wave_phase_errors) > 12000 * 8
        ) or len(samples) < SAMPLES_PER_PRN_TRANSMISSION:
            for tracker in reversed(self.satellite_trackers):
                sat = tracker.tracking_params
                self.decode_nav_bits(sat)
                plt.plot(sat.doppler_shifts[::50])
                plt.title(f"Doppler shift for {sat.satellite.satellite_id.id}")
                plt.show(block=True)
                plt.plot(sat.carrier_wave_phases[::50])
                plt.title(f"Carrier phase for {sat.satellite.satellite_id.id}")
                plt.show(block=True)
                plt.plot(sat.carrier_wave_phase_errors[::50])
                plt.title(f"Carrier phase errors for {sat.satellite.satellite_id.id}")
                plt.show(block=True)
            import sys

            sys.exit(0)

        # Firstly, record this sample in our rolling buffer
        self.rolling_samples_buffer.append(samples)

        # If we need to perform acquisition, do so now
        if len(self.satellite_trackers) < MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX:
            _logger.info(
                f"Will perform acquisition search because we're only "
                f"tracking {len(self.satellite_trackers)} satellites"
            )
            self._perform_acquisition()

        # Continue tracking each acquired satellite
        self._track_acquired_satellites(samples, sample_index)

    def decode_nav_bits(self, sat: GpsSatelliteTrackingParameters):
        navigation_bit_pseudosymbols = sat.navigation_bit_pseudosymbols
        confidence_scores = []
        for roll in range(0, 20):
            phase_shifted_bits = navigation_bit_pseudosymbols[roll:]
            confidences = []
            for twenty_pseudosymbols in chunks(phase_shifted_bits, 20):
                integrated_value = sum(twenty_pseudosymbols)
                confidences.append(abs(integrated_value))
            # Compute an overall confidence score for this offset
            confidence_scores.append(np.mean(confidences))

        # print(f"Confidence scores: {confidence_scores}")
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
        print(f"Bit count: {len(digital_bits)}")
        print(f"Bits:          {digital_bits}")
        print(f"Inverted bits: {inverted_bits}")

        preamble = [1, 0, 0, 0, 1, 0, 1, 1]
        print(f"Preamble {preamble} found in bits? {does_list_contain_sublist(digital_bits, preamble)}")
        print(f"Preamble {preamble} found in inverted bits? {does_list_contain_sublist(inverted_bits, preamble)}")

        def get_matches(l, sub):
            return [l[pos : pos + len(sub)] == sub for pos in range(0, len(l) - len(sub) + 1)]

        preamble_starts_in_digital_bits = [
            x[0] for x in (np.argwhere(np.array(get_matches(digital_bits, preamble)) == True))
        ]
        print(f"Preamble starts in bits:          {preamble_starts_in_digital_bits}")
        preamble_starts_in_inverted_bits = [
            x[0] for x in (np.argwhere(np.array(get_matches(inverted_bits, preamble)) == True))
        ]
        print(f"Preamble starts in inverted bits: {preamble_starts_in_inverted_bits}")

    def _perform_acquisition(self) -> None:
        # To improve signal-to-noise ratio during acquisition, we integrate antenna data over 20ms.
        # Therefore, we keep a rolling buffer of the last few samples.
        # If this buffer isn't primed yet, we can't do any work yet.
        if len(self.rolling_samples_buffer) < ACQUISITION_INTEGRATION_PERIOD_MS:
            _logger.info(f"Skipping acquisition attempt because the history buffer isn't primed yet.")
            return

        _logger.info(
            f"Performing acquisition search over {len(self.satellite_ids_eligible_for_acquisition)} satellites."
        )

        samples_for_integration_period = np.concatenate(self.rolling_samples_buffer)
        newly_acquired_satellites = self.satellite_detector.detect_satellites_in_antenna_data(
            self.satellite_ids_eligible_for_acquisition,
            samples_for_integration_period,
        )
        for satellite_acquisition_result in newly_acquired_satellites:
            tracking_params = GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[satellite_acquisition_result.satellite_id],
                current_doppler_shift=satellite_acquisition_result.doppler_shift,
                current_carrier_wave_phase_shift=satellite_acquisition_result.carrier_wave_phase_shift,
                current_prn_code_phase_shift=satellite_acquisition_result.prn_phase_shift,
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            )
            self.satellite_trackers.append(GpsSatelliteTracker(tracking_params))

    def _track_acquired_satellites(self, samples: AntennaSamplesSpanningOneMs, sample_index: int):
        for tracker in self.satellite_trackers:
            pseudosymbol = tracker.process_samples(samples, sample_index)
            satellite = tracker.tracking_params.satellite
            maybe_navigation_bit = self.navigation_bit_integrator.process_pseudosymbol_from_satellite(
                satellite,
                pseudosymbol
            )

            if not maybe_navigation_bit:
                continue

            self.navigation_message_decoder.process_bit_from_satellite(satellite, maybe_navigation_bit)
