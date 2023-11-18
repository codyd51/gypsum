import collections
import logging

import numpy as np

from gypsum.acquisition import GpsSatelliteDetector
from gypsum.antenna_sample_provider import AntennaSampleProvider
from gypsum.config import ACQUISITION_INTEGRATION_PERIOD_MS
from gypsum.constants import SAMPLES_PER_PRN_TRANSMISSION
from gypsum.gps_ca_prn_codes import GpsSatelliteId, generate_replica_prn_signals
from gypsum.navigation_bit_intergrator import Event
from gypsum.navigation_message_decoder import EmitSubframeEvent
from gypsum.satellite import GpsSatellite
from gypsum.satellite_signal_processing_pipeline import GpsSatelliteSignalProcessingPipeline
from gypsum.satellite_signal_processing_pipeline import LostSatelliteLockError
from gypsum.tracker import GpsSatelliteTrackingParameters
from gypsum.utils import AntennaSamplesSpanningOneMs, chunks, does_list_contain_sublist

_logger = logging.getLogger(__name__)


class GpsReceiver:
    def __init__(self, antenna_samples_provider: AntennaSampleProvider) -> None:
        self.antenna_samples_provider = antenna_samples_provider

        # Generate the replica signals that we'll use to correlate against the received antenna signals upfront
        satellites_to_replica_prn_signals = generate_replica_prn_signals()
        self.satellites_by_id = {
            satellite_id: GpsSatellite(satellite_id=satellite_id, prn_code=code)
            for satellite_id, code in satellites_to_replica_prn_signals.items()
        }
        # TODO(PT): Perhaps this state should belong to the detector.
        # The receiver can remove satellites from the pool when it decides a satellite has been acquired
        #self.satellite_ids_eligible_for_acquisition = deepcopy(ALL_SATELLITE_IDS)
        self.satellite_ids_eligible_for_acquisition = [GpsSatelliteId(id=32)]
        self.satellite_detector = GpsSatelliteDetector(self.satellites_by_id)
        # Used during acquisition to integrate correlation over a longer period than a millisecond.
        self.rolling_samples_buffer = collections.deque(maxlen=ACQUISITION_INTEGRATION_PERIOD_MS)

        self.tracked_satellite_ids_to_processing_pipelines: dict[
            GpsSatelliteId, GpsSatelliteSignalProcessingPipeline
        ] = {}
        self.subframe_count = 0

    def step(self):
        """Run one 'iteration' of the GPS receiver. This consumes one millisecond of antenna data."""
        sample_index = self.antenna_samples_provider.cursor
        samples: AntennaSamplesSpanningOneMs = self.antenna_samples_provider.get_samples(SAMPLES_PER_PRN_TRANSMISSION)
        # Firstly, record this sample in our rolling buffer
        self.rolling_samples_buffer.append(samples)

        # If we need to perform acquisition, do so now
        # if len(self.tracked_satellite_ids_to_processing_pipelines) < MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX:
        if len(self.tracked_satellite_ids_to_processing_pipelines) < 1:
            _logger.info(
                f"Will perform acquisition search because we're only "
                f"tracking {len(self.tracked_satellite_ids_to_processing_pipelines)} satellites."
            )
            _logger.info(f"{self.antenna_samples_provider.cursor}: Subframe count: {self.subframe_count}")
            self._perform_acquisition()

        # Continue tracking each acquired satellite
        satellite_ids_to_subframes = self._track_acquired_satellites(samples, sample_index)
        if satellite_ids_to_subframes:
            print(satellite_ids_to_subframes)
        for satellite_id, emit_subframe_events in satellite_ids_to_subframes.items():
            for emit_subframe_event in emit_subframe_events:
                emit_subframe_event: EmitSubframeEvent = emit_subframe_event
                subframe = emit_subframe_event.subframe
                print(f'*** Subframe {subframe.subframe_id.name} from {satellite_id}:')
                from dataclasses import fields
                for field in fields(subframe):
                    print(f'\t{field.name}: {getattr(subframe, field.name)}')

            self.subframe_count += len(emit_subframe_events)

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
        inverted = [0, 1, 1, 1, 0, 1, 0, 0]
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
        self._perform_acquisition_on_satellite_ids(self.satellite_ids_eligible_for_acquisition)

    def _perform_acquisition_on_satellite_ids(self, satellite_ids: list[GpsSatelliteId]) -> list[GpsSatelliteId]:
        # To improve signal-to-noise ratio during acquisition, we integrate antenna data over 20ms.
        # Therefore, we keep a rolling buffer of the last few samples.
        # If this buffer isn't primed yet, we can't do any work yet.
        if len(self.rolling_samples_buffer) < ACQUISITION_INTEGRATION_PERIOD_MS:
            _logger.info(f"Skipping acquisition attempt because the history buffer isn't primed yet.")
            return []

        # TODO(PT): Properly model the cursor field
        _logger.info(
            f"{self.antenna_samples_provider.cursor}: Performing acquisition search over {len(satellite_ids)} satellites ({self.subframe_count} subframes so far)."
        )

        samples_for_integration_period = np.concatenate(self.rolling_samples_buffer)
        newly_acquired_satellites = self.satellite_detector.detect_satellites_in_antenna_data(
            satellite_ids,
            samples_for_integration_period,
        )
        for satellite_acquisition_result in newly_acquired_satellites:
            sat_id = satellite_acquisition_result.satellite_id
            satellite = self.satellites_by_id[sat_id]
            self.tracked_satellite_ids_to_processing_pipelines[sat_id] = GpsSatelliteSignalProcessingPipeline(
                satellite, satellite_acquisition_result
            )
        return [n.satellite_id for n in newly_acquired_satellites]

    def _track_acquired_satellites(
        self, samples: AntennaSamplesSpanningOneMs, sample_index: int
    ) -> dict[GpsSatelliteId, list[Event]]:
        satellite_ids_to_events = {}
        satellite_ids_to_reacquire = []
        for satellite_id, pipeline in self.tracked_satellite_ids_to_processing_pipelines.items():
            try:
                if events := pipeline.process_samples(samples, sample_index):
                    satellite_ids_to_events[satellite_id] = events
            except LostSatelliteLockError:
                satellite_ids_to_reacquire.append(satellite_id)

        for satellite_id in satellite_ids_to_reacquire:
            del self.tracked_satellite_ids_to_processing_pipelines[satellite_id]
            print("Trying to re-acquire...")
            acquired_satellite_ids = self._perform_acquisition_on_satellite_ids([satellite_id])
            if len(acquired_satellite_ids) != 1:
                # Failed to re-acquire this satellite
                print(f"Failed to re-acquire!")
                # TODO(PT): Put it back on the queue of available-to-acquire?
        return satellite_ids_to_events
