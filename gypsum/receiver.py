import collections
import logging
from copy import deepcopy

import numpy as np

from gypsum.acquisition import GpsSatelliteDetector
from gypsum.antenna_sample_provider import AntennaSampleProvider, ReceiverTimestampSeconds
from gypsum.config import ACQUISITION_INTEGRATION_PERIOD_MS
from gypsum.config import ACQUISITION_SCAN_FREQUENCY
from gypsum.constants import MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX
from gypsum.constants import PRN_CHIP_COUNT
from gypsum.gps_ca_prn_codes import GpsSatelliteId, generate_replica_prn_signals
from gypsum.navigation_bit_intergrator import Event
from gypsum.navigation_message_decoder import EmitSubframeEvent
from gypsum.satellite import ALL_SATELLITE_IDS, GpsSatellite
from gypsum.satellite_signal_processing_pipeline import GpsSatelliteSignalProcessingPipeline, LostSatelliteLockError
from gypsum.utils import AntennaSamplesSpanningOneMs
from gypsum.world_model import DeterminedSatelliteOrbitEvent, GpsWorldModel

_logger = logging.getLogger(__name__)


class GpsReceiver:
    def __init__(self, antenna_samples_provider: AntennaSampleProvider) -> None:
        self.antenna_samples_provider = antenna_samples_provider

        # Generate the replica signals that we'll use to correlate against the received antenna signals upfront
        satellites_to_replica_prn_signals = generate_replica_prn_signals()
        self.satellites_by_id = {
            satellite_id: GpsSatellite(
                satellite_id=satellite_id,
                prn_code=code,
                scale_factor=antenna_samples_provider.get_attributes().samples_per_prn_transmission // PRN_CHIP_COUNT,
            )
            for satellite_id, code in satellites_to_replica_prn_signals.items()
        }
        # TODO(PT): Perhaps this state should belong to the detector.
        # And further, perhaps it should be somewhere easy to configure?
        # The receiver can remove satellites from the pool when it decides a satellite has been acquired
        # self.satellite_ids_eligible_for_acquisition = deepcopy(ALL_SATELLITE_IDS)
        # PT: The phase isn't about the chip offset, it's about "the timestamp of where the PRN starts"
        # Literally they're the same, but the latter makes more sense conceptually in terms of 'measuring the delay' -
        # you look at the timestamp where the PRN starts.
        # Example: timestamped HOW and we receive it 7 milliseconds later (for 20km distance)
        self.satellite_ids_eligible_for_acquisition = [GpsSatelliteId(id=32)]
        self.satellite_detector = GpsSatelliteDetector(self.satellites_by_id)
        # Used during acquisition to integrate correlation over a longer period than a millisecond.
        self.rolling_samples_buffer: collections.deque = collections.deque(maxlen=ACQUISITION_INTEGRATION_PERIOD_MS)

        self.tracked_satellite_ids_to_processing_pipelines: dict[
            GpsSatelliteId, GpsSatelliteSignalProcessingPipeline
        ] = {}
        self.subframe_count = 0

        self.world_model = GpsWorldModel()

        self._time_since_last_acquisition_scan = 0.0

    def step(self) -> None:
        """Run one 'iteration' of the GPS receiver. This consumes one millisecond of antenna data."""
        receiver_timestamp: ReceiverTimestampSeconds
        samples: AntennaSamplesSpanningOneMs
        # TODO(PT): Cache this somewhere?
        receiver_timestamp, samples = self.antenna_samples_provider.get_samples(
            self.antenna_samples_provider.get_attributes().samples_per_prn_transmission,
        )
        # receiver_timestamp, samples = self.antenna_samples_provider.get_samples(SAMPLES_PER_PRN_TRANSMISSION)
        # Firstly, record this sample in our rolling buffer
        self.rolling_samples_buffer.append(samples)

        # If we need to perform acquisition, do so now
        seconds_since_start = self.antenna_samples_provider.seconds_since_start()
        if (
            seconds_since_start - self._time_since_last_acquisition_scan
            >= ACQUISITION_SCAN_FREQUENCY
        ):
            # Update the timestamp even if we decide not to try to acquire more satellites
            self._time_since_last_acquisition_scan = seconds_since_start
            if len(self.tracked_satellite_ids_to_processing_pipelines) < MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX:
                _logger.info(
                    f"Will perform acquisition search because we're only "
                    f"tracking {len(self.tracked_satellite_ids_to_processing_pipelines)} satellites."
                )
                _logger.info(f"{receiver_timestamp}: Subframe count: {self.subframe_count}")
                self._perform_acquisition()

        # Continue tracking each acquired satellite
        satellite_ids_to_tracker_events = self._track_acquired_satellites(receiver_timestamp, samples)
        # And keep track of updates to our world model
        satellite_ids_to_world_model_events = {}
        for satellite_id, events in satellite_ids_to_tracker_events.items():
            for event in events:
                if isinstance(event, EmitSubframeEvent):
                    self.subframe_count += 1
                    emit_subframe_event: EmitSubframeEvent = event
                    subframe = emit_subframe_event.subframe
                    print(f"*** Subframe {subframe.subframe_id.name} from {satellite_id}:")
                    from dataclasses import fields

                    for field in fields(subframe):
                        print(f"\t{field.name}: {getattr(subframe, field.name)}")

                    world_model_events_from_this_satellite = self.world_model.handle_subframe_emitted(
                        satellite_id, emit_subframe_event
                    )
                    satellite_ids_to_world_model_events[satellite_id] = world_model_events_from_this_satellite
                else:
                    raise NotImplementedError(f"Unhandled event type: {type(event)}")

        # Process updates to our world model
        for satellite_id, world_model_events in satellite_ids_to_world_model_events.items():
            for world_model_event in world_model_events:
                if isinstance(world_model_event, DeterminedSatelliteOrbitEvent):
                    print(f"Determined the orbit of {satellite_id}! {world_model_event.orbital_parameters}")
                    orbit_params = world_model_event.orbital_parameters

    def _perform_acquisition(self) -> None:
        newly_acquired_satellite_ids = self._perform_acquisition_on_satellite_ids(self.satellite_ids_eligible_for_acquisition)
        # The satellites that we've just acquired no longer need to be searched for in the acquisition stage
        self.satellite_ids_eligible_for_acquisition = [x for x in self.satellite_ids_eligible_for_acquisition if x not in newly_acquired_satellite_ids]

    def _perform_acquisition_on_satellite_ids(self, satellite_ids: list[GpsSatelliteId]) -> list[GpsSatelliteId]:
        # To improve signal-to-noise ratio during acquisition, we integrate antenna data over 20ms.
        # Therefore, we keep a rolling buffer of the last few samples.
        # If this buffer isn't primed yet, we can't do any work yet.
        if len(self.rolling_samples_buffer) < ACQUISITION_INTEGRATION_PERIOD_MS:
            # _logger.info(f"Skipping acquisition attempt because the history buffer isn't primed yet.")
            return []

        # TODO(PT): Properly model the cursor field
        _logger.info(
            f"{self.antenna_samples_provider.seconds_since_start() + self.antenna_samples_provider.utc_start_time}: Performing acquisition search over {len(satellite_ids)} satellites ({self.subframe_count} subframes so far)."
        )

        samples_for_integration_period = np.concatenate(self.rolling_samples_buffer)
        newly_acquired_satellites = self.satellite_detector.detect_satellites_in_antenna_data(
            satellite_ids,
            samples_for_integration_period,
            self.antenna_samples_provider.get_attributes(),
        )
        for satellite_acquisition_result in newly_acquired_satellites:
            sat_id = satellite_acquisition_result.satellite_id
            satellite = self.satellites_by_id[sat_id]
            self.tracked_satellite_ids_to_processing_pipelines[sat_id] = GpsSatelliteSignalProcessingPipeline(
                satellite, satellite_acquisition_result, self.antenna_samples_provider.get_attributes()
            )
        return [n.satellite_id for n in newly_acquired_satellites]

    def _track_acquired_satellites(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        samples: AntennaSamplesSpanningOneMs,
    ) -> dict[GpsSatelliteId, list[Event]]:
        satellite_ids_to_events = {}
        satellite_ids_to_reacquire = []
        for satellite_id, pipeline in self.tracked_satellite_ids_to_processing_pipelines.items():
            try:
                if events := pipeline.process_samples(
                    receiver_timestamp,
                    self.antenna_samples_provider.seconds_since_start(),
                    samples,
                ):
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
