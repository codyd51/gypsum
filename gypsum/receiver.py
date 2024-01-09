import collections
import logging
from copy import deepcopy
# PT: requests is just used to communicate with our own dashboard webserver and display the current receiver state.
import requests

import numpy as np

from gypsum.acquisition import GpsSatelliteDetector
from gypsum.antenna_sample_provider import AntennaSampleProvider, ReceiverTimestampSeconds
from gypsum.config import ACQUISITION_INTEGRATION_PERIOD_MS
from gypsum.config import ACQUISITION_SCAN_FREQUENCY
from gypsum.config import DASHBOARD_WEBSERVER_SCAN_PERIOD
from gypsum.config import DASHBOARD_WEBSERVER_URL
from gypsum.constants import MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX
from gypsum.constants import PRN_CHIP_COUNT
from gypsum.gps_ca_prn_codes import GpsSatelliteId, generate_replica_prn_signals
from gypsum.navigation_bit_intergrator import Event
from gypsum.navigation_message_decoder import EmitSubframeEvent
from gypsum.satellite import ALL_SATELLITE_IDS, GpsSatellite
from gypsum.tracker import LostSatelliteLockError
from gypsum.satellite_signal_processing_pipeline import GpsSatelliteSignalProcessingPipeline
from gypsum.units import ReceiverDataSeconds
from gypsum.units import Seconds
from gypsum.utils import AntennaSamplesSpanningOneMs
from gypsum.world_model import DeterminedSatelliteOrbitEvent, GpsWorldModel
from web_dashboard.messages import GpsReceiverState
from web_dashboard.messages import SetCurrentReceiverStateRequest

_logger = logging.getLogger(__name__)


class GpsReceiver:
    def __init__(
        self,
        antenna_samples_provider: AntennaSampleProvider,
        only_acquire_satellite_ids: list[GpsSatelliteId] | None = None,
        present_matplotlib_satellite_tracker: bool = False,
        present_web_ui: bool = False,
    ) -> None:
        self.antenna_samples_provider = antenna_samples_provider
        self.should_present_matplotlib_satellite_tracker = present_matplotlib_satellite_tracker
        self.should_present_web_ui = present_web_ui

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
        # PT: The phase isn't about the chip offset, it's about "the timestamp of where the PRN starts"
        # Literally they're the same, but the latter makes more sense conceptually in terms of 'measuring the delay' -
        # you look at the timestamp where the PRN starts.
        # Example: timestamped HOW and we receive it 7 milliseconds later (for 20km distance)
        self.satellite_ids_eligible_for_acquisition = deepcopy(ALL_SATELLITE_IDS)
        if only_acquire_satellite_ids is not None:
            _logger.info(f'Only acquiring user-specified satellite IDs: {", ".join([str(x.id) for x in only_acquire_satellite_ids])}')
            self.satellite_ids_eligible_for_acquisition = only_acquire_satellite_ids

        self.satellite_detector = GpsSatelliteDetector(self.satellites_by_id)
        # Used during acquisition to integrate correlation over a longer period than a millisecond.
        self.rolling_samples_buffer: collections.deque = collections.deque(maxlen=ACQUISITION_INTEGRATION_PERIOD_MS)

        self.tracked_satellite_ids_to_processing_pipelines: dict[
            GpsSatelliteId, GpsSatelliteSignalProcessingPipeline
        ] = {}
        self.subframe_count = 0

        self.world_model = GpsWorldModel(self.antenna_samples_provider.get_attributes().samples_per_prn_transmission)

        self._time_since_last_acquisition_scan: ReceiverDataSeconds | None = None
        self._timestamp_of_last_dashboard_update: ReceiverDataSeconds | None = None

        self._time_since_last_dashboard_server_scan = 0.0
        self._is_connected_to_dashboard_server = False

    def step(self) -> None:
        """Run one 'iteration' of the GPS receiver. This consumes one millisecond of antenna data."""
        receiver_timestamp: ReceiverTimestampSeconds
        samples: AntennaSamplesSpanningOneMs
        # TODO(PT): Cache this somewhere?
        receiver_timestamp, samples = self.antenna_samples_provider.get_samples(
            self.antenna_samples_provider.get_attributes().samples_per_prn_transmission,
        )

        # Inform the world model that another millisecond has elapsed for our receiver, since we just listened
        # to the antenna for 1ms.
        self.world_model.handle_processed_1ms_of_antenna_data()

        # Hook up to the dashboard webserver. Periodically try to connect to the dashboard, and send our state
        # update if we're connected.
        # Do this before we process the samples. The only reason for this is so that the dashboard can get some
        # initial state to display before we perform the initial acquisition scan.
        self._scan_for_dashboard_webserver_if_necessary()
        self._send_receiver_state_to_dashboard_if_necessary(receiver_timestamp)

        # Record this sample in our rolling buffer
        self.rolling_samples_buffer.append(samples)

        # If we need to perform acquisition, do so now
        self._perform_acquisition_if_necessary()

        # Continue tracking each acquired satellite
        satellite_ids_to_tracker_events = self._track_acquired_satellites(receiver_timestamp, samples)
        # And keep track of updates to our world model
        # Firstly, note that each of these satellites has emitted another PRN.
        # This allows us to keep track of the passage of satellite time with reference to the HOW timestamp.
        for satellite_id in self.tracked_satellite_ids_to_processing_pipelines.keys():
            tracker_params = self.tracked_satellite_ids_to_processing_pipelines[satellite_id].tracker.tracking_params
            self.world_model.handle_prn_observed(satellite_id, tracker_params.current_prn_code_phase_shift)

        satellite_ids_to_world_model_events = {}
        for satellite_id, events in satellite_ids_to_tracker_events.items():
            for event in events:
                if isinstance(event, EmitSubframeEvent):
                    events = self._handle_subframe_emitted_event(satellite_id, event)
                    satellite_ids_to_world_model_events[satellite_id] = events
                else:
                    raise NotImplementedError(f"Unhandled event type: {type(event)}")

        # Process updates to our world model
        for satellite_id, world_model_events in satellite_ids_to_world_model_events.items():
            for world_model_event in world_model_events:
                if isinstance(world_model_event, DeterminedSatelliteOrbitEvent):
                    print(f"Determined the orbit of {satellite_id}! {world_model_event.orbital_parameters}")
                else:
                    raise NotImplementedError(f'Unhandled event type: {type(world_model_event)}')

    def _perform_acquisition_if_necessary(self):
        seconds_since_start = self.antenna_samples_provider.seconds_since_start()
        if (
            self._time_since_last_acquisition_scan is not None
            and seconds_since_start - self._time_since_last_acquisition_scan < ACQUISITION_SCAN_FREQUENCY
        ):
            # We're not scheduled to run acquisition now
            return

        if not self._can_perform_acquisition():
            return

        # Don't update the last acquisition timestamp unless we'll really have an opportunity to run acquisition
        # But do update the timestamp even if we decide not to try to acquire more satellites
        self._time_since_last_acquisition_scan = seconds_since_start

        # TODO(PT): We could introduce a 'channel count' config parameter that controls how many satellites we'll
        # simultaneously track
        #if len(self.tracked_satellite_ids_to_processing_pipelines) >= MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX:
        #    return

        _logger.info(
            f"Will perform acquisition search because we're only "
            f"tracking {len(self.tracked_satellite_ids_to_processing_pipelines)} satellites."
        )
        self._perform_acquisition()

    def _handle_subframe_emitted_event(self, satellite_id: GpsSatelliteId, event: EmitSubframeEvent) -> list[Event]:
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
        return list(world_model_events_from_this_satellite)

    def _perform_acquisition(self) -> None:
        newly_acquired_satellite_ids = self._perform_acquisition_on_satellite_ids(self.satellite_ids_eligible_for_acquisition)
        # The satellites that we've just acquired no longer need to be searched for in the acquisition stage
        self.satellite_ids_eligible_for_acquisition = [x for x in self.satellite_ids_eligible_for_acquisition if x not in newly_acquired_satellite_ids]

    def _can_perform_acquisition(self) -> bool:
        # If we haven't seen enough samples to integrate the PRN correlation over a few milliseconds,
        # we can't do any work yet.
        if len(self.rolling_samples_buffer) < ACQUISITION_INTEGRATION_PERIOD_MS:
            return False

        if len(self.satellite_ids_eligible_for_acquisition) == 0:
            return False

        return True

    def _perform_acquisition_on_satellite_ids(self, satellite_ids: list[GpsSatelliteId]) -> list[GpsSatelliteId]:
        # To improve signal-to-noise ratio during acquisition, we integrate antenna data over 20ms.
        # Therefore, we keep a rolling buffer of the last few samples.
        if not self._can_perform_acquisition():
            return []

        # TODO(PT): Properly model the cursor field
        _logger.info(
            f"{self.antenna_samples_provider.seconds_since_start() + self.antenna_samples_provider.utc_start_time}: "
            f"Performing acquisition search over {len(satellite_ids)} "
            f"satellites ({self.subframe_count} subframes so far)."
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
                should_present_matplotlib_satellite_tracker=self.should_present_matplotlib_satellite_tracker,
                should_present_web_ui=self.should_present_web_ui,
            )
        return [n.satellite_id for n in newly_acquired_satellites]

    def _track_acquired_satellites(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        samples: AntennaSamplesSpanningOneMs,
    ) -> dict[GpsSatelliteId, list[Event]]:
        satellite_ids_to_events = {}
        satellite_ids_to_reacquire = []
        satellite_ids_to_drop = []
        for satellite_id, pipeline in self.tracked_satellite_ids_to_processing_pipelines.items():
            try:
                if events := pipeline.process_samples(
                    receiver_timestamp,
                    self.antenna_samples_provider.seconds_since_start(),
                    samples,
                ):
                    satellite_ids_to_events[satellite_id] = events
            except LostSatelliteLockError:
                pipeline.handle_satellite_dropped()
                satellite_ids_to_reacquire.append(satellite_id)
                # Don't modify the mapping while iterating it
                satellite_ids_to_drop.append(satellite_id)

        for satellite_id in satellite_ids_to_drop:
            self._drop_satellite(satellite_id)

        return satellite_ids_to_events

    def _drop_satellite(self, satellite_id: GpsSatelliteId) -> None:
        if satellite_id not in self.tracked_satellite_ids_to_processing_pipelines:
            raise ValueError(f'Tried to drop an untracked satellite {satellite_id}')

        del self.tracked_satellite_ids_to_processing_pipelines[satellite_id]
        # Inform the world model that we're no longer reliably counting PRNs for this satellite
        self.world_model.handle_lost_satellite_lock(satellite_id)

    def _send_receiver_state_to_dashboard_if_necessary(self, receiver_timestamp: ReceiverTimestampSeconds) -> None:
        # Nothing to do if we're not connected to the webserver
        if not self._is_connected_to_dashboard_server:
            return

        # TODO(PT): Promote to config file item?
        dashboard_refresh_interval = 1
        if (
            self._timestamp_of_last_dashboard_update is not None
            and receiver_timestamp - self._timestamp_of_last_dashboard_update < dashboard_refresh_interval
        ):
            return

        self._timestamp_of_last_dashboard_update = receiver_timestamp

        try:
            resp = requests.post(
                DASHBOARD_WEBSERVER_URL,
                json=SetCurrentReceiverStateRequest(
                    current_state=GpsReceiverState(
                        receiver_timestamp=receiver_timestamp,
                        satellite_ids_eligible_for_acquisition=self.satellite_ids_eligible_for_acquisition,
                        dashboard_figures=[x.tracker_visualizer.rendered_dashboard_png_base64 for x in self.tracked_satellite_ids_to_processing_pipelines.values()],
                        tracked_satellite_count=len(self.tracked_satellite_ids_to_processing_pipelines),
                        processed_subframe_count=self.subframe_count,
                        # TODO(PT): Fix
                        #satellite_ids_to_orbital_parameters=self.world_model.satellite_ids_to_orbital_parameters,
                        satellite_ids_to_orbital_parameters={},
                        tracked_satellite_ids=[x for x in self.tracked_satellite_ids_to_processing_pipelines.keys()],
                        satellite_ids_ineligible_for_acquisition=[GpsSatelliteId(id=x) for x in range(0, 33) if x not in [32, 25, 28]]
                    )
                ).model_dump_json()
            )
            resp.raise_for_status()
        except:
            _logger.info('Lost connection to webserver while pushing receiver state update.')
            self._is_connected_to_dashboard_server = False

    def _scan_for_dashboard_webserver_if_necessary(self):
        # No work to do if we're already connected
        if self._is_connected_to_dashboard_server:
            return

        # No work to do if we haven't waited long enough since our last scan
        seconds_since_start = self.antenna_samples_provider.seconds_since_start()
        if (
            seconds_since_start == 0
            or seconds_since_start - self._time_since_last_dashboard_server_scan
            < DASHBOARD_WEBSERVER_SCAN_PERIOD
        ):
            return

        # Time to scan for the dashboard webserver
        #_logger.info(f'Scanning for dashboard webserver...')
        # TODO(PT): seconds_since_start() is currently "recording seconds", but here we actually want "process seconds"
        self._time_since_last_dashboard_server_scan = seconds_since_start
        try:
            resp = requests.get(DASHBOARD_WEBSERVER_URL)
            resp.raise_for_status()
        except (requests.ConnectionError, requests.HTTPError):
            # _logger.info(f'Did not detect the webserver.')
            pass
        else:
            _logger.info(f'Detected that the webserver is now live.')
            self._is_connected_to_dashboard_server = True
