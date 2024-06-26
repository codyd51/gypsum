import logging
from enum import Enum, auto
from typing import Callable, Type

from gypsum.acquisition import SatelliteAcquisitionAttemptResult
from gypsum.antenna_sample_provider import AntennaSampleChunk
from gypsum.antenna_sample_provider import SampleProviderAttributes
from gypsum.events import UnknownEventError
from gypsum.navigation_bit_intergrator import (
    CannotDetermineBitPhaseEvent,
    EmitNavigationBitEvent,
    Event,
    LostBitCoherenceEvent,
    NavigationBitIntegrator,
)
from gypsum.navigation_message_decoder import (
    CannotDetermineSubframePhaseEvent,
    DeterminedSubframePhaseEvent,
    EmitSubframeEvent,
    NavigationMessageDecoder,
)
from gypsum.satellite import GpsSatellite
from gypsum.tracker import GpsSatelliteTracker, GpsSatelliteTrackingParameters, LostSatelliteLockError
from gypsum.tracker_visualizer import GpsSatelliteTrackerVisualizer

_logger = logging.getLogger(__name__)


class TrackingState(Enum):
    # TODO(PT): Used?
    PROVISIONAL_PROBE = auto()
    LOCKED = auto()


class GpsSatelliteSignalProcessingPipeline:
    satellite: GpsSatellite
    state: TrackingState

    # Tracks PRN code phase shift, carrier wave Doppler shift, and carrier wave phase
    tracker: GpsSatelliteTracker
    tracker_visualizer: GpsSatelliteTrackerVisualizer

    pseudosymbol_integrator: NavigationBitIntegrator
    navigation_message_decoder: NavigationMessageDecoder

    def __init__(
        self,
        satellite: GpsSatellite,
        acquisition_result: SatelliteAcquisitionAttemptResult,
        stream_attributes: SampleProviderAttributes,
        should_present_matplotlib_satellite_tracker: bool = False,
        should_present_web_ui: bool = False,
    ) -> None:
        self.satellite = satellite
        self.state = TrackingState.PROVISIONAL_PROBE
        tracking_params = GpsSatelliteTrackingParameters(
            satellite=satellite,
            current_doppler_shift=acquisition_result.doppler_shift,
            current_carrier_wave_phase_shift=acquisition_result.carrier_wave_phase_shift,
            current_prn_code_phase_shift=acquisition_result.prn_phase_shift,
            doppler_shifts=[],
        )
        self.tracker = GpsSatelliteTracker(tracking_params, stream_attributes)
        # TODO(PT): Add another option so that we can render to the dashboard without also presenting the matplotlib window
        self.tracker_visualizer = GpsSatelliteTrackerVisualizer(
            satellite.satellite_id,
            should_render=should_present_matplotlib_satellite_tracker or should_present_web_ui,
            should_present=should_present_matplotlib_satellite_tracker,
        )
        self.pseudosymbol_integrator = NavigationBitIntegrator(satellite.satellite_id)
        self.navigation_message_decoder = NavigationMessageDecoder()

    def process_samples(
        self,
        receiver_samples_chunk: AntennaSampleChunk,
    ) -> list[Event]:
        pseudosymbol = self.tracker.process_samples(receiver_samples_chunk)

        integrator_events = self.pseudosymbol_integrator.process_pseudosymbol(receiver_samples_chunk.start_time, pseudosymbol)

        integrator_event_type_to_callback: dict[Type[Event], Callable[[Event], list[Event] | None]] = {  # type: ignore
            CannotDetermineBitPhaseEvent: self._handle_integrator_cannot_determine_bit_phase,  # type: ignore
            LostBitCoherenceEvent: self._handle_integrator_lost_bit_coherence,  # type: ignore
            EmitNavigationBitEvent: self._handle_integrator_emitted_bit,  # type: ignore
        }
        events_to_return: list[Event] = []
        for event in integrator_events:
            event_type = type(event)
            if event_type not in integrator_event_type_to_callback:
                raise UnknownEventError(event_type)
            callback = integrator_event_type_to_callback[event_type]
            events_to_return.extend(callback(event) or [])

        # Pipeline is all done with this chunk of samples, push state updates to the GUI
        self.tracker_visualizer.step(
            receiver_samples_chunk.end_time,
            self.tracker.tracking_params,
            self.pseudosymbol_integrator.history,
            self.navigation_message_decoder.history,
        )

        return events_to_return

    def _handle_integrator_cannot_determine_bit_phase(self, event: CannotDetermineBitPhaseEvent) -> None:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(
            f"Integrator for SV({satellite_id}) could not determine bit phase. Confidence: {int(event.confidence*100)}%"
        )
        # Untrack this satellite as the bits are low confidence
        raise LostSatelliteLockError()

    def _handle_integrator_lost_bit_coherence(self, event: LostBitCoherenceEvent) -> None:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(
            f"Integrator for SV({satellite_id}) lost bit coherence. "
            f"Confidence for bit {self.pseudosymbol_integrator.history.emitted_bit_count}: {event.confidence}%"
        )
        # Untrack this satellite as our bit quality went too far downhill
        raise LostSatelliteLockError()

    def _handle_integrator_emitted_bit(self, bit_event: EmitNavigationBitEvent) -> list[Event]:
        decoder_events = self.navigation_message_decoder.process_bit_from_satellite(bit_event)

        decoder_event_type_to_callback: dict[Type[Event], Callable[[Event], list[Event] | None]] = {  # type: ignore
            DeterminedSubframePhaseEvent: self._handle_decoder_determined_subframe_phase,  # type: ignore
            CannotDetermineSubframePhaseEvent: self._handle_decoder_cannot_determine_subframe_phase,  # type: ignore
            EmitSubframeEvent: self._handle_decoder_emitted_subframe,  # type: ignore
        }
        events_to_return = []
        for event in decoder_events:
            event_type = type(event)
            if event_type not in decoder_event_type_to_callback:
                raise UnknownEventError(event_type)
            callback = decoder_event_type_to_callback[event_type]
            events_to_return.extend(callback(event) or [])
        return events_to_return

    def _handle_decoder_determined_subframe_phase(self, event: DeterminedSubframePhaseEvent) -> None:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(f"Decoder for SV({satellite_id}) has determined subframe phase {event.subframe_phase}")

    def _handle_decoder_cannot_determine_subframe_phase(self, event: CannotDetermineSubframePhaseEvent) -> None:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(f"Decoder for SV({satellite_id}) could not determine subframe phase.")
        # Untrack this satellite as we weren't able to identify subframe boundaries
        # (as our bit quality must be too low).
        raise LostSatelliteLockError()

    def _handle_decoder_emitted_subframe(self, event: EmitSubframeEvent) -> list[Event]:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(f"Decoder for SV({satellite_id}) emitted a subframe:")
        _logger.info(f"\tTelemetry word: {event.telemetry_word}")
        _logger.info(f"\tHandover word: {event.handover_word}")
        return [event]

    def handle_satellite_dropped(self) -> None:
        # Allow the visualizer to clean up
        self.tracker_visualizer.handle_satellite_dropped()
