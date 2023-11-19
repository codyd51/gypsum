import logging
from enum import Enum, auto
from typing import Callable, Type

from gypsum.acquisition import SatelliteAcquisitionAttemptResult
from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.config import SECONDARY_PLL_BANDWIDTH
from gypsum.events import UnknownEventError
from gypsum.navigation_bit_intergrator import (
    CannotDetermineBitPhaseEvent,
    DeterminedBitPhaseEvent,
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
from gypsum.tracker import GpsSatelliteTracker, GpsSatelliteTrackingParameters
from gypsum.utils import AntennaSamplesSpanningOneMs


_logger = logging.getLogger(__name__)


class LostSatelliteLockError(Exception):
    pass


class TrackingState(Enum):
    # TODO(PT): Used?
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
        self.tracker = GpsSatelliteTracker(tracking_params, SECONDARY_PLL_BANDWIDTH)
        self.pseudosymbol_integrator = NavigationBitIntegrator()
        self.navigation_message_decoder = NavigationMessageDecoder()
        self.current_receiver_timestamp = 0

    def process_samples(self, receiver_timestamp: ReceiverTimestampSeconds, samples: AntennaSamplesSpanningOneMs) -> list[Event]:
        self.current_receiver_timestamp = receiver_timestamp
        pseudosymbol = self.tracker.process_samples(receiver_timestamp, samples)
        integrator_events = self.pseudosymbol_integrator.process_pseudosymbol(receiver_timestamp, pseudosymbol)

        integrator_event_type_to_callback: dict[Type[Event], Callable[[Event], list[Event] | None]] = {  # type: ignore
            DeterminedBitPhaseEvent: self._handle_integrator_determined_bit_phase,
            CannotDetermineBitPhaseEvent: self._handle_integrator_cannot_determine_bit_phase,
            LostBitCoherenceEvent: self._handle_integrator_lost_bit_coherence,
            EmitNavigationBitEvent: self._handle_integrator_emitted_bit,
        }
        events_to_return = []
        for event in integrator_events:
            event_type = type(event)
            if event_type not in integrator_event_type_to_callback:
                raise UnknownEventError(event_type)
            callback = integrator_event_type_to_callback[event_type]
            events_to_return.extend(callback(event) or [])
        return events_to_return

    def _handle_integrator_determined_bit_phase(self, event: DeterminedBitPhaseEvent) -> None:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(f"Integrator for SV({satellite_id}) has determined bit phase {event.bit_phase}")

    def _handle_integrator_cannot_determine_bit_phase(self, event: CannotDetermineBitPhaseEvent) -> None:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(
            f"{self.current_receiver_timestamp}: Integrator for SV({satellite_id}) could not determine bit phase. Confidence: {int(event.confidence*100)}%"
        )
        # Untrack this satellite as the bits are low confidence
        raise LostSatelliteLockError()

    def _handle_integrator_lost_bit_coherence(self, event: LostBitCoherenceEvent) -> None:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(
            f"{self.current_receiver_timestamp}: Integrator for SV({satellite_id}) lost bit coherence. "
            f"Confidence for bit {self.pseudosymbol_integrator.bit_index}: {event.confidence}%"
        )
        raise LostSatelliteLockError()

    def _handle_integrator_emitted_bit(self, event: EmitNavigationBitEvent) -> list[Event]:
        # _logger.info(f'handling bit {self.pseudosymbol_integrator.bit_index-1}')
        decoder_events = self.navigation_message_decoder.process_bit_from_satellite(event)

        decoder_event_type_to_callback: dict[Type[Event], Callable[[Event], list[Event] | None]] = {  # type: ignore
            DeterminedSubframePhaseEvent: self._handle_decoder_determined_subframe_phase,
            CannotDetermineSubframePhaseEvent: self._handle_decoder_cannot_determine_subframe_phase,
            EmitSubframeEvent: self._handle_decoder_emitted_subframe,
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
        # TODO(PT): Wait longer for a subframe to appear..?
        raise NotImplementedError(f"Should wait longer for a subframe to appear..?")

    def _handle_decoder_emitted_subframe(self, event: EmitSubframeEvent) -> list[Event]:
        satellite_id = self.satellite.satellite_id.id
        _logger.info(f"Decoder for SV({satellite_id}) emitted a subframe:")
        _logger.info(f"\tTelemetry word: {event.telemetry_word}")
        _logger.info(f"\tHandover word: {event.handover_word}")
        return [event]


