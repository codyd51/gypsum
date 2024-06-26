import collections
import logging
from dataclasses import dataclass

import numpy as np

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.config import (
    RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_MEMORY_SIZE,
    RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_THRESHOLD,
    RECALCULATE_PSEUDOSYMBOL_PHASE_PERIOD,
)
from gypsum.constants import BITS_PER_SECOND, PSEUDOSYMBOLS_PER_NAVIGATION_BIT, PSEUDOSYMBOLS_PER_SECOND
from gypsum.events import Event
from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.tracker import BitValue, NavigationBitPseudosymbol
from gypsum.tracker import EmittedPseudosymbol
from gypsum.utils import chunks

_logger = logging.getLogger(__name__)


Percentage = float
BitPseudosymbolPhase = int


# TODO(PT): All the integrator events should subclass a common IntegratorEvent.
#  This makes typing clearer in the event handlers.
class EmitNavigationBitEvent(Event):
    def __init__(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        trailing_edge_receiver_timestamp: ReceiverTimestampSeconds,
        bit_value: BitValue
    ) -> None:
        self.receiver_timestamp = receiver_timestamp
        self.trailing_edge_receiver_timestamp = trailing_edge_receiver_timestamp
        self.bit_value = bit_value


class CannotDetermineBitPhaseEvent(Event):
    def __init__(self, confidence: Percentage) -> None:
        self.confidence = confidence


class LostBitCoherenceEvent(Event):
    def __init__(self, confidence: Percentage) -> None:
        self.confidence = confidence


class LostBitPhaseCoherenceError(Exception):
    pass


@dataclass
class NavigationBitIntegratorHistory:
    last_seen_pseudosymbols: collections.deque[EmittedPseudosymbol] = None

    last_emitted_bits: collections.deque[BitValue] = None
    previous_bit_phase_decision: int | None = None
    determined_bit_phase: int | None = None
    failed_bit_count: int = 0
    emitted_bit_count: int = 0
    processed_pseudosymbol_count: int = 0
    sequential_unknown_bit_value_counter: int = 0

    queued_pseudosymbols: list[EmittedPseudosymbol] = None
    # This should be a multiple of the pseudosymbols per bit, plus the determined bit phase.
    # This will jitter back and forth, both as we consume bits and as our phase decider makes small adjustments.
    pseudosymbol_cursor_within_queue: int = 0

    # Maintain a rolling average of the last half-bit of pseudosymbols that we've seen
    rolling_average_window_size: int = PSEUDOSYMBOLS_PER_NAVIGATION_BIT // 2
    rolling_average_window: collections.deque[int] = None

    def __post_init__(self) -> None:
        if self.last_seen_pseudosymbols is not None:
            raise ValueError(f"Cannot be set explicitly")
        if self.queued_pseudosymbols is not None:
            raise ValueError(f"Cannot be set explicitly")
        if self.last_emitted_bits is not None:
            raise ValueError(f"Cannot be set explicitly")
        if self.rolling_average_window is not None:
            raise ValueError(f"Cannot be set explicitly")
        # 1000 to store the display the last 1 second of pseudosymbols, which matches the tracker history
        self.last_seen_pseudosymbols = collections.deque(maxlen=1000)
        # 50 to match a 1-second history period
        self.last_emitted_bits = collections.deque(maxlen=BITS_PER_SECOND)

        # This is our 'working buffer' of pseudosymbols. Incoming pseudosymbols will be queued up in this buffer
        # until we emit bits from them. Additionally, this buffer provides a short 'history' of pseudosymbols. This is
        # extremely useful when our phase detector decides it needs to shift our phase 'backwards'. If we didn't keep
        # a short history of the last pseudosymbols we consumed, we wouldn't be able to provide this.
        self.queued_pseudosymbols = []
        # Integrate the pseudosymbol values we observe over half a bit duration.
        # This provides some resilience against high-frequency spurious pseudosymbol flips.
        self.rolling_average_window = collections.deque(maxlen=self.rolling_average_window_size)


class NavigationBitIntegrator:
    def __init__(self, satellite_id: GpsSatelliteId) -> None:
        self.satellite_id = satellite_id
        self.history = NavigationBitIntegratorHistory()
        self.pseudosymbol_count_to_use_for_bit_phase_selection = PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 4
        self.resynchronize_bit_phase_period = PSEUDOSYMBOLS_PER_SECOND * RECALCULATE_PSEUDOSYMBOL_PHASE_PERIOD
        self.resynchronize_bit_phase_memory_size = RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_MEMORY_SIZE
        self.slide = 0

    def _reset_selected_bit_phase(self):
        _logger.info(f"Resetting selected bit phase...")
        self.history.determined_bit_phase = None

    def _compute_bit_confidence_score(self, pseudosymbols: list[NavigationBitPseudosymbol]) -> float:
        pseudosymbol_sums_per_bit = []
        for i, pseudosymbols_in_bit in enumerate(chunks(pseudosymbols, PSEUDOSYMBOLS_PER_NAVIGATION_BIT)):
            # A bit is 'strongest' if all the pseudosymbols have the same sign
            # Therefore, sum all the pseudosymbols and then take the absolute value. The closer the sum is to 20,
            # the 'stronger' the agreement of the pseudosymbols.
            summed_values = sum(x.as_val() for x in pseudosymbols_in_bit)
            pseudosymbol_sums_per_bit.append(summed_values)
        strength_score = sum([abs(x) for x in pseudosymbol_sums_per_bit]) / (
            len(pseudosymbols) / PSEUDOSYMBOLS_PER_NAVIGATION_BIT
        )

        # Average the strength scores across all the bits provided
        return strength_score / PSEUDOSYMBOLS_PER_NAVIGATION_BIT

    def _redetermine_bit_phase(self) -> BitPseudosymbolPhase | None:
        if len(self.history.last_seen_pseudosymbols) < self.pseudosymbol_count_to_use_for_bit_phase_selection:
            # We haven't yet seen enough pseudosymbols to select a bit phase
            return None

        # Only look at the last few bits
        pseudosymbols_to_consider = list(self.history.last_seen_pseudosymbols)[-PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 16 :]
        # Try every possible bit phase
        pseudosymbols = np.array(list([x.pseudosymbol for x in pseudosymbols_to_consider]))
        bit_phase_to_confidence_score = dict()
        for possible_bit_phase in range(0, PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
            pseudosymbols_aligned_with_bit_phase = np.roll(pseudosymbols, -possible_bit_phase)
            # Compute a confidence score
            confidence = self._compute_bit_confidence_score(pseudosymbols_aligned_with_bit_phase)  # type: ignore
            bit_phase_to_confidence_score[possible_bit_phase] = confidence

        best_bit_phase = max(bit_phase_to_confidence_score, key=bit_phase_to_confidence_score.get)  # type: ignore
        return best_bit_phase

    def _get_bit_value_from_pseudosymbols(self, pseudosymbols: list[EmittedPseudosymbol]) -> BitValue:
        pseudosymbol_sum = sum([s.pseudosymbol.as_val() for s in pseudosymbols])
        if pseudosymbol_sum > 0:
            bit_value = BitValue.ONE
        else:
            bit_value = BitValue.ZERO

        # Divide by however many pseudosymbols are actually in the bit, as we might have been provided
        # with a partial bit
        confidence_score: Percentage = abs(int((pseudosymbol_sum / len(pseudosymbols)) * 100))
        if confidence_score <= 50:
            bit_value = BitValue.UNKNOWN
        return bit_value

    def _emit_bit_from_pseudosymbols(self, pseudosymbols: list[EmittedPseudosymbol]) -> EmitNavigationBitEvent:
        bit_value = self._get_bit_value_from_pseudosymbols(pseudosymbols)
        self.history.last_emitted_bits.append(bit_value)
        if bit_value == BitValue.UNKNOWN:
            self.history.sequential_unknown_bit_value_counter += 1
            self.history.failed_bit_count += 1
            if self.history.sequential_unknown_bit_value_counter >= 30:
                # TODO(PT): This might cause issues because it throws our subframe phase out of alignment?
                # We might need to emit an event to tell the subframe decoder to select a new phase now
                _logger.info(f"Resetting bit phase because we failed to resolve too many bits in a row...")
                self._reset_selected_bit_phase()
        else:
            self.history.sequential_unknown_bit_value_counter = 0

        # TODO(PT): Come up with some condition to emit a LostBitCoherenceEvent (i.e. X% unknown bits
        # emitted in the last Y seconds).

        # The timestamp of the bit comes from the receiver timestamp of
        # the first pseudosymbol in the bit.
        first_pseudosymbol = pseudosymbols[0]
        last_pseudosymbol = pseudosymbols[-1]
        timestamp = first_pseudosymbol.start_of_pseudosymbol
        trailing_edge_timestamp = last_pseudosymbol.end_of_pseudosymbol
        return EmitNavigationBitEvent(
            receiver_timestamp=timestamp,
            trailing_edge_receiver_timestamp=trailing_edge_timestamp,
            bit_value=bit_value,
        )

    def _emit_bits_from_queued_pseudosymbols(self) -> list[Event]:
        if self.history.determined_bit_phase is None:
            return []

        events = []
        cursor = self.history.pseudosymbol_cursor_within_queue
        for chunk in chunks(self.history.queued_pseudosymbols[cursor:], PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
            events.append(self._emit_bit_from_pseudosymbols(list(chunk)))
            self.history.pseudosymbol_cursor_within_queue += PSEUDOSYMBOLS_PER_NAVIGATION_BIT
            self.history.emitted_bit_count += 1

        # Drop old symbols that we don't need to keep around anymore
        # To keep everything in alignment, only do this once per bit period
        if len(self.history.queued_pseudosymbols) >= PSEUDOSYMBOLS_PER_NAVIGATION_BIT:
            offset_from_end = len(self.history.queued_pseudosymbols) - self.history.pseudosymbol_cursor_within_queue
            self.history.queued_pseudosymbols = self.history.queued_pseudosymbols[-PSEUDOSYMBOLS_PER_NAVIGATION_BIT:]
            self.history.pseudosymbol_cursor_within_queue = PSEUDOSYMBOLS_PER_NAVIGATION_BIT - offset_from_end

        return events

    def _should_resynchronize_bit_phase(self) -> bool:
        if self.history.processed_pseudosymbol_count % self.resynchronize_bit_phase_period == 0:
            # _logger.info(f"Resynchronizing bit phase because the periodic job has fired")
            return True

        # Can't determine bit phase without any pseudosymbols to work with
        if self.history.processed_pseudosymbol_count == 0:
            return False

        # In the best case, we'd be on a bit boundary
        if self.history.processed_pseudosymbol_count % PSEUDOSYMBOLS_PER_NAVIGATION_BIT != 0:
            return False

        # Have we never detected a bit phase?
        if self.history.previous_bit_phase_decision is None:
            # _logger.info(f"Resynchronizing bit phase because we've never selected a phase before")
            return True

        # Have we failed too many bits in a row?
        last_few_bits = list(self.history.last_emitted_bits)[-self.resynchronize_bit_phase_memory_size :]
        # Ensure we have enough bits in the buffer
        if len(last_few_bits) == self.resynchronize_bit_phase_memory_size:
            failed_bit_count = len(list(filter(lambda x: x == BitValue.UNKNOWN, last_few_bits)))
            proportion_failures = failed_bit_count / len(last_few_bits)
            percent_failures = proportion_failures * 100
            if percent_failures >= RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_THRESHOLD:
                # _logger.info(f"Resynchronizing bit phase because too many of the last few bits were unresolved")
                return True

        return False

    def _resynchronize_bit_phase_if_necessary(self) -> list[Event]:
        if not self._should_resynchronize_bit_phase():
            return []

        # We're going to try resynchronizing our bit phase
        events = []
        previous_bit_phase_decision = self.history.previous_bit_phase_decision
        new_bit_phase = self._redetermine_bit_phase()
        # _logger.info(f'Resynchronizing bit phase {previous_bit_phase_decision=}, new={new_bit_phase=}...')

        self.history.previous_bit_phase_decision = new_bit_phase
        self.history.determined_bit_phase = new_bit_phase

        did_determine_first_bit_phase = previous_bit_phase_decision is None and new_bit_phase is not None
        if did_determine_first_bit_phase:
            if new_bit_phase > 0:
                self.history.pseudosymbol_cursor_within_queue = new_bit_phase
                self.slide = new_bit_phase
        else:
            did_change_bit_phase = (
                previous_bit_phase_decision is not None
                and new_bit_phase is not None
                and previous_bit_phase_decision != new_bit_phase
            )
            if did_change_bit_phase:
                diff = new_bit_phase - previous_bit_phase_decision
                self.slide += diff
                self.history.pseudosymbol_cursor_within_queue += diff

        return events

    def process_pseudosymbol(self, receiver_timestamp: ReceiverTimestampSeconds, pseudosymbol: EmittedPseudosymbol) -> list[Event]:
        events: list[Event] = []
        pseudosymbol.cursor_at_emit_time = self.slide
        self.history.queued_pseudosymbols.append(pseudosymbol)
        self.history.last_seen_pseudosymbols.append(pseudosymbol)

        # TODO(PT): Make this more robust...
        # Currently, it appears as though bit phase realignment is kicking satellite #32 into a bad state,
        # and this is a bandaid to help us get to a position fix.
        if receiver_timestamp < 40:
            self._resynchronize_bit_phase_if_necessary()

        events.extend(self._emit_bits_from_queued_pseudosymbols())

        self.history.processed_pseudosymbol_count += 1

        return events
