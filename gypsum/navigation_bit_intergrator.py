import collections
import logging
from dataclasses import dataclass

import numpy as np

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.config import RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_MEMORY_SIZE
from gypsum.config import RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_THRESHOLD
from gypsum.config import RECALCULATE_PSEUDOSYMBOL_PHASE_PERIOD
from gypsum.constants import BITS_PER_SECOND
from gypsum.constants import PSEUDOSYMBOLS_PER_NAVIGATION_BIT
from gypsum.constants import PSEUDOSYMBOLS_PER_SECOND
from gypsum.events import Event
from gypsum.tracker import BitValue, NavigationBitPseudosymbol
from gypsum.utils import chunks

_logger = logging.getLogger(__name__)


Percentage = float
BitPseudosymbolPhase = int


class EmitNavigationBitEvent(Event):
    def __init__(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        bit_value: BitValue
    ) -> None:
        self.receiver_timestamp = receiver_timestamp
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
class EmittedPseudosymbol:
    receiver_timestamp: ReceiverTimestampSeconds
    pseudosymbol: NavigationBitPseudosymbol


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

    def __post_init__(self) -> None:
        if self.last_seen_pseudosymbols is not None:
            raise ValueError(f'Cannot be set explicitly')
        if self.last_emitted_bits is not None:
            raise ValueError(f'Cannot be set explicitly')
        # 1000 to store the display the last 1 second of pseudosymbols, which matches the tracker history
        self.last_seen_pseudosymbols = collections.deque(maxlen=1000)
        # 50 to match a 1-second history period
        self.last_emitted_bits = collections.deque(maxlen=BITS_PER_SECOND)


class NavigationBitIntegrator:
    def __init__(self) -> None:
        self.history = NavigationBitIntegratorHistory()
        self.all_symbols: list[EmittedPseudosymbol] = []

        self.pseudosymbol_count_to_use_for_bit_phase_selection = PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 4

        # Maintain a rolling average of the last half-bit of pseudosymbols that we've seen
        self.rolling_average_window_size = PSEUDOSYMBOLS_PER_NAVIGATION_BIT // 2
        self.rolling_average_window = collections.deque(maxlen=self.rolling_average_window_size)

        self.resynchronize_bit_phase_period = PSEUDOSYMBOLS_PER_SECOND * RECALCULATE_PSEUDOSYMBOL_PHASE_PERIOD
        self.resynchronize_bit_phase_memory_size = RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_MEMORY_SIZE
        self.pseudosymbol_cursor = 0

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
        strength_score = sum([abs(x) for x in pseudosymbol_sums_per_bit]) / (len(pseudosymbols) / PSEUDOSYMBOLS_PER_NAVIGATION_BIT)

        # Average the strength scores across all the bits provided
        return strength_score / PSEUDOSYMBOLS_PER_NAVIGATION_BIT

    def _redetermine_bit_phase(self) -> BitPseudosymbolPhase | None:
        if len(self.history.last_seen_pseudosymbols) < self.pseudosymbol_count_to_use_for_bit_phase_selection:
            # We haven't yet seen enough pseudosymbols to select a bit phase
            return None

        # Only look at the last few bits
        pseudosymbols_to_consider = list(self.history.last_seen_pseudosymbols)[-PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 16:]
        # Try every possible bit phase
        pseudosymbols = np.array(list([x.pseudosymbol for x in pseudosymbols_to_consider]))
        bit_phase_to_confidence_score = dict()
        for possible_bit_phase in range(0, PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
            pseudosymbols_aligned_with_bit_phase = np.roll(pseudosymbols, -possible_bit_phase)
            # Compute a confidence score
            confidence = self._compute_bit_confidence_score(pseudosymbols_aligned_with_bit_phase)   # type: ignore
            bit_phase_to_confidence_score[possible_bit_phase] = confidence

        best_bit_phase = max(
            bit_phase_to_confidence_score, key=bit_phase_to_confidence_score.get    # type: ignore
        )
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
                _logger.info(f'Resetting bit phase because we failed to resolve too many bits in a row...')
                self._reset_selected_bit_phase()
        else:
            self.history.sequential_unknown_bit_value_counter = 0

        # TODO(PT): Come up with some condition to emit a LostBitCoherenceEvent (i.e. X% unknown bits
        # emitted in the last Y seconds).

        # The timestamp of the bit comes from the receiver timestamp of
        # the first pseudosymbol in the bit.
        timestamp = pseudosymbols[0].receiver_timestamp
        return EmitNavigationBitEvent(receiver_timestamp=timestamp, bit_value=bit_value)

    def _emit_bits_from_queued_pseudosymbols(self) -> list[Event]:
        if self.history.determined_bit_phase is None:
            return []

        events = []
        cursor = self.pseudosymbol_cursor
        for chunk in chunks(self.all_symbols[cursor:], PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
            events.append(self._emit_bit_from_pseudosymbols(list(chunk)))
            self.pseudosymbol_cursor += PSEUDOSYMBOLS_PER_NAVIGATION_BIT
            self.history.emitted_bit_count += 1

        return events

    def _should_resynchronize_bit_phase(self) -> bool:
        if self.history.processed_pseudosymbol_count % self.resynchronize_bit_phase_period == 0:
            _logger.info(f'Resynchronizing bit phase because the periodic job has fired')
            return True

        # Can't determine bit phase without any pseudosymbols to work with
        if self.history.processed_pseudosymbol_count == 0:
            return False

        # In the best case, we'd be on a bit boundary
        if self.history.processed_pseudosymbol_count % PSEUDOSYMBOLS_PER_NAVIGATION_BIT != 0:
            return False

        # Have we never detected a bit phase?
        if self.history.previous_bit_phase_decision is None:
            _logger.info(f'Resynchronizing bit phase because we\'ve never selected a phase before')
            return True

        # Have we failed too many bits in a row?
        last_few_bits = list(self.history.last_emitted_bits)[-self.resynchronize_bit_phase_memory_size:]
        # Ensure we have enough bits in the buffer
        if len(last_few_bits) == self.resynchronize_bit_phase_memory_size:
            failed_bit_count = len(list(filter(lambda x: x == BitValue.UNKNOWN, last_few_bits)))
            proportion_failures = failed_bit_count / len(last_few_bits)
            percent_failures = proportion_failures * 100
            if percent_failures >= RECALCULATE_PSEUDOSYMBOL_PHASE_BIT_HEALTH_THRESHOLD:
                _logger.info(f'Resynchronizing bit phase because too many of the last few bits were unresolved')
                return True

        return False

    def _resynchronize_bit_phase_if_necessary(self) -> list[Event]:
        if not self._should_resynchronize_bit_phase():
            return []

        # We're going to try resynchronizing our bit phase
        events = []
        previous_bit_phase_decision = self.history.previous_bit_phase_decision
        new_bit_phase = self._redetermine_bit_phase()
        self.history.previous_bit_phase_decision = new_bit_phase

        self.history.determined_bit_phase = new_bit_phase

        did_determine_first_bit_phase = previous_bit_phase_decision is None and new_bit_phase is not None
        if did_determine_first_bit_phase:
            print(f'******* FIRST Bit phase! {new_bit_phase}')
            if new_bit_phase > 0:
                self.pseudosymbol_cursor = new_bit_phase
        else:
            did_change_bit_phase = previous_bit_phase_decision is not None and new_bit_phase is not None and previous_bit_phase_decision != new_bit_phase
            if did_change_bit_phase:
                diff = new_bit_phase - previous_bit_phase_decision
                print(f'******* CHANGED bit phase {new_bit_phase}, prev {previous_bit_phase_decision}, diff {diff}!')
                self.pseudosymbol_cursor += diff

        return events

    def process_pseudosymbol(self, receiver_timestamp: ReceiverTimestampSeconds, pseudosymbol: NavigationBitPseudosymbol) -> list[Event]:
        # Smooth out the current pseuodsymbol value over a rolling average of half a bit's worth of pseudosymbols
        self.rolling_average_window.append(pseudosymbol.as_val())
        if len(self.rolling_average_window) < self.rolling_average_window_size:
            # Haven't yet seen enough symbols to start using our rolling average
            return []
        averaged_pseudosymbol_value = sum(self.rolling_average_window) // len(self.rolling_average_window)
        rounded_pseudosymbol_value = -1 if averaged_pseudosymbol_value < 0 else 1

        if False:
            self.all_symbols.append(pseudosymbol)
            self.smoothed_symbols.append(0 if rounded_pseudosymbol_value == -1 else 1)

            if len(self.all_symbols) > 24000:
                num = 4000
                all_symbols = self.all_symbols[-num:]
                smoothed_symbols = self.smoothed_symbols[-num:]
                emitted_bits = self.emitted_bits[-(num//20):]

                plt.ioff()
                fig = plt.figure(figsize=(6, 9))
                ax1 = fig.add_subplot(3, 1, 1)
                ax1.set_title("Recevied Pseudosymbols")
                ax1.plot([x.as_val() for x in all_symbols])

                ax2 = fig.add_subplot(3, 1, 2)
                ax2.set_title("Rolling Average Pseudosymbols")
                ax2.plot([x for x in smoothed_symbols])

                ax3 = fig.add_subplot(3, 1, 3)
                ax3.set_title("Emitted Bits")
                bits_as_runs = [*[0.5 for _ in range(180)]]
                for bit in emitted_bits:
                    val = bit.as_val() if bit != BitValue.UNKNOWN else 0.5
                    bits_as_runs.extend([val for _ in range(20)])
                #ax2.plot([x.as_val() if x != BitValue.UNKNOWN else 0.5 for x in self.history.last_emitted_bits])
                ax3.plot(bits_as_runs)
                plt.show(block=True)

        events: list[Event] = []
        emitted_pseudosymbol = EmittedPseudosymbol(
            receiver_timestamp=receiver_timestamp,
            pseudosymbol=NavigationBitPseudosymbol.from_val(rounded_pseudosymbol_value),
        )
        self.all_symbols.append(emitted_pseudosymbol)
        self.history.last_seen_pseudosymbols.append(emitted_pseudosymbol)

        self._resynchronize_bit_phase_if_necessary()
        events.extend(self._emit_bits_from_queued_pseudosymbols())

        self.history.processed_pseudosymbol_count += 1

        return events
