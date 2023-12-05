import logging
from dataclasses import dataclass

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.constants import PSEUDOSYMBOLS_PER_NAVIGATION_BIT
from gypsum.constants import PSEUDOSYMBOLS_PER_SECOND
from gypsum.events import Event
from gypsum.tracker import BitValue, NavigationBitPseudosymbol
from gypsum.utils import chunks

_logger = logging.getLogger(__name__)


Percentage = float


class EmitNavigationBitEvent(Event):
    def __init__(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        bit_value: BitValue
    ) -> None:
        self.receiver_timestamp = receiver_timestamp
        self.bit_value = bit_value


class DeterminedBitPhaseEvent(Event):
    def __init__(self, bit_phase: int) -> None:
        self.bit_phase = bit_phase


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


class NavigationBitIntegrator:
    def __init__(self) -> None:
        self.queued_pseudosymbols: list[EmittedPseudosymbol] = []
        self.determined_bit_phase: int | None = None
        self.bit_index = 0
    def _determine_bit_phase_from_queued_bits(self) -> list[Event]:
        # Have we seen enough bits to determine the navigation bit phase?
        # (The bound here can probably be lowered if useful)
        # PT: What if we determine a bit phase too early, before we've locked, and it's wrong?
        # Give ourselves 6 seconds to lock
        seconds_count_to_buffer_symbols_before_attempting_phase_selection = 60
        seconds_count_to_consider_for_symbol_phase_selection = 2
        if len(self.queued_pseudosymbols) < (PSEUDOSYMBOLS_PER_SECOND * seconds_count_to_buffer_symbols_before_attempting_phase_selection):
            # Not enough queued symbols to detect a bit phase
            # _logger.info(
            #    f'Pseudosymbol integrator hasn\'t yet determined bit phase '
            #    f'but has only processed {len(self.queued_pseudosymbols)} pseudosymbols'
            # )
            return []

        events = []
        _logger.info(
            f"Pseudosymbol integrator has seen enough bits ({len(self.queued_pseudosymbols)}), selecting a bit phase..."
        )
        # Look at the symbols from the final four seconds, as they'll probably be much closer to correct than the
        # initial symbols we see.
        # We'll need two seconds worth of symbols to select a phase, as we try different offsets up to 20 symbols
        # This could instead be 1 second of we rolled instead of sliding the start.

        # TODO(PT): This only considers one bit! Perhaps we should do multiple bits again
        symbols_considered_for_phase_selection = self.queued_pseudosymbols[
            PSEUDOSYMBOLS_PER_NAVIGATION_BIT * (
                seconds_count_to_buffer_symbols_before_attempting_phase_selection -
                seconds_count_to_consider_for_symbol_phase_selection
            ):
        ]
        phase_guess_to_confidence_score = {}
        for phase_guess in range(0, PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
            phase_guess_confidence = 0
            symbols_grouped_into_bits = list(
                chunks(symbols_considered_for_phase_selection[phase_guess:], PSEUDOSYMBOLS_PER_NAVIGATION_BIT)
            )
            for symbols_in_bit in symbols_grouped_into_bits:
                bit_confidence = abs(sum([symbol.pseudosymbol.as_val() for symbol in symbols_in_bit]))
                phase_guess_confidence += bit_confidence
            # Normalize based on the number of bit groups we looked at
            phase_guess_to_confidence_score[phase_guess] = phase_guess_confidence / len(
                symbols_grouped_into_bits
            )
            # This could be sensitive to tracking errors in this particular second of processing...

        highest_confidence_phase_offset = max(
            phase_guess_to_confidence_score, key=phase_guess_to_confidence_score.get  # type: ignore
        )
        highest_confidence_score = phase_guess_to_confidence_score[highest_confidence_phase_offset]
        # highest_confidence_phase_offset = int(np.argmax(confidence_scores))
        # highest_confidence_score = confidence_scores[highest_confidence_phase_offset]
        _logger.info(
            f"Highest confidence phase offset: {highest_confidence_phase_offset}. "
            f"Score: {highest_confidence_score}"
        )

        highest_confidence_as_percentage: Percentage = abs(
            highest_confidence_score / PSEUDOSYMBOLS_PER_NAVIGATION_BIT
        )

        if highest_confidence_as_percentage >= 0.70:
            self.determined_bit_phase = highest_confidence_phase_offset
            # Discard queued symbols from the first partial symbol
            self.queued_pseudosymbols = self.queued_pseudosymbols[self.determined_bit_phase:]
            events.append(DeterminedBitPhaseEvent(highest_confidence_phase_offset))
        else:
            _logger.info(
                f"Highest confidence bit phase was below confidence threshold: {highest_confidence_as_percentage}"
            )
            events.append(CannotDetermineBitPhaseEvent(highest_confidence_as_percentage))
        return events

    def process_pseudosymbol(self, receiver_timestamp: ReceiverTimestampSeconds, pseudosymbol: NavigationBitPseudosymbol) -> list[Event]:
        events: list[Event] = []
        self.queued_pseudosymbols.append(EmittedPseudosymbol(
            receiver_timestamp=receiver_timestamp,
            pseudosymbol=pseudosymbol,
        ))

        if self.determined_bit_phase is None:
            events.extend(self._determine_bit_phase_from_queued_bits())

        # We may have just determined the bit phase above, so check again
        if self.determined_bit_phase is not None:
            # Drain the symbol queue as much as we can
            while True:
                if len(self.queued_pseudosymbols) >= PSEUDOSYMBOLS_PER_NAVIGATION_BIT:
                    # _logger.info(f'Emitting bit from pseudosymbol queue with {len(self.queued_pseudosymbols)} symbols')
                    bit_pseudosymbols = self.queued_pseudosymbols[:PSEUDOSYMBOLS_PER_NAVIGATION_BIT]
                    # Consume these pseudosymbols by removing them from the queue
                    self.queued_pseudosymbols = self.queued_pseudosymbols[PSEUDOSYMBOLS_PER_NAVIGATION_BIT:]
                    pseudosymbol_sum = sum([s.pseudosymbol.as_val() for s in bit_pseudosymbols])
                    if pseudosymbol_sum > 0:
                        bit_value = BitValue.ONE
                    else:
                        bit_value = BitValue.ZERO

                    confidence_score: Percentage = abs(int((pseudosymbol_sum / PSEUDOSYMBOLS_PER_NAVIGATION_BIT) * 100))
                    self.bit_index += 1
                    if confidence_score >= 60:
                        # The timestamp of the bit comes from the receiver timestamp of
                        # the first pseudosymbol in the bit.
                        timestamp = bit_pseudosymbols[0].receiver_timestamp
                        events.append(EmitNavigationBitEvent(receiver_timestamp=timestamp, bit_value=bit_value))
                    else:
                        events.append(LostBitCoherenceEvent(confidence_score))
                        # Stop consuming bits now
                        break
                else:
                    # Not enough pseudosymbols to emit a bit
                    break

        return events
