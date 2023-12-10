import collections
import logging
import statistics
from dataclasses import dataclass

import math
import numpy as np

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
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


@dataclass
class NavigationBitIntegratorHistory:
    last_seen_pseudosymbols: collections.deque[EmittedPseudosymbol] = None
    last_emitted_bits: collections.deque[BitValue] = None
    previous_bit_phase_decision: int | None = None
    determined_bit_phase: int | None = None
    is_bit_phase_locked: bool = False
    consecutive_agreeing_bit_phase_decisions: int = 0

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
        self.queued_pseudosymbols: list[EmittedPseudosymbol] = []
        self.pseudosymbol_count_to_use_for_bit_phase_selection = PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 4
        #self.last_few_pseudosymbols = collections.deque(maxlen=self.pseudosymbol_count_to_use_for_bit_phase_selection)
        self.history = NavigationBitIntegratorHistory()
        self.bit_index = 0
        self.sequential_unknown_bit_value_counter = 0
        self._processed_pseudosymbol_count = 0

    def _determine_bit_phase_from_pseudosymbols(self, pseudosymbols: list[EmittedPseudosymbol]):
        # TODO(PT): I think we need a better way of continuously adjusting the bit phase.
        # We're trying to determine one phase after many seconds of processing, then apply it retrospectively to
        # pseudosymbols that were collected seconds ago. If the phase has drifted just a bit (presumably due to
        # tracking oscillations), the whole thing gets off-kilter, as we trim a bit halfway through and have seconds
        # of bad output bits in the face of good input pseudosymbols.
        # We could say "average of last 10 symbols is -1", and "average of these 10 symbols is 1", so there's a bit transition
        if len(pseudosymbols) % PSEUDOSYMBOLS_PER_NAVIGATION_BIT != 0:
            raise ValueError(f'This method must be provided a multiple of PSEUDOSYMBOLS_PER_NAVIGATION_BIT')

        # Combine all the pseudosymbols we've been provided in a circular buffer of pseudosymbols, so we can
        # consider the whole range for phase selection

        import matplotlib.pyplot as plt
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot([p.pseudosymbol.as_val() for p in pseudosymbols[-200:]])
        plt.show()
        plt.ion()

        phase_guess_to_confidence_score = {}
        for phase_guess in range(0, PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
            phase_guess_confidence = 0
            symbols_rolled_due_to_phase_guess = np.roll(pseudosymbols, phase_guess)
            symbols_grouped_into_bits = list(chunks(symbols_rolled_due_to_phase_guess, PSEUDOSYMBOLS_PER_NAVIGATION_BIT))
            for symbols_in_bit in symbols_grouped_into_bits:
                bit_confidence = abs(sum([symbol.pseudosymbol.as_val() for symbol in symbols_in_bit]))
                phase_guess_confidence += bit_confidence
            # Normalize based on the number of bit groups we looked at
            phase_guess_to_confidence_score[phase_guess] = phase_guess_confidence / len(
                symbols_grouped_into_bits
            )

        highest_confidence_phase_offset = max(
            phase_guess_to_confidence_score, key=phase_guess_to_confidence_score.get  # type: ignore
        )
        highest_confidence_score = phase_guess_to_confidence_score[highest_confidence_phase_offset]

        highest_confidence_as_percentage: Percentage = abs(
            highest_confidence_score / PSEUDOSYMBOLS_PER_NAVIGATION_BIT
        )

        _logger.info(
            f"Highest confidence phase offset: {highest_confidence_phase_offset}. "
            f"Score: {highest_confidence_as_percentage * 100}"
        )

        if highest_confidence_as_percentage >= 0.80:
            return highest_confidence_phase_offset % PSEUDOSYMBOLS_PER_NAVIGATION_BIT
        return None

    def _determine_bit_phase_from_queued_pseudosymbols(self) -> list[Event]:
        # Have we seen enough pseudosymbols to make an initial phase selection?
        symbols_required_to_make_initial_phase_selection = PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 2
        if len(self.queued_pseudosymbols) < symbols_required_to_make_initial_phase_selection:
            # Not enough queued symbols to detect a bit phase
            # _logger.info(
            #    f'Pseudosymbol integrator hasn\'t yet determined bit phase '
            #    f'but has only processed {len(self.queued_pseudosymbols)} pseudosymbols'
            # )
            return []

        # Only attempt to select a bit phase once we've collected an even multiple of pseudosymbols per bit
        if len(self.queued_pseudosymbols) % PSEUDOSYMBOLS_PER_NAVIGATION_BIT != 0:
            return []

        events = []

        if determined_bit_phase := self._determine_bit_phase_from_pseudosymbols(self.queued_pseudosymbols):
            self.history.determined_bit_phase = determined_bit_phase
            # Discard queued symbols from the first partial bit
            self.queued_pseudosymbols = self.queued_pseudosymbols[self.history.determined_bit_phase:]
            events.append(DeterminedBitPhaseEvent(determined_bit_phase))
        else:
            # Keep waiting for more pseudosymbols to come in, up to a maximum allowance
            if len(self.queued_pseudosymbols) < PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 50 * 30:
                # Continue to allow the tracker to run, although we haven't been able to identify a bit phase so far.
                pass
            else:
                _logger.info(f'Failed to identify a bit phase in a generous tracking span.')
                # TODO(PT): -1 isn't super meaningful here, this argument should be removed?
                events.append(CannotDetermineBitPhaseEvent(-1))
        return events

    def _determine_bit_phase_from_queued_pseudosymbols2(self) -> list[Event]:
        # Have we seen enough bits to determine the navigation bit phase?
        # Give ourselves 6 seconds to lock
        seconds_count_to_buffer_symbols_before_attempting_phase_selection = 60
        # (The bound here can probably be lowered if useful)
        # PT: What if we determine a bit phase too early, before we've locked, and it's wrong?
        seconds_count_to_consider_for_symbol_phase_selection = seconds_count_to_buffer_symbols_before_attempting_phase_selection // 2
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
            self.history.determined_bit_phase = highest_confidence_phase_offset % PSEUDOSYMBOLS_PER_NAVIGATION_BIT
            # Discard queued symbols from the first partial symbol
            self.queued_pseudosymbols = self.queued_pseudosymbols[self.history.determined_bit_phase:]
            events.append(DeterminedBitPhaseEvent(highest_confidence_phase_offset))
        else:
            _logger.info(
                f"Highest confidence bit phase was below confidence threshold: {highest_confidence_as_percentage}"
            )
            events.append(CannotDetermineBitPhaseEvent(highest_confidence_as_percentage))
        return events

    def _reset_selected_bit_phase(self):
        _logger.info(f"Resetting selected bit phase...")
        self.history.determined_bit_phase = None

    def _compute_bit_confidence_score(self, pseudosymbols: list[NavigationBitPseudosymbol]) -> float:

        strength_scores = []
        pseudosymbol_sums_per_bit = []
        for i, pseudosymbols_in_bit in enumerate(chunks(pseudosymbols, PSEUDOSYMBOLS_PER_NAVIGATION_BIT)):
            # A bit is 'strongest' if all the pseudosymbols have the same sign
            # Therefore, sum all the pseudosymbols and then take the absolute value. The closer the sum is to 20,
            # the 'stronger' the agreement of the pseudosymbols.
            summed_values = sum(x.as_val() for x in pseudosymbols_in_bit)
            pseudosymbol_sums_per_bit.append(summed_values)
            #strength_score = abs(summed_values) / PSEUDOSYMBOLS_PER_NAVIGATION_BIT
            #strength_scores.append(strength_score)
            #_logger.info(f'Strength for bit {i}: {strength_score} ({summed_values})')
        strength_score = sum([abs(x) for x in pseudosymbol_sums_per_bit]) / (len(pseudosymbols) / PSEUDOSYMBOLS_PER_NAVIGATION_BIT)
        #_logger.info(f'\tPseudosymbol sums per bit: {pseudosymbol_sums_per_bit}, strength {strength_score}')

        # Average the strength scores across all the bits provided
        #return statistics.mean(strength_scores)
        return strength_score / 20

    def _redetermine_bit_phase(self) -> BitPseudosymbolPhase | None:
        if len(self.history.last_seen_pseudosymbols) < self.pseudosymbol_count_to_use_for_bit_phase_selection:
            # We haven't yet seen enough pseudosymbols to select a bit phase
            return None

        # Only look at the last 4 bits
        #pseudosymbols_to_consider = list(self.history.last_seen_pseudosymbols)[-PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 16:]
        pseudosymbols_to_consider = self.history.last_seen_pseudosymbols
        # Try every possible bit phase
        pseudosymbols = np.array(list([x.pseudosymbol for x in pseudosymbols_to_consider]))
        bit_phase_to_confidence_score = dict()
        for possible_bit_phase in range(0, PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
            #_logger.info(f'try bit phase {possible_bit_phase}')
            pseudosymbols_aligned_with_bit_phase = np.roll(pseudosymbols, possible_bit_phase)
            # Compute a confidence score
            confidence = self._compute_bit_confidence_score(pseudosymbols_aligned_with_bit_phase)   # type: ignore
            bit_phase_to_confidence_score[possible_bit_phase] = confidence

        best_bit_phase = max(
            bit_phase_to_confidence_score, key=bit_phase_to_confidence_score.get    # type: ignore
        )
        highest_confidence_score = bit_phase_to_confidence_score[best_bit_phase]
        if False:
            _logger.info(
                f"Highest confidence bit phase offset: {best_bit_phase}. "
                f"Score: {highest_confidence_score * 100}"
            )
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.plot([x.as_val() for x in pseudosymbols])
            plt.show()

        #if highest_confidence_as_percentage >= 0.80:
        #    return highest_confidence_phase_offset % PSEUDOSYMBOLS_PER_NAVIGATION_BIT
        #return None
        #best_bit_phase
        return best_bit_phase

    def process_pseudosymbol(self, receiver_timestamp: ReceiverTimestampSeconds, pseudosymbol: NavigationBitPseudosymbol) -> list[Event]:
        events: list[Event] = []
        emitted_pseudosymbol = EmittedPseudosymbol(
            receiver_timestamp=receiver_timestamp,
            pseudosymbol=pseudosymbol,
        )
        self.queued_pseudosymbols.append(emitted_pseudosymbol)
        self.history.last_seen_pseudosymbols.append(emitted_pseudosymbol)

        self._processed_pseudosymbol_count += 1
        if not self.history.is_bit_phase_locked:
            # We're still trying to figure out a bit phase...
            # Is this a second boundary? (TODO(PT): This was just to cut down how much work we're doing to determine the bit
            # phase, but maybe we can drop it since we have a lock state?)
            if self._processed_pseudosymbol_count % PSEUDOSYMBOLS_PER_NAVIGATION_BIT == 0:
                previous_bit_phase_decision = self.history.previous_bit_phase_decision
                new_bit_phase_decision = self._redetermine_bit_phase()
                self.history.previous_bit_phase_decision = new_bit_phase_decision
                if new_bit_phase_decision is not None and previous_bit_phase_decision == new_bit_phase_decision:
                    # Found a consecutive agreement between two bit phase decisions
                    self.history.consecutive_agreeing_bit_phase_decisions += 1
                    # Have we seen enough consecutive agreements to decide our bit phase is probably correct?
                    # TODO(PT): Pull this out into a constant?
                    if self.history.consecutive_agreeing_bit_phase_decisions >= BITS_PER_SECOND * 4:
                        # TODO(PT): Is it important to emit the event again, or can we drop it?
                        self.history.is_bit_phase_locked = True
                        self.history.determined_bit_phase = new_bit_phase_decision
                        # Trim partial pseudosymbols
                        self.queued_pseudosymbols = self.queued_pseudosymbols[new_bit_phase_decision:]
                else:
                    self.history.consecutive_agreeing_bit_phase_decisions = 0

        # We may have just determined the bit phase above, so check now
        if self.history.determined_bit_phase is not None:
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
                    if confidence_score <= 50:
                        # TODO(PT): Could it be because the symbol phase has shifted?
                        bit_value = BitValue.UNKNOWN
                        self.sequential_unknown_bit_value_counter += 1
                        if self.sequential_unknown_bit_value_counter >= 30:
                            # TODO(PT): This might cause issues because it throws our subframe phase out of alignment?
                            # We might need to emit an event to tell the subframe decoder to select a new phase now
                            _logger.info(f'Resetting bit phase because we failed to resolve too many bits in a row...')
                            self._reset_selected_bit_phase()
                    else:
                        self.sequential_unknown_bit_value_counter = 0

                    # The timestamp of the bit comes from the receiver timestamp of
                    # the first pseudosymbol in the bit.
                    timestamp = bit_pseudosymbols[0].receiver_timestamp
                    events.append(EmitNavigationBitEvent(receiver_timestamp=timestamp, bit_value=bit_value))
                    self.history.last_emitted_bits.append(bit_value)

                    # TODO(PT): It looks as though our Doppler shift isn't following nearly fast enough (30Hz off on a re-acquire)
                    # Should check the timestamps on this to get a feel for how fast it drops over time?

                    # TODO(PT): Come up with some condition to emit a LostBitCoherenceEvent (i.e. X% unknown bits
                    # emitted in the last Y seconds).
                    #else:
                    #    events.append(LostBitCoherenceEvent(confidence_score))
                    #    # Stop consuming bits now
                    #    break
                else:
                    # Not enough pseudosymbols to emit a bit
                    break

        return events
