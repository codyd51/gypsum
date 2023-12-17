import logging
from dataclasses import dataclass
from enum import Enum, auto

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.events import Event
from gypsum.navigation_bit_intergrator import EmitNavigationBitEvent
from gypsum.navigation_message_parser import (
    GpsSubframeId,
    HandoverWord,
    IncorrectPreludeBitsError,
    InvalidSubframeIdError,
    NavigationMessageSubframe,
    NavigationMessageSubframeParser,
    TelemetryWord,
)
from gypsum.tracker import BitValue
from gypsum.utils import get_indexes_of_sublist

_logger = logging.getLogger(__name__)


BITS_PER_SUBFRAME = 300
# TELEMETRY_WORD_PREAMBLE = [1, 0, 0, 0, 1, 0, 1, 1]
TELEMETRY_WORD_PREAMBLE = [
    BitValue.ONE,
    BitValue.ZERO,
    BitValue.ZERO,
    BitValue.ZERO,
    BitValue.ONE,
    BitValue.ZERO,
    BitValue.ONE,
    BitValue.ONE,
]


class BitPolarity(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()

    def inverse(self) -> "BitPolarity":
        return {
            self.POSITIVE: self.NEGATIVE,
            self.NEGATIVE: self.POSITIVE,
        }[self]


class CannotDetermineSubframePhaseEvent(Event):
    pass


class DeterminedSubframePhaseEvent(Event):
    def __init__(self, subframe_phase: int, polarity: BitPolarity) -> None:
        self.subframe_phase = subframe_phase
        self.polarity = polarity


class EmitSubframeEvent(Event):
    def __init__(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        telemetry_word: TelemetryWord,
        handover_word: HandoverWord,
        subframe: NavigationMessageSubframe,
    ) -> None:
        self.receiver_timestamp = receiver_timestamp
        self.telemetry_word = telemetry_word
        self.handover_word = handover_word
        self.subframe = subframe


@dataclass
class NavigationMessageDecoderHistory:
    """TODO(PT): "State" might be a better word, but might imply we should move *all* state here, which would be odd"""

    # PT: Looking at it, perhaps we *should* move all state here
    determined_subframe_phase: int | None = None
    emitted_subframe_count: int = 0


class NavigationMessageDecoder:
    def __init__(self) -> None:
        self.queued_bit_events: list[EmitNavigationBitEvent] = []
        self.history = NavigationMessageDecoderHistory()
        self.determined_polarity: BitPolarity | None = None

    def _identify_preamble_in_queued_bits(
        self,
        preamble: list[BitValue],
    ) -> int | None:
        queued_bits = [e.bit_value for e in self.queued_bit_events]
        # print(queued_bits)
        preamble_candidates = get_indexes_of_sublist(queued_bits, preamble)
        # We need at least two preambles
        if len(preamble_candidates) < 2:
            return None

        # There could be other copies of the preamble that are not actually the preamble, but are instead
        # coincidence.
        # We'll need to look at our candidates and see if there's another preamble 300 bits away. If we do, it's
        # very likely it's the real preamble.
        #
        # We can't look after the last candidate, so stop just before it
        for candidate in preamble_candidates[:-1]:
            next_preamble_after_this_candidate = candidate + BITS_PER_SUBFRAME
            # Did we also see this next preamble?
            if next_preamble_after_this_candidate in preamble_candidates:
                # We've found two preambles 300 bits apart. Consider this a valid detection, and stop looking.
                # Our first subframe starts at this candidate
                return candidate

        # Failed to find any preambles
        # TODO(PT): Maybe we could just keep waiting for a better lock?
        return None

    def _reset_selected_subframe_phase(self):
        _logger.info(f"Resetting selected subframe phase...")
        # TODO(PT): Add an @property setter so we can just say determined_subframe_phase?
        self.history.determined_subframe_phase = None
        self.determined_polarity = None
        # self.queued_bit_events = []

    def _determine_subframe_phase_from_queued_bits(self) -> list[Event]:
        # We'll need at least two preambles to identify a subframe phase
        if len(self.queued_bit_events) < BITS_PER_SUBFRAME * 2:
            return []

        events = []
        # Depending on the phase of our PRN correlations, our bits could appear either 'upright' or 'inverted'.
        # To determine which, we'll need to search for the preamble both as 'upright' and 'inverted'. Whichever
        # version produces a match will tell us the polarity of our bits.
        #
        # Search our bits for the subframe preamble
        preamble_and_polarity = [
            (TELEMETRY_WORD_PREAMBLE, BitPolarity.POSITIVE),
            ([b.inverted() for b in TELEMETRY_WORD_PREAMBLE], BitPolarity.NEGATIVE),
        ]
        for preamble, polarity in preamble_and_polarity:
            first_identified_preamble_index = self._identify_preamble_in_queued_bits(preamble)
            if first_identified_preamble_index is not None:
                events.append(DeterminedSubframePhaseEvent(first_identified_preamble_index, polarity))
                self.history.determined_subframe_phase = first_identified_preamble_index
                self.determined_polarity = polarity
                # Discard queued bits from the first partial subframe
                bit_count_outside_first_subframe_to_discard = self.history.determined_subframe_phase % BITS_PER_SUBFRAME
                self.queued_bit_events = self.queued_bit_events[bit_count_outside_first_subframe_to_discard:]
                _logger.info(
                    f"Identified preamble at bit phase {self.history.determined_subframe_phase} "
                    f"when probing with bit polarity: {polarity.name}"
                )
                break
        else:
            # Didn't find the preamble phase in either polarity
            # Keep waiting for more bits to come in, up to a maximum allowance.
            if len(self.queued_bit_events) < BITS_PER_SUBFRAME * 12:
                # Continue to allow the tracker to operate, although we haven't been able to identify a subframe
                # phase so far.
                # (This could be because the tracker is trying to recover lock
                # and currently has some messy bits in the data)
                pass
            else:
                unknown_bit_indexes = [
                    i for i, b in enumerate(self.queued_bit_events) if b.bit_value == BitValue.UNKNOWN
                ]
                unknown_count = len(unknown_bit_indexes)
                _logger.info(
                    f"Failed to identify subframe phase in a generous tracking span. "
                    f"({len(self.queued_bit_events)} bits ({unknown_count} unknown, unknown bit indexes: {unknown_bit_indexes})"
                )
                events.append(CannotDetermineSubframePhaseEvent())
        return events

    def process_bit_from_satellite(self, bit_event: EmitNavigationBitEvent) -> list[Event]:
        events: list[Event] = []
        self.queued_bit_events.append(bit_event)

        # Try to identify subframe phase once we have enough bits to see a few subframes
        if self.history.determined_subframe_phase is None:
            events.extend(self._determine_subframe_phase_from_queued_bits())

        # We may have just determined the subframe phase above, so check again
        if self.history.determined_subframe_phase is not None:
            # Drain the bit queue as much as we can
            while True:
                if len(self.queued_bit_events) >= BITS_PER_SUBFRAME:
                    try:
                        maybe_subframe = self.parse_subframe()
                        if maybe_subframe:
                            events.append(maybe_subframe)
                            self.history.emitted_subframe_count += 1
                    except Exception as e:
                        # TODO(PT): This will probably break everything because we stop parsing in the middle of a frame...
                        raise
                        # print(f'*** swallowing exception {e}')
                        # continue
                else:
                    break

        return events

    def parse_subframe(self) -> EmitSubframeEvent | None:
        subframe_bits = self.queued_bit_events[:BITS_PER_SUBFRAME]
        subframe_receiver_timestamp = subframe_bits[0].receiver_timestamp
        _logger.info(f"Emitting subframe timestamped at receiver at {subframe_receiver_timestamp}")
        # Consume these bits by removing them from the queue
        self.queued_bit_events = self.queued_bit_events[BITS_PER_SUBFRAME:]

        # First, discard the subframe entirely if any bits couldn't be resolved
        # print(subframe_bits)

        has_failed_bits = any(bit.bit_value == BitValue.UNKNOWN for bit in subframe_bits)
        if has_failed_bits:
            failed_bit_count = len([b for b in subframe_bits if b.bit_value == BitValue.UNKNOWN])
            # TODO(PT): It'd be pretty neat to be able to parse a subframe that contained unknown bits. Perhaps just
            # a few fields in the subframe could be unreadable due to bad bits, while the rest of the subframe can be
            # kept.
            _logger.info(
                f"Skipping subframe that has at least one unknown bit: {failed_bit_count} / {len(subframe_bits)} bits failed"
            )
            # We will need to recalculate our polarity.
            # Currently this also involves re-determining the subframe phase, but doesn't need to
            # This is because an unknown bit corresponds to a slip - we're no longer tracking the cycle exactly and there may have been an inversion due to the slip
            self._reset_selected_subframe_phase()

            if failed_bit_count > 150:
                import gypsum.utils

                gypsum.utils.DEBUG = True
            return None

        # Flip the bit polarity so everything looks upright
        preprocessed_bits = [b.bit_value for b in subframe_bits]
        if self.determined_polarity == BitPolarity.NEGATIVE:
            preprocessed_bits = [b.inverted() for b in preprocessed_bits]
        bits_as_ints = [b.as_val() for b in preprocessed_bits]
        subframe_parser = NavigationMessageSubframeParser(bits_as_ints)
        try:
            telemetry_word = subframe_parser.parse_telemetry_word()
        except IncorrectPreludeBitsError:
            _logger.info("Resetting phase because the prelude was incorrect")
            self._reset_selected_subframe_phase()
            return None

        try:
            handover_word = subframe_parser.parse_handover_word()
        except InvalidSubframeIdError:
            _logger.info("Resetting phase because the subframe ID was invalid")
            self._reset_selected_subframe_phase()
            return None

        _logger.info(f"Handover word time of week: {handover_word.time_of_week_in_seconds}")

        subframe_id = handover_word.subframe_id
        subframe: NavigationMessageSubframe
        if subframe_id == GpsSubframeId.ONE:
            subframe = subframe_parser.parse_subframe_1()
        elif subframe_id == GpsSubframeId.TWO:
            subframe = subframe_parser.parse_subframe_2()
        elif subframe_id == GpsSubframeId.THREE:
            subframe = subframe_parser.parse_subframe_3()
        elif subframe_id == GpsSubframeId.FOUR:
            subframe = subframe_parser.parse_subframe_4()
        elif subframe_id == GpsSubframeId.FIVE:
            subframe = subframe_parser.parse_subframe_5()
        else:
            raise NotImplementedError(subframe_id)

        return EmitSubframeEvent(
            receiver_timestamp=subframe_receiver_timestamp,
            telemetry_word=telemetry_word,
            handover_word=handover_word,
            subframe=subframe,
        )
