import logging
from enum import Enum
from enum import auto

from gypsum.events import Event
from gypsum.navigation_message_parser import HandoverWord
from gypsum.navigation_message_parser import NavigationMessageSubframeParser
from gypsum.navigation_message_parser import TelemetryWord
from gypsum.tracker import BitValue
from gypsum.utils import get_indexes_of_sublist

_logger = logging.getLogger(__name__)


BITS_PER_SUBFRAME = 300
#TELEMETRY_WORD_PREAMBLE = [1, 0, 0, 0, 1, 0, 1, 1]
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


class CannotDetermineSubframePhaseEvent(Event):
    pass


class DeterminedSubframePhaseEvent(Event):
    def __init__(self, subframe_phase: int, polarity: BitPolarity) -> None:
        self.subframe_phase = subframe_phase
        self.polarity = polarity


class EmitSubframeEvent(Event):
    def __init__(
        self,
        telemetry_word: TelemetryWord,
        handover_word: HandoverWord,
    ) -> None:
        self.telemetry_word = telemetry_word
        self.handover_word = handover_word


class NavigationMessageDecoder:
    def __init__(self):
        self.queued_bits = []
        self.determined_subframe_phase: int | None = None
        self.determined_polarity: BitPolarity | None = None

    def _identify_preamble_in_queued_bits(
        self,
        preamble: list[BitValue],
    ) -> int | None:
        preamble_candidates = get_indexes_of_sublist(self.queued_bits, preamble)
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

    def process_bit_from_satellite(self, bit: BitValue) -> list[Event]:
        events = []
        self.queued_bits.append(bit)
        # _logger.info(f'Queued bits: {"".join([str(b.as_val()) for b in self.queued_bits])}')

        # Try to identify subframe phase once we have enough bits to see a few subframes
        if len(self.queued_bits) == BITS_PER_SUBFRAME * 4:
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
                print(f'try polarity {polarity}')
                first_identified_preamble_index = self._identify_preamble_in_queued_bits(preamble)
                if first_identified_preamble_index:
                    events.append(DeterminedSubframePhaseEvent(first_identified_preamble_index, polarity))
                    self.determined_subframe_phase = first_identified_preamble_index
                    self.determined_polarity = polarity
                    # Discard queued bits from the first partial subframe
                    self.queued_bits = self.queued_bits[self.determined_subframe_phase:]
                    print('found')
                    break
            else:
                # Didn't find the preamble phase in either polarity
                events.append(CannotDetermineSubframePhaseEvent())

        # We may have just determined the subframe phase above, so check again
        if self.determined_subframe_phase is not None:
            # Drain the bit queue as much as we can
            while True:
                if len(self.queued_bits) >= BITS_PER_SUBFRAME:
                    events.append(self.parse_subframe())
                else:
                    break

        return events

    def parse_subframe(self) -> EmitSubframeEvent:
        _logger.info(f'Emitting subframe')
        subframe_bits = self.queued_bits[:BITS_PER_SUBFRAME]
        # Consume these bits by removing them from the queue
        self.queued_bits = self.queued_bits[BITS_PER_SUBFRAME:]

        # Flip the bit polarity so everything looks upright
        preprocessed_bits = subframe_bits
        if self.determined_polarity == BitPolarity.NEGATIVE:
            preprocessed_bits = [b.inverted() for b in preprocessed_bits]
        bits_as_ints = [b.as_val() for b in preprocessed_bits]
        subframe_parser = NavigationMessageSubframeParser(bits_as_ints)
        telemetry_word = subframe_parser.parse_telemetry_word()
        handover_word = subframe_parser.parse_handover_word()
        _logger.info(f'Handover word time of week: {handover_word.time_of_week_in_seconds}')

        return EmitSubframeEvent(
            telemetry_word,
            handover_word
        )

