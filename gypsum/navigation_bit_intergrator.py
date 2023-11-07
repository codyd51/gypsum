import logging
from dataclasses import dataclass

import numpy as np

from gypsum.constants import PSEUDOSYMBOLS_PER_NAVIGATION_BIT
from gypsum.satellite import GpsSatellite
from gypsum.tracker import BitValue
from gypsum.tracker import NavigationBitPseudosymbol
from gypsum.utils import chunks

_logger = logging.getLogger(__name__)


@dataclass
class GpsSatellitePseudosymbolIntegratorState:
    satellite: GpsSatellite

    @property
    def is_satellite_lock_provisional(self) -> bool:
        # If we haven't chosen a bit phase, this satellite is still in the provisional tracking phase
        return self.bit_phase is None

    # finalize


@dataclass
class Event:
    pass


class EmitNavigationBitEvent(Event):
    def __init__(self, bit_value: BitValue) -> None:
        self.bit_value = bit_value


class DeterminedBitPhaseEvent(Event):
    def __init__(self, bit_phase: int) -> None:
        self.bit_phase = bit_phase


class NavigationBitIntegrator:
    def __init__(self):
        self.queued_pseudosymbols: list[NavigationBitPseudosymbol] = []
        self.determined_bit_phase: int | None = None

    def process_pseudosymbol(self, pseudosymbol: NavigationBitPseudosymbol) -> list[Event]:
        events = []
        self.queued_pseudosymbols.append(pseudosymbol)

        if self.determined_bit_phase is None:
            # Have we seen enough bits to determine the navigation bit phase?
            # (The bound here can probably be lowered if useful)
            # PT: What if we determine a bit phase too early, before we've locked, and it's wrong?
            # Give ourselves 6 seconds to lock
            if len(self.queued_pseudosymbols) == (PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 6):
                _logger.info(f'Pseudosymbol integrator has seen enough bits, selecting a bit phase...')
                # Look at the symbols from the last 2 seconds, as they'll probably be much closer to correct than the
                # initial symbols we see.
                # We'll need two seconds worth of symbols to select a phase, as we try different offsets up to 20 symbols
                # This could instead be 1 second of we rolled instead of sliding the start.
                symbols_considered_for_phase_selection = self.queued_pseudosymbols[PSEUDOSYMBOLS_PER_NAVIGATION_BIT * 4:]
                confidence_scores = []
                for phase_guess in range(0, PSEUDOSYMBOLS_PER_NAVIGATION_BIT):
                    phase_shifted_symbols = symbols_considered_for_phase_selection[phase_guess:phase_guess + PSEUDOSYMBOLS_PER_NAVIGATION_BIT]
                    confidence_scores.append(sum([s.as_val() for s in phase_shifted_symbols]))
                    # This could be sensitive to tracking errors in this particular second of processing...

                highest_confidence_phase_offset = int(np.argmax(confidence_scores))
                _logger.info(
                    f'Highest confidence phase offset: {highest_confidence_phase_offset}. '
                    f'Score: {confidence_scores[highest_confidence_phase_offset]}'
                )
                self.determined_bit_phase = highest_confidence_phase_offset
                # Discard queued symbols from the first partial symbol
                self.queued_pseudosymbols = self.queued_pseudosymbols[self.determined_bit_phase:]
                events.append(DeterminedBitPhaseEvent(highest_confidence_phase_offset))
            else:
                _logger.info(
                    f'Pseudosymbol integrator hasn\'t yet determined bit phase '
                    f'but has only processed {len(self.queued_pseudosymbols)} pseudosymbols'
                )

        # We may have just determined the bit phase above, so check again
        if self.determined_bit_phase is not None:
            # Drain the symbol queue as much as we can
            while True:
                if len(self.queued_pseudosymbols) >= PSEUDOSYMBOLS_PER_NAVIGATION_BIT:
                    #_logger.info(f'Emitting bit from pseudosymbol queue with {len(self.queued_pseudosymbols)} symbols')
                    bit_pseudosymbols = self.queued_pseudosymbols[:PSEUDOSYMBOLS_PER_NAVIGATION_BIT]
                    # Consume these pseudosymbols by removing them from the queue
                    self.queued_pseudosymbols = self.queued_pseudosymbols[PSEUDOSYMBOLS_PER_NAVIGATION_BIT:]
                    pseudosymbol_sum = sum([s.as_val() for s in bit_pseudosymbols])
                    if pseudosymbol_sum > 0:
                        bit_value = BitValue.ONE
                    else:
                        bit_value = BitValue.ZERO
                    events.append(EmitNavigationBitEvent(bit_value))
                else:
                    break

        return events
