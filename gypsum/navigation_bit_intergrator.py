from dataclasses import dataclass

from gypsum.satellite import GpsSatellite
from gypsum.tracker import BitValue
from gypsum.tracker import NavigationBitPseudosymbol


@dataclass
class GpsSatellitePseudosymbolIntegratorState:
    satellite: GpsSatellite
    bit_phase: int | None

    @property
    def is_satellite_lock_provisional(self) -> bool:
        # If we haven't chosen a bit phase, this satellite is still in the provisional tracking phase
        return self.bit_phase is None

    # finalize


class NavigationBitIntegrator:
    def __init__(self):
        self.satellites_to_integrator_states: dict[GpsSatellite, GpsSatellitePseudosymbolIntegratorState] = {}

    def process_pseudosymbol_from_satellite(self, satellite: GpsSatellite, pseudosymbol: NavigationBitPseudosymbol) -> BitValue | None:
        if satellite not in self.satellites_to_integrator_states:
            pass
        pass
