from gypsum.satellite import GpsSatellite
from gypsum.tracker import BitValue


class NavigationMessageDecoder:
    def __init__(self):
        self.queued_bits = []

    def process_bit_from_satellite(self, satellite: GpsSatellite, bit: BitValue) -> None:
        pass
