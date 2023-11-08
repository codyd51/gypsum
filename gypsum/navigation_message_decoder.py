import logging

from gypsum.tracker import BitValue


_logger = logging.getLogger(__name__)


class NavigationMessageDecoder:
    def __init__(self):
        self.queued_bits = []

    def process_bit_from_satellite(self, bit: BitValue) -> None:
        self.queued_bits.append(bit)
        _logger.info(f'Queued bits: {"".join([str(b.as_val()) for b in self.queued_bits])}')
