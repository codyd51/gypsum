from dataclasses import dataclass


@dataclass
class TelemetryWord:
    # TODO(PT): Introduce a dedicated type? Ints are nicer than bools because we'll probably do bit manipulation?
    telemetry_message: list[int]
    integrity_status_flag: int
    spare_bit: int
    parity_bits: list[int]


@dataclass
class HandoverWord:
    time_of_week: list[int]
    alert_flag: int
    anti_spoof_flag: int
    subframe_id_codes: list[int]
    to_be_solved: list[int]
    parity_bits: list[int]


class NavigationMessageSubframeParser:
    def __init__(self, bits: list[int]) -> None:
        self.bits = bits
        self.cursor = 0

    def peek_bit_count(self, n: int) -> list[int]:
        return self.bits[self.cursor : self.cursor + n]

    def get_bit_count(self, n: int) -> list[int]:
        out = self.peek_bit_count(n)
        self.cursor += n
        return out

    def get_bit(self) -> int:
        return self.get_bit_count(1)[0]

    def match_bits(self, expected_bits: list[int]) -> None:
        actual_bits = self.get_bit_count(len(expected_bits))
        if actual_bits != expected_bits:
            raise ValueError(
                f'Expected to read {"".join([str(b) for b in expected_bits])}, '
                f'but read {"".join([str(b) for b in actual_bits])}'
            )

    def parse_telemetry_word(self) -> TelemetryWord:
        # Ref: IS-GPS-200L, Figure 20-2
        tlm_prelude = [1, 0, 0, 0, 1, 0, 1, 1]
        self.match_bits(tlm_prelude)
        telemetry_message = self.get_bit_count(14)
        integrity_status_flag = self.get_bit()
        spare_bit = self.get_bit()
        parity_bits = self.get_bit_count(6)
        return TelemetryWord(
            telemetry_message=telemetry_message,
            integrity_status_flag=integrity_status_flag,
            spare_bit=spare_bit,
            parity_bits=parity_bits,
        )

    def parse_handover_word(self) -> HandoverWord:
        time_of_week = self.get_bit_count(17)
        alert_flag = self.get_bit()
        anti_spoof_flag = self.get_bit()
        subframe_id_codes = self.get_bit_count(3)
        to_be_solved = self.get_bit_count(2)
        parity_bits = self.get_bit_count(6)
        return HandoverWord(
            time_of_week=time_of_week,
            alert_flag=alert_flag,
            anti_spoof_flag=anti_spoof_flag,
            subframe_id_codes=subframe_id_codes,
            to_be_solved=to_be_solved,
            parity_bits=parity_bits,
        )
