import collections
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Self

# Every word ends in 6 parity bits
_DATA_BIT_COUNT_PER_WORD = 24
_PARITY_BIT_COUNT_PER_WORD = 6


class GpsSubframeId(Enum):
    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()

    @classmethod
    def from_bits(cls, bits: list[int]) -> Self:
        return {
            (0, 0, 1): GpsSubframeId.ONE,
            (0, 1, 0): GpsSubframeId.TWO,
            (0, 1, 1): GpsSubframeId.THREE,
            (1, 0, 0): GpsSubframeId.FOUR,
            (1, 0, 1): GpsSubframeId.FIVE,
        }[*bits]


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
    subframe_id: GpsSubframeId
    to_be_solved: list[int]
    parity_bits: list[int]

    @property
    def time_of_week_in_seconds(self) -> int:
        time_of_week_accumulator = 0
        # Each bit represents (1.5*(2^(bit_position+2))) seconds
        # Start with the LSB
        for i, bit in enumerate(reversed(self.time_of_week)):
            bit_granularity = 1.5 * (2 ** (i + 2))
            if bit == 1:
                time_of_week_accumulator += bit_granularity
        return time_of_week_accumulator


@dataclass
class Subframe5:
    data_id: list[int]
    satellite_id: list[int]
    eccentricity: float
    time_of_ephemeris: float
    delta_inclination_angle:float
    right_ascension_rate: float
    sv_health: int
    semi_major_axis_sqrt: float
    longitude_of_ascension_mode: float
    argument_of_perigree: float
    mean_anomaly_at_reference_time: float
    a_f0: float
    a_f1: float


class NavigationMessageSubframeParser:
    def __init__(self, bits: list[int]) -> None:
        self.bits = bits
        self.cursor = 0
        self.word_bits = collections.deque(maxlen=_DATA_BIT_COUNT_PER_WORD)

    def peek_bit_count(self, n: int) -> list[int]:
        return self.bits[self.cursor : self.cursor + n]

    def get_bit_count(self, n: int) -> list[int]:
        out = self.peek_bit_count(n)
        self.cursor += n
        self.word_bits.extend(out)
        return out

    def get_bit(self) -> int:
        return self.get_bit_count(1)[0]

    def match_bits(self, expected_bits: list[int]) -> list[int]:
        actual_bits = self.get_bit_count(len(expected_bits))
        if actual_bits != expected_bits:
            raise ValueError(
                f'Expected to read {"".join([str(b) for b in expected_bits])}, '
                f'but read {"".join([str(b) for b in actual_bits])}'
            )
        return expected_bits

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
            subframe_id=GpsSubframeId.from_bits(subframe_id_codes),
            to_be_solved=to_be_solved,
            parity_bits=parity_bits,
        )

    def get_bit_string(self, bit_count: int) -> str:
        bits = self.get_bit_count(bit_count)
        # Convert array of bits to an integer
        bits_as_str = "".join([str(b) for b in bits])
        return bits_as_str

    def get_num(
        self,
        bit_count: int,
        scale_factor_exp2: int = 0,
        twos_complement: bool = False,
    ) -> float:
        bits_as_str = self.get_bit_string(bit_count)
        value = int(bits_as_str, 2)
        if twos_complement:
            def twos_comp(val, bits):
                """compute the 2's complement of int value val"""
                if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
                    val = val - (1 << bits)  # compute negative value
                return val
            value = twos_comp(value, len(bits_as_str))
            #and (.Params) (

            #{{ $style = cond (and (.Params) (isset .Params `adjust_y`)) (printf "transform: translate(0%, %s%)" .Get "adjust_y") ("") }}

        return value * (2 ** scale_factor_exp2)

    def validate_parity(self):
        parity_bits = self.get_bit_count(_PARITY_BIT_COUNT_PER_WORD)
        data_bits = list(self.word_bits)
        self.word_bits.clear()
        print(f'TODO: Validate parity bits {parity_bits} for {data_bits}')

    def parse_subframe_5(self) -> Subframe5:
        # Ref: IS-GPS-200L, 20.3.3.5 Subframes 4 and 5
        data_id = self.match_bits([0, 1])
        satellite_id = self.get_bit_count(5)

        eccentricity = self.get_num(bit_count=16, scale_factor_exp2=-21)
        self.validate_parity()

        time_of_ephemeris = self.get_num(bit_count=8, scale_factor_exp2=12)
        delta_inclination_angle = self.get_num(bit_count=16, scale_factor_exp2=2, twos_complement=True)
        self.validate_parity()

        right_ascension_rate = self.get_num(bit_count=16, scale_factor_exp2=-38, twos_complement=True)
        sv_health = self.get_num(bit_count=8)
        self.validate_parity()

        semi_major_axis_sqrt = self.get_num(bit_count=24, scale_factor_exp2=-11, twos_complement=True)
        self.validate_parity()

        longitude_of_ascension_mode = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)
        self.validate_parity()

        argument_of_perigree = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)
        self.validate_parity()

        mean_anomaly_at_reference_time = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)
        self.validate_parity()

        #a_f0_high = self.get_num(bit_count=8, scale_factor_exp2=-20)
        a_f0_high = self.get_num(bit_count=8)
        a_f1 = self.get_num(bit_count=11, scale_factor_exp2=-38)
        #a_f0_low = self.get_num(bit_count=3, scale_factor_exp2=-20)
        a_f0_low = self.get_num(bit_count=3)
        t = self.get_num(bit_count=2)
        self.validate_parity()

        print(f'a_f0_high {a_f0_high} a_f0_low {a_f0_low}')
        a_f0 = ((int(a_f0_high) << 3) | int(a_f0_low)) * (2 ** -20)

        return Subframe5(
            data_id=data_id,
            satellite_id=satellite_id,
            eccentricity=eccentricity,
            time_of_ephemeris=time_of_ephemeris,
            delta_inclination_angle=delta_inclination_angle,
            right_ascension_rate=right_ascension_rate,
            sv_health=sv_health,
            semi_major_axis_sqrt=semi_major_axis_sqrt,
            longitude_of_ascension_mode=longitude_of_ascension_mode,
            argument_of_perigree=argument_of_perigree,
            mean_anomaly_at_reference_time=mean_anomaly_at_reference_time,
            a_f0=a_f0,
            a_f1=a_f1,
        )
