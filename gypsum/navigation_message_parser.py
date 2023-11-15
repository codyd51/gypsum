import collections
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

from gypsum.config import GPS_EPOCH_BASE_WEEK_NUMBER

# Every word ends in 6 parity bits
_DATA_BIT_COUNT_PER_WORD = 24
_PARITY_BIT_COUNT_PER_WORD = 6


def _get_twos_complement(num: int, bit_count: int) -> int:
    # Check whether the high sign bit is set
    if (num & (1 << (bit_count - 1))) == 0:
        # Positive, we can return it as-is
        return num
    # Negate to return the negative representation
    return num - (1 << bit_count)


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
class NavigationMessageSubframe:
    @property
    def subframe_id(self) -> GpsSubframeId:
        raise NotImplementedError("Must be provided by subclasses")


@dataclass
class NavigationMessageSubframe1(NavigationMessageSubframe):
    week_num: list[int]
    ca_or_p_on_l2: list[int]
    ura_index: list[int]
    sv_health: list[int]
    iodc: int
    l2_p_data_flag: int
    estimated_group_delay_differential: float
    t_oc: float
    a_f2: float
    a_f1: float
    a_f0: float

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.ONE


@dataclass
class NavigationMessageSubframe2(NavigationMessageSubframe):
    issue_of_data_ephemeris: list[int]
    correction_to_radius_sin: float
    mean_motion_difference_from_computed_value: float
    mean_anomaly_at_reference_time: float
    correction_to_latitude_cos: float
    eccentricity: float
    correction_to_latitude_sin: float
    sqrt_semi_major_axis: float
    reference_time_ephemeris: float
    fit_interval_flag: bool
    age_of_data_offset: list[int]

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.TWO


@dataclass
class Subframe5(NavigationMessageSubframe):
    data_id: list[int]
    satellite_id: list[int]
    eccentricity: float
    time_of_ephemeris: float
    delta_inclination_angle: float
    right_ascension_rate: float
    sv_health: list[int]
    semi_major_axis_sqrt: float
    longitude_of_ascension_mode: float
    argument_of_perigree: float
    mean_anomaly_at_reference_time: float
    a_f0: float
    a_f1: float

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.FIVE


class NavigationMessageSubframeParser:
    def __init__(self, bits: list[int]) -> None:
        self.bits = bits
        self.cursor = 0
        self.word_bits = collections.deque(maxlen=_DATA_BIT_COUNT_PER_WORD)

    def peek_bit_count(self, n: int) -> list[int]:
        return self.bits[self.cursor : self.cursor + n]

    def get_bits(self, n: int) -> list[int]:
        out = self.peek_bit_count(n)
        self.cursor += n
        self.word_bits.extend(out)
        return out

    def get_bit(self) -> int:
        return self.get_bits(1)[0]

    def match_bits(self, expected_bits: list[int]) -> list[int]:
        actual_bits = self.get_bits(len(expected_bits))
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
        telemetry_message = self.get_bits(14)
        integrity_status_flag = self.get_bit()
        spare_bit = self.get_bit()
        parity_bits = self.get_bits(6)
        return TelemetryWord(
            telemetry_message=telemetry_message,
            integrity_status_flag=integrity_status_flag,
            spare_bit=spare_bit,
            parity_bits=parity_bits,
        )

    def parse_handover_word(self) -> HandoverWord:
        time_of_week = self.get_bits(17)
        alert_flag = self.get_bit()
        anti_spoof_flag = self.get_bit()
        subframe_id_codes = self.get_bits(3)
        to_be_solved = self.get_bits(2)
        parity_bits = self.get_bits(6)
        return HandoverWord(
            time_of_week=time_of_week,
            alert_flag=alert_flag,
            anti_spoof_flag=anti_spoof_flag,
            subframe_id=GpsSubframeId.from_bits(subframe_id_codes),
            to_be_solved=to_be_solved,
            parity_bits=parity_bits,
        )

    @staticmethod
    def get_bit_string_from_bits(bits: list[int]) -> str:
        # Convert array of bits to an integer
        return "".join([str(b) for b in bits])

    def get_bit_string(self, bit_count: int) -> str:
        return self.get_bit_string_from_bits(self.get_bits(bit_count))

    def get_num_from_bits(
        self,
        # Force arguments to be specified as kwargs, for readability
        *,
        bits: list[int],
        scale_factor_exp2: int,
        twos_complement: bool,
    ) -> float:
        bits_as_str = self.get_bit_string_from_bits(bits)
        value = int(bits_as_str, 2)
        if twos_complement:
            value = _get_twos_complement(value, len(bits_as_str))
        return value * (2**scale_factor_exp2)

    def get_num(
        self,
        # Force arguments to be specified as kwargs, for readability
        *,
        bit_count: int,
        scale_factor_exp2: int,
        twos_complement: bool,
    ) -> float:
        return self.get_num_from_bits(
            bits=self.get_bits(bit_count),
            scale_factor_exp2=scale_factor_exp2,
            twos_complement=twos_complement,
        )

    def get_unscaled_num(
        self,
        bit_count: int,
        *,
        # Force arguments to be specified as kwargs, for readability
        twos_complement: bool,
    ) -> int:
        return int(
            self.get_num(
                bit_count=bit_count,
                scale_factor_exp2=0,
                twos_complement=twos_complement,
            )
        )

    def get_unscaled_num_from_bits(
        self,
        bits: list[int],
        *,
        # Force arguments to be specified as kwargs, for readability
        twos_complement: bool,
    ) -> int:
        return int(
            self.get_num_from_bits(
                bits=bits,
                scale_factor_exp2=0,
                twos_complement=twos_complement,
            )
        )

    def validate_parity(self):
        parity_bits = self.get_bits(_PARITY_BIT_COUNT_PER_WORD)
        # TODO(PT): Verify we have exactly 24 bits in the buffer?
        data_bits = list(self.word_bits)
        self.word_bits.clear()
        #print(f"TODO: Validate parity bits {parity_bits} for {data_bits}")

    def parse_subframe_1(self) -> NavigationMessageSubframe1:
        # Ref: IS-GPS-200L, 20.3.3.5 Subframes 1, Figure 20-1. Data Format (sheet 1 of 11)
        # Word 3
        # PT: This field stores the week number, mod 1024 weeks. See the comment on GPS_EPOCH_BASE_WEEK_NUMBER.
        # **This means that this field rolls over to zero every 19.6 years**.
        # See the comment on GPS_EPOCH_BASE_WEEK_NUMBER.
        week_num_mod_1024 = self.get_unscaled_num(10, twos_complement=False)
        week_num = week_num_mod_1024 + GPS_EPOCH_BASE_WEEK_NUMBER
        ca_or_p_on_l2 = self.get_bits(2)
        ura_index = self.get_bits(4)
        sv_health = self.get_bits(6)
        iodc_high = self.get_bits(2)
        self.validate_parity()

        # Word 4
        l2_p_data_flag = self.get_bit()
        _reserved_block1 = self.get_bits(23)
        self.validate_parity()

        # Word 5
        _reserved_block2 = self.get_bits(24)
        self.validate_parity()

        # Word 6
        _reserved_block3 = self.get_bits(24)
        self.validate_parity()

        # Word 7
        _reserved_block4 = self.get_bits(16)
        estimated_group_delay_differential = self.get_num(bit_count=8, scale_factor_exp2=-31, twos_complement=True)
        self.validate_parity()

        # Word 8
        iodc_low = self.get_bits(8)
        t_oc = self.get_num(bit_count=16, scale_factor_exp2=4, twos_complement=False)
        self.validate_parity()

        # Word 9
        a_f2 = self.get_num(bit_count=8, scale_factor_exp2=-55, twos_complement=True)
        a_f1 = self.get_num(bit_count=16, scale_factor_exp2=-43, twos_complement=True)
        self.validate_parity()

        # Word 10
        a_f0 = self.get_num(bit_count=22, scale_factor_exp2=-31, twos_complement=True)
        _to_be_solved = self.get_bits(2)
        self.validate_parity()

        iodc = self.get_unscaled_num_from_bits(
            bits=[*iodc_high, *iodc_low],
            twos_complement=False,
        )

        return NavigationMessageSubframe1(
            week_num=week_num,
            ca_or_p_on_l2=ca_or_p_on_l2,
            ura_index=ura_index,
            sv_health=sv_health,
            iodc=iodc,
            l2_p_data_flag=l2_p_data_flag,
            estimated_group_delay_differential=estimated_group_delay_differential,
            t_oc=t_oc,
            a_f2=a_f2,
            a_f1=a_f1,
            a_f0=a_f0,
        )

    def parse_subframe_2(self) -> NavigationMessageSubframe2:
        # Ref: IS-GPS-200L, Figure 20-1. Data Format (sheet 2 of 11)
        # Ref: IS-GPS-200L, 20.3.3.4.1 Content of Subframes 2 and 3
        #
        # Word 3
        issue_of_data_ephemeris = self.get_bits(8)
        correction_to_radius_sin = self.get_num(bit_count=16, scale_factor_exp2=-5, twos_complement=True)
        self.validate_parity()

        # Word 4
        mean_motion_difference_from_computed_value = self.get_num(bit_count=16, scale_factor_exp2=-43, twos_complement=True)
        mean_anomaly_at_reference_time_high = self.get_bits(8)
        self.validate_parity()

        # Word 5
        mean_anomaly_at_reference_time_low = self.get_bits(24)
        mean_anomaly_at_reference_time_bits = [*mean_anomaly_at_reference_time_high, *mean_anomaly_at_reference_time_low]
        mean_anomaly_at_reference_time = self.get_num_from_bits(
            bits=mean_anomaly_at_reference_time_bits,
            scale_factor_exp2=-31,
            twos_complement=True
        )
        self.validate_parity()

        # Word 6
        correction_to_latitude_cos = self.get_num(bit_count=16, scale_factor_exp2=-29, twos_complement=True)
        eccentricity_high = self.get_bits(8)
        self.validate_parity()

        # Word 7
        eccentricity_low = self.get_bits(24)
        eccentricity_bits = [*eccentricity_high, *eccentricity_low]
        eccentricity = self.get_num_from_bits(
            bits=eccentricity_bits,
            scale_factor_exp2=-33,
            twos_complement=False
        )
        self.validate_parity()

        # Word 8
        correction_to_latitude_sin = self.get_num(bit_count=16, scale_factor_exp2=-29, twos_complement=True)
        sqrt_semi_major_axis_high = self.get_bits(8)
        self.validate_parity()

        sqrt_semi_major_axis_low = self.get_bits(24)
        sqrt_semi_major_axis_bits = [*sqrt_semi_major_axis_high, *sqrt_semi_major_axis_low]
        sqrt_semi_major_axis = self.get_num_from_bits(
            bits=sqrt_semi_major_axis_bits,
            scale_factor_exp2=-19,
            twos_complement=False
        )
        self.validate_parity()

        reference_time_ephemeris = self.get_num(bit_count=16, scale_factor_exp2=4, twos_complement=False)
        fit_interval_flag = self.get_bit()
        age_of_data_offset = self.get_bits(5)
        _to_be_solved = self.get_bits(2)
        self.validate_parity()

        # TODO(PT): Add units to each of these
        return NavigationMessageSubframe2(
            issue_of_data_ephemeris=issue_of_data_ephemeris,
            correction_to_radius_sin=correction_to_radius_sin,
            mean_motion_difference_from_computed_value=mean_motion_difference_from_computed_value,
            mean_anomaly_at_reference_time=mean_anomaly_at_reference_time,
            correction_to_latitude_cos=correction_to_latitude_cos,
            eccentricity=eccentricity,
            correction_to_latitude_sin=correction_to_latitude_sin,
            sqrt_semi_major_axis=sqrt_semi_major_axis,
            reference_time_ephemeris=reference_time_ephemeris,
            fit_interval_flag=bool(fit_interval_flag),
            age_of_data_offset=age_of_data_offset,
        )

    def parse_subframe_5(self) -> Subframe5:
        # Ref: IS-GPS-200L, 20.3.3.5 Subframes 4 and 5
        # TODO(PT): The below is only valid for pages 1 through 24!
        # TODO(PT): Validate page ID?!

        # Word 3
        data_id = self.match_bits([0, 1])
        satellite_id = self.get_bits(6)
        eccentricity = self.get_num(bit_count=16, scale_factor_exp2=-21, twos_complement=False)
        self.validate_parity()

        # Word 4
        time_of_ephemeris = self.get_num(bit_count=8, scale_factor_exp2=12, twos_complement=False)
        delta_inclination_angle = self.get_num(bit_count=16, scale_factor_exp2=-19, twos_complement=True)
        self.validate_parity()

        # Word 5
        right_ascension_rate = self.get_num(bit_count=16, scale_factor_exp2=-38, twos_complement=True)
        sv_health = self.get_bits(8)
        self.validate_parity()

        # Word 6
        semi_major_axis_sqrt = self.get_num(bit_count=24, scale_factor_exp2=-11, twos_complement=False)
        self.validate_parity()

        # Word 7
        longitude_of_ascension_mode = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)
        self.validate_parity()

        # Word 8
        argument_of_perigree = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)
        self.validate_parity()

        # Word 9
        mean_anomaly_at_reference_time = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)
        self.validate_parity()

        # Word 10
        a_f0_high = self.get_bits(8)
        a_f1 = self.get_num(bit_count=11, scale_factor_exp2=-38, twos_complement=True)
        a_f0_low = self.get_bits(3)
        a_f0_bits = [*a_f0_high, *a_f0_low]
        a_f0 = self.get_num_from_bits(bits=a_f0_bits, scale_factor_exp2=-20, twos_complement=True)

        _to_be_solved = self.get_bits(2)
        self.validate_parity()

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
