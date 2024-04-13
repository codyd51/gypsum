import logging
from dataclasses import dataclass
from enum import Enum, auto

import math

from gypsum.config import GPS_EPOCH_BASE_WEEK_NUMBER
from gypsum.units import Seconds

Meters = float
SemiCirclesPerSecond = float
SemiCircles = float
Radians = float
SecondsPerSecond = float


_logger = logging.getLogger(__name__)

# Every word ends in 6 parity bits
_DATA_BIT_COUNT_PER_WORD = 24
_PARITY_BIT_COUNT_PER_WORD = 6
_BIT_COUNT_PER_WORD = _DATA_BIT_COUNT_PER_WORD + _PARITY_BIT_COUNT_PER_WORD

# The parity calculation uses two bits from the previous word
_BIT_COUNT_PER_PARITY_CHECK = 2 + _DATA_BIT_COUNT_PER_WORD


def _get_twos_complement(num: int, bit_count: int) -> int:
    # Check whether the high sign bit is set
    if (num & (1 << (bit_count - 1))) == 0:
        # Positive, we can return it as-is
        return num
    # Negate to return the negative representation
    return num - (1 << bit_count)


class IncorrectPreludeBitsError(Exception):
    pass


class InvalidSubframeIdError(Exception):
    pass


class GpsSubframeId(Enum):
    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()

    @classmethod
    def from_bits(cls, bits: list[int]) -> "GpsSubframeId":
        try:
            return {
                (0, 0, 1): GpsSubframeId.ONE,
                (0, 1, 0): GpsSubframeId.TWO,
                (0, 1, 1): GpsSubframeId.THREE,
                (1, 0, 0): GpsSubframeId.FOUR,
                (1, 0, 1): GpsSubframeId.FIVE,
            }[
                *bits
            ]  # type: ignore
        except KeyError:
            raise InvalidSubframeIdError()


@dataclass
class TelemetryWord:
    # TODO(PT): Introduce a dedicated type? Ints are nicer than bools because we'll probably do bit manipulation?
    telemetry_message: list[int]
    integrity_status_flag: int
    spare_bit: int


@dataclass
class HandoverWord:
    time_of_week: list[int]
    alert_flag: int
    anti_spoof_flag: int
    subframe_id: GpsSubframeId
    to_be_solved: list[int]

    @property
    def time_of_week_in_seconds(self) -> int:
        time_of_week_accumulator = 0
        # Each bit represents (1.5*(2^(bit_position+2))) seconds
        # Start with the LSB
        for i, bit in enumerate(reversed(self.time_of_week)):
            bit_granularity = 1.5 * (math.pow(2, i + 2))
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
    week_num_mod_1024_bits: int
    ca_or_p_on_l2: list[int]
    ura_index: list[int]
    sv_health: list[int]
    # Ref: 20.3.3.3.1.5 Issue of Data, Clock (IODC)
    issue_of_data_clock: list[int]
    l2_p_data_flag: int
    estimated_group_delay_differential: float
    t_oc: float
    a_f2: float
    a_f1: float
    a_f0: float

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.ONE

    @property
    def week_num(self) -> int:
        # PT: This field stores the week number, mod 1024 weeks. See the comment on GPS_EPOCH_BASE_WEEK_NUMBER.
        # **This means that this field rolls over to zero every 19.6 years**.
        # See the comment on GPS_EPOCH_BASE_WEEK_NUMBER.
        return self.week_num_mod_1024_bits + GPS_EPOCH_BASE_WEEK_NUMBER


@dataclass
class NavigationMessageSubframe2(NavigationMessageSubframe):
    issue_of_data_ephemeris: list[int]
    correction_to_orbital_radius_sin: Meters
    mean_motion_difference_from_computed_value: SemiCirclesPerSecond
    mean_anomaly_at_reference_time: SemiCircles
    correction_to_latitude_cos: Radians
    eccentricity: float
    correction_to_latitude_sin: Radians
    sqrt_semi_major_axis: Meters
    reference_time_ephemeris: Seconds
    fit_interval_flag: bool
    age_of_data_offset: list[int]

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.TWO


@dataclass
class NavigationMessageSubframe3(NavigationMessageSubframe):
    correction_to_inclination_angle_cos: Radians
    longitude_of_ascending_node: SemiCircles
    correction_to_inclination_angle_sin: Radians
    inclination_angle: SemiCircles
    correction_to_orbital_radius_cos: Meters
    argument_of_perigee: SemiCircles
    rate_of_right_ascension: SemiCirclesPerSecond
    rate_of_inclination_angle: SemiCirclesPerSecond
    issue_of_data_ephemeris: list[int]

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.THREE


@dataclass
class NavigationMessageSubframe4(NavigationMessageSubframe):
    data_id: int
    page_id: int

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.FOUR


@dataclass
class NavigationMessageSubframe5(NavigationMessageSubframe):
    data_id: list[int]
    satellite_id: list[int]
    eccentricity: float
    time_of_ephemeris: Seconds
    delta_inclination_angle: SemiCircles
    right_ascension_rate: SemiCirclesPerSecond
    sv_health: list[int]
    semi_major_axis_sqrt: Meters
    # TODO(PT): Typo, should be refactored to ascending_node?
    longitude_of_ascension_mode: SemiCircles
    argument_of_perigree: SemiCircles
    mean_anomaly_at_reference_time: SemiCircles
    a_f0: Seconds
    a_f1: SecondsPerSecond

    @property
    def subframe_id(self) -> GpsSubframeId:
        return GpsSubframeId.FIVE


class NavigationMessageSubframeParser:
    def __init__(self, bits: list[int]) -> None:
        self.bits = bits
        self.cursor = 0
        self.preprocessed_data_bits_of_current_word: list[int] = []
        # Every parity check relies on the last two bits from the previous word
        # For the first word, we prime the buffer with 00.
        self.last_two_parity_bits = [0, 0]

    def peek_bit_count(self, n: int, from_preprocessed_word_bits: bool = True) -> list[int]:
        # Transparently provide decoded bits, post-parity check, unless the caller really wants raw bits
        if from_preprocessed_word_bits:
            return self.preprocessed_data_bits_of_current_word[:n]
        return self.bits[self.cursor : self.cursor + n]

    def get_bits(self, n: int, from_preprocessed_word_bits: bool = True) -> list[int]:
        if from_preprocessed_word_bits:
            # If we've run out of bits in this word, prepare another word now
            if len(self.preprocessed_data_bits_of_current_word) == 0:
                self.preprocess_next_word()

        out = self.peek_bit_count(n, from_preprocessed_word_bits=from_preprocessed_word_bits)

        if from_preprocessed_word_bits:
            # Drop these bits from the queued, preprocessed bits of the current word
            self.preprocessed_data_bits_of_current_word = self.preprocessed_data_bits_of_current_word[n:]
        else:
            # Advance our position in the underlying buffer
            self.cursor += n
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

    def preprocess_next_word(self) -> None:
        # _logger.info(f'Preprocessing next subframe word...')
        # Each word needs to be decoded based on the last parity bits from the previous word
        prev_d29 = self.last_two_parity_bits[0]
        prev_d30 = self.last_two_parity_bits[1]
        word_bits = self.get_bits(
            _BIT_COUNT_PER_WORD,
            # Read raw bits from our subframe buffer
            from_preprocessed_word_bits=False,
        )
        # _logger.info(f'Word bits: {word_bits}')
        data_bits = word_bits[:_DATA_BIT_COUNT_PER_WORD]

        complemented_data_bits = []
        for i, data_bit in enumerate(data_bits):
            complemented_bit = (data_bit + prev_d30) % 2
            complemented_data_bits.append(complemented_bit)

        # Validate each parity bit
        actual_parity_bits = word_bits[-_PARITY_BIT_COUNT_PER_WORD:]
        # The bit indexes are written out to match the equation as-written in Table 20-XIV. Parity Encoding Equations,
        # for reader clarity. This means we need to do a bit of post-processing on the bit indexes.
        self._validate_parity_bit(
            complemented_data_bits,
            actual_parity_bits[0],
            prev_d29,
            [1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23],
        )
        self._validate_parity_bit(
            complemented_data_bits,
            actual_parity_bits[1],
            prev_d30,
            [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24],
        )
        self._validate_parity_bit(
            complemented_data_bits,
            actual_parity_bits[2],
            prev_d29,
            [1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22],
        )
        self._validate_parity_bit(
            complemented_data_bits,
            actual_parity_bits[3],
            prev_d30,
            [2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23],
        )
        self._validate_parity_bit(
            complemented_data_bits,
            actual_parity_bits[4],
            prev_d30,
            [1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24],
        )
        self._validate_parity_bit(
            complemented_data_bits,
            actual_parity_bits[5],
            prev_d29,
            [3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24],
        )

        # Keep the last two parity bits for the next word
        self.last_two_parity_bits = actual_parity_bits[-2:]
        # And persist the preprocessed word bits
        self.preprocessed_data_bits_of_current_word = complemented_data_bits

    @staticmethod
    def _validate_parity_bit(
        complemented_data_bits: list[int],
        actual_parity_bit: int,
        bit_from_previous_parity_word: int,
        spec_bit_indexes_to_use_as_xor_inputs: list[int],
    ):
        data_bits_to_use_for_xor_inputs = [complemented_data_bits[i - 1] for i in spec_bit_indexes_to_use_as_xor_inputs]
        bits_to_use_for_xor_inputs = [bit_from_previous_parity_word, *data_bits_to_use_for_xor_inputs]
        accumulator = 0
        for bit in bits_to_use_for_xor_inputs:
            accumulator = (accumulator + bit) % 2
        if accumulator != actual_parity_bit:
            # raise ValueError(
            #    f'Failed parity check: {complemented_data_bits}, {actual_parity_bit}, '
            #    f'{bit_from_previous_parity_word}, {spec_bit_indexes_to_use_as_xor_inputs}'
            # )
            _logger.info(
                f"Failed parity check: {complemented_data_bits}, {accumulator} != {actual_parity_bit}, "
                f"d30*={bit_from_previous_parity_word}, xors={spec_bit_indexes_to_use_as_xor_inputs}"
            )

    def parse_telemetry_word(self) -> TelemetryWord:
        # Ref: IS-GPS-200L, Figure 20-2
        tlm_prelude = [1, 0, 0, 0, 1, 0, 1, 1]
        # TODO(PT): We need to be able to tell if the polarity of the bits has just flipped?
        # Maybe we could do it each time we detect some kind of cycle slip?
        try:
            self.match_bits(tlm_prelude)
        except ValueError:
            raise IncorrectPreludeBitsError()
        telemetry_message = self.get_bits(14)
        integrity_status_flag = self.get_bit()
        spare_bit = self.get_bit()
        return TelemetryWord(
            telemetry_message=telemetry_message,
            integrity_status_flag=integrity_status_flag,
            spare_bit=spare_bit,
        )

    def parse_handover_word(self) -> HandoverWord:
        time_of_week = self.get_bits(17)
        alert_flag = self.get_bit()
        anti_spoof_flag = self.get_bit()
        subframe_id_codes = self.get_bits(3)
        to_be_solved = self.get_bits(2)
        subframe_id = GpsSubframeId.from_bits(subframe_id_codes)
        return HandoverWord(
            time_of_week=time_of_week,
            alert_flag=alert_flag,
            anti_spoof_flag=anti_spoof_flag,
            subframe_id=subframe_id,
            to_be_solved=to_be_solved,
        )

    def parse_subframe_1(self) -> NavigationMessageSubframe1:
        # Ref: IS-GPS-200L, 20.3.3.5 Subframes 1, Figure 20-1. Data Format (sheet 1 of 11)
        # Word 3
        week_num_mod_1024 = self.get_unscaled_num(10, twos_complement=False)
        ca_or_p_on_l2 = self.get_bits(2)
        ura_index = self.get_bits(4)
        sv_health = self.get_bits(6)
        issue_of_data_clock_high = self.get_bits(2)

        # Word 4
        l2_p_data_flag = self.get_bit()
        _reserved_block1 = self.get_bits(23)

        # Word 5
        _reserved_block2 = self.get_bits(24)

        # Word 6
        _reserved_block3 = self.get_bits(24)

        # Word 7
        _reserved_block4 = self.get_bits(16)
        estimated_group_delay_differential = self.get_num(bit_count=8, scale_factor_exp2=-31, twos_complement=True)

        # Word 8
        issue_of_data_clock_low = self.get_bits(8)
        issue_of_data_clock = [*issue_of_data_clock_high, *issue_of_data_clock_low]
        t_oc = self.get_num(bit_count=16, scale_factor_exp2=4, twos_complement=False)

        # Word 9
        a_f2 = self.get_num(bit_count=8, scale_factor_exp2=-55, twos_complement=True)
        a_f1 = self.get_num(bit_count=16, scale_factor_exp2=-43, twos_complement=True)

        # Word 10
        a_f0 = self.get_num(bit_count=22, scale_factor_exp2=-31, twos_complement=True)
        _to_be_solved = self.get_bits(2)

        return NavigationMessageSubframe1(
            week_num_mod_1024_bits=week_num_mod_1024,
            ca_or_p_on_l2=ca_or_p_on_l2,
            ura_index=ura_index,
            sv_health=sv_health,
            issue_of_data_clock=issue_of_data_clock,
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
        correction_to_orbital_radius_sin = self.get_num(bit_count=16, scale_factor_exp2=-5, twos_complement=True)

        # Word 4
        mean_motion_difference_from_computed_value = self.get_num(
            bit_count=16, scale_factor_exp2=-43, twos_complement=True
        )
        mean_anomaly_at_reference_time_high = self.get_bits(8)

        # Word 5
        mean_anomaly_at_reference_time_low = self.get_bits(24)
        mean_anomaly_at_reference_time_bits = [
            *mean_anomaly_at_reference_time_high,
            *mean_anomaly_at_reference_time_low,
        ]
        mean_anomaly_at_reference_time = self.get_num_from_bits(
            bits=mean_anomaly_at_reference_time_bits, scale_factor_exp2=-31, twos_complement=True
        )

        # Word 6
        correction_to_latitude_cos = self.get_num(bit_count=16, scale_factor_exp2=-29, twos_complement=True)
        eccentricity_high = self.get_bits(8)

        # Word 7
        eccentricity_low = self.get_bits(24)
        eccentricity_bits = [*eccentricity_high, *eccentricity_low]
        eccentricity = self.get_num_from_bits(bits=eccentricity_bits, scale_factor_exp2=-33, twos_complement=False)

        # Word 8
        correction_to_latitude_sin = self.get_num(bit_count=16, scale_factor_exp2=-29, twos_complement=True)
        sqrt_semi_major_axis_high = self.get_bits(8)

        sqrt_semi_major_axis_low = self.get_bits(24)
        sqrt_semi_major_axis_bits = [*sqrt_semi_major_axis_high, *sqrt_semi_major_axis_low]
        sqrt_semi_major_axis = self.get_num_from_bits(
            bits=sqrt_semi_major_axis_bits, scale_factor_exp2=-19, twos_complement=False
        )

        reference_time_ephemeris = self.get_num(bit_count=16, scale_factor_exp2=4, twos_complement=False)
        fit_interval_flag = self.get_bit()
        age_of_data_offset = self.get_bits(5)
        _to_be_solved = self.get_bits(2)

        # TODO(PT): Add units to each of these
        return NavigationMessageSubframe2(
            issue_of_data_ephemeris=issue_of_data_ephemeris,
            correction_to_orbital_radius_sin=correction_to_orbital_radius_sin,
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

    def parse_subframe_3(self) -> NavigationMessageSubframe3:
        # Ref: IS-GPS-200L, Figure 20-1. Data Format (sheet 3 of 11)
        # Ref: IS-GPS-200L, 20.3.3.4.1 Content of Subframes 2 and 3
        #
        # Word 3
        correction_to_inclination_angle_cos = self.get_num(bit_count=16, scale_factor_exp2=-29, twos_complement=True)
        longitude_of_ascending_node_high = self.get_bits(8)

        # Word 4
        longitude_of_ascending_node_low = self.get_bits(24)
        longitude_of_ascending_node = self.get_num_from_bits(
            bits=[*longitude_of_ascending_node_high, *longitude_of_ascending_node_low],
            scale_factor_exp2=-31,
            twos_complement=True,
        )

        # Word 5
        correction_to_inclination_angle_sin = self.get_num(bit_count=16, scale_factor_exp2=-29, twos_complement=True)
        inclination_angle_high = self.get_bits(8)

        # Word 6
        inclination_angle_low = self.get_bits(24)
        inclination_angle = self.get_num_from_bits(
            bits=[*inclination_angle_high, *inclination_angle_low],
            scale_factor_exp2=-31,
            twos_complement=True,
        )

        # Word 7
        correction_to_orbital_radius_cos = self.get_num(bit_count=16, scale_factor_exp2=-5, twos_complement=True)
        argument_of_perigee_high = self.get_bits(8)

        # Word 8
        argument_of_perigee_low = self.get_bits(24)
        argument_of_perigee = self.get_num_from_bits(
            bits=[*argument_of_perigee_high, *argument_of_perigee_low],
            scale_factor_exp2=-31,
            twos_complement=True,
        )

        # Word 9
        rate_of_right_ascension = self.get_num(bit_count=24, scale_factor_exp2=-43, twos_complement=True)

        # Word 10
        issue_of_data_ephemeris = self.get_bits(8)
        rate_of_inclination_angle = self.get_num(bit_count=14, scale_factor_exp2=-43, twos_complement=True)
        _to_be_solved = self.get_bits(2)

        return NavigationMessageSubframe3(
            correction_to_inclination_angle_cos=correction_to_inclination_angle_cos,
            longitude_of_ascending_node=longitude_of_ascending_node,
            correction_to_inclination_angle_sin=correction_to_inclination_angle_sin,
            inclination_angle=inclination_angle,
            correction_to_orbital_radius_cos=correction_to_orbital_radius_cos,
            argument_of_perigee=argument_of_perigee,
            rate_of_right_ascension=rate_of_right_ascension,
            issue_of_data_ephemeris=issue_of_data_ephemeris,
            rate_of_inclination_angle=rate_of_inclination_angle,
        )

    def parse_subframe_4(self) -> NavigationMessageSubframe4:
        # Ref: IS-GPS-200L, Figure 20-1. Data Format, sheets 6 through 11
        # Ref: IS-GPS-200L, 20.3.3.5 Subframes 4 and 5
        #
        # Word 3
        data_id = self.get_unscaled_num(2, twos_complement=False)
        page_id = self.get_unscaled_num(6, twos_complement=False)

        # PT: The rest of the bits depend on the page we're in
        # For now, skip the rest of the subframe
        _skipped_word_remainder = self.get_bits(8)
        # PT: Current implementation limitation of get_bits(): we need to invoke it once per word. Otherwise, the
        # parity-processed bits won't be refilled after the first word. Would be easy to fix.
        for remaining_word_idx in range(8):
            _skipped_word_bits = self.get_bits(_DATA_BIT_COUNT_PER_WORD, from_preprocessed_word_bits=False)

        return NavigationMessageSubframe4(
            data_id,
            page_id,
        )

    def parse_subframe_5(self) -> NavigationMessageSubframe5:
        # Ref: IS-GPS-200L, 20.3.3.5 Subframes 4 and 5
        # TODO(PT): The below is only valid for pages 1 through 24!
        # TODO(PT): Validate page ID?!

        # Word 3
        data_id = self.match_bits([0, 1])
        satellite_id = self.get_bits(6)
        eccentricity = self.get_num(bit_count=16, scale_factor_exp2=-21, twos_complement=False)

        # Word 4
        time_of_ephemeris = self.get_num(bit_count=8, scale_factor_exp2=12, twos_complement=False)
        delta_inclination_angle = self.get_num(bit_count=16, scale_factor_exp2=-19, twos_complement=True)

        # Word 5
        right_ascension_rate = self.get_num(bit_count=16, scale_factor_exp2=-38, twos_complement=True)
        sv_health = self.get_bits(8)

        # Word 6
        semi_major_axis_sqrt = self.get_num(bit_count=24, scale_factor_exp2=-11, twos_complement=False)

        # Word 7
        longitude_of_ascension_mode = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)

        # Word 8
        argument_of_perigree = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)

        # Word 9
        mean_anomaly_at_reference_time = self.get_num(bit_count=24, scale_factor_exp2=-23, twos_complement=True)

        # Word 10
        a_f0_high = self.get_bits(8)
        a_f1 = self.get_num(bit_count=11, scale_factor_exp2=-38, twos_complement=True)
        a_f0_low = self.get_bits(3)
        a_f0_bits = [*a_f0_high, *a_f0_low]
        a_f0 = self.get_num_from_bits(bits=a_f0_bits, scale_factor_exp2=-20, twos_complement=True)

        _to_be_solved = self.get_bits(2)

        return NavigationMessageSubframe5(
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
