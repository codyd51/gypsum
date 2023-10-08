from dataclasses import dataclass

import numpy as np
from scipy.signal import max_len_seq, resample


@dataclass
class ChipDelayMs:
    """New-type to represent the delay assigned to a given GPS satellite PRN.
    Each chip is transmitted over 1 millisecond, so the chip delay is directly expressed in milliseconds.
    """
    delay_ms: int

    def __init__(self, delay_ms: int) -> None:
        self.delay_ms = delay_ms


@dataclass
class GpsSatelliteId:
    """New-type to semantically store GPS satellite IDs by their PRN signal ID"""
    id: int

    def __init__(self, id: int) -> None:
        self.id = id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class GpsReplicaPrnSignal:
    """New-type to semantically store the 'replica' PRN signal for a given satellite"""
    inner: np.ndarray


def _generate_ca_code_rolled_by(delay_ms: int) -> np.ndarray:
    """Generate the C/A code by generating the 'G1' and 'G2' sequences, then mixing them.
    Note that we generate the 'pure' G2 code, then roll it by `delay_ms` chips before mixing. This code is only ever
    used in the context of the PRN for a particular GPS satellite, so we don't need to access it pre-roll (though it's
    fine to pass `delay_ms=0` if this is desired.

    Note that the signal generated by this function needs further post-processing (to adjust its domain and range)
    before it represents a replica PRN signal.

    Ref: IS-GPS-200L §3.3.2.3: C/A-Code Generation
    """
    seq_bit_count = 10
    # PT: I'm out of my depth here, but it appears as though the scipy implementation needs to be passed the 'companion
    # set' of taps to produce the correct output sequence.
    #
    # If I'm understanding correctly, the GPS documentation specifies the taps in 'Galois generator' form, while the
    # scipy implementation needs to be provided the taps in 'Fibonacci generator' form.
    # Further reading: https://web.archive.org/web/20181001062252/http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm
    #
    # Another way of understanding this which may also be correct: the GPS documentation vs. software libraries simply
    # describe the shift register in different orders from each other.
    #
    # (Terms 1 and X^n can be omitted as they're implicit)
    # G1 = X^10 + X^3 + 1
    g1 = max_len_seq(seq_bit_count, taps=[seq_bit_count - 3])[0]
    # G2 = X^10 + X^9 + X^8 + X^6 + X^3 + X^2 + 1
    g2 = max_len_seq(seq_bit_count, taps=[
        seq_bit_count - 9,
        seq_bit_count - 8,
        seq_bit_count - 6,
        seq_bit_count - 3,
        seq_bit_count - 2,
        ])[0]
    return np.bitwise_xor(
        g1,
        np.roll(g2, delay_ms)
    )


def generate_replica_prn_signals(sample_rate: int) -> dict[GpsSatelliteId, GpsReplicaPrnSignal]:
    # Ref: https://www.gps.gov/technical/icwg/IS-GPS-200L.pdf
    # Table 3-Ia. Code Phase Assignments
    # "The G2i sequence is a G2 sequence selectively delayed by pre-assigned number of chips, thereby
    # generating a set of different C/A-codes."
    # "The PRN C/A-code for SV ID number i is a Gold code, Gi(t), of 1 millisecond in length at a chipping
    # rate of 1023 kbps."
    # In other words, the C/A code for each satellite is a time-shifted version of the same signal, and the
    # delay is expressed in terms of a number of chips (each of which occupies 1ms).
    # PT: The above comment is wrong, the *total 1023-chip PRN* is transmitted every 1ms!
    satellite_id_to_delay_by_chip_count = {
        GpsSatelliteId(1): ChipDelayMs(5),
        GpsSatelliteId(2): ChipDelayMs(6),
        GpsSatelliteId(3): ChipDelayMs(7),
        GpsSatelliteId(4): ChipDelayMs(8),
        GpsSatelliteId(5): ChipDelayMs(17),
        GpsSatelliteId(6): ChipDelayMs(18),
        GpsSatelliteId(7): ChipDelayMs(139),
        GpsSatelliteId(8): ChipDelayMs(140),
        GpsSatelliteId(9): ChipDelayMs(141),
        GpsSatelliteId(10): ChipDelayMs(251),
        GpsSatelliteId(11): ChipDelayMs(252),
        GpsSatelliteId(12): ChipDelayMs(254),
        GpsSatelliteId(13): ChipDelayMs(255),
        GpsSatelliteId(14): ChipDelayMs(256),
        GpsSatelliteId(15): ChipDelayMs(257),
        GpsSatelliteId(16): ChipDelayMs(258),
        GpsSatelliteId(17): ChipDelayMs(469),
        GpsSatelliteId(18): ChipDelayMs(470),
        GpsSatelliteId(19): ChipDelayMs(471),
        GpsSatelliteId(20): ChipDelayMs(472),
        GpsSatelliteId(21): ChipDelayMs(473),
        GpsSatelliteId(22): ChipDelayMs(474),
        GpsSatelliteId(23): ChipDelayMs(509),
        GpsSatelliteId(24): ChipDelayMs(512),
        GpsSatelliteId(25): ChipDelayMs(513),
        GpsSatelliteId(26): ChipDelayMs(514),
        GpsSatelliteId(27): ChipDelayMs(515),
        GpsSatelliteId(28): ChipDelayMs(516),
        GpsSatelliteId(29): ChipDelayMs(859),
        GpsSatelliteId(30): ChipDelayMs(860),
        GpsSatelliteId(31): ChipDelayMs(861),
        GpsSatelliteId(32): ChipDelayMs(862),
    }
    satellite_id_to_replica_prn = {}
    for sat_name, prn_chip_delay_ms in satellite_id_to_delay_by_chip_count.items():

        # Generate the pure PRN signal
        prn_signal = _generate_ca_code_rolled_by(prn_chip_delay_ms.delay_ms)

        # We're going to be searching for this marker in a real/analog signal that we digitally sample at
        # a discrete rate.
        # Therefore, we need to 'sample' the PRN signal we've just created, so we know what it *should* look like in the
        # real measured signal.
        #
        # Firstly, the domain of our generated signal is currently [0 to 1], centered at 0.5.
        # The data that comes in via our antenna will instead vary from [-1 to 1], centered at 0.
        # Translate our generated signal so that it's centered at 0 instead of ...
        translated_prn_signal = prn_signal - 0.5
        # And scale it so that the domain goes from [-1 to 1] instead of [-0.5 to 0.5].
        # Note our signal has exactly 1023 data points (which is the correct/exact length of the G2 code)
        scaled_prn_signal = translated_prn_signal * 2
        # Finally, sample our generated signal so that we can see the sort of thing to look for in the data on the wire
        sampled_signal = resample(scaled_prn_signal, sample_rate // 1000)

        # All done, save the replica signal
        satellite_id_to_replica_prn[sat_name] = GpsReplicaPrnSignal(sampled_signal)
    return satellite_id_to_replica_prn
