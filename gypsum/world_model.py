from collections import defaultdict
from enum import Enum, auto
from typing import Generic, Sequence, Type, TypeVar, cast

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.constants import SECONDS_PER_WEEK, UNIX_TIMESTAMP_OF_GPS_EPOCH
from gypsum.events import Event
from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.navigation_message_decoder import EmitSubframeEvent
from gypsum.navigation_message_parser import (
    GpsSubframeId,
    Meters,
    NavigationMessageSubframe1,
    NavigationMessageSubframe2,
    NavigationMessageSubframe3,
    NavigationMessageSubframe4,
    NavigationMessageSubframe5,
    SemiCircles,
    SemiCirclesPerSecond,
)
from gypsum.units import MetersPerSecond, Radians, RadiansPerSecond, Seconds

_ParameterType = TypeVar("_ParameterType")
_ParameterValueType = TypeVar("_ParameterValueType")


class ParameterSet(Generic[_ParameterType, _ParameterValueType]):
    """Tracks a 'set' of parameters that are progressively fleshed out"""

    # Must be set by subclasses
    # PT: It's a lot more convenient to set this explicitly than trying to pull it out of the TypeVar
    _PARAMETER_TYPE = None

    def __init_subclass__(cls, **kwargs):
        if cls._PARAMETER_TYPE is None:
            raise RuntimeError(f"_PARAMETER_TYPE must be set by subclasses")

    def __init__(self) -> None:
        self.parameter_type_to_value: dict[_ParameterType, _ParameterValueType | None] = {
            t: None for t in self._PARAMETER_TYPE
        }

    def is_complete(self) -> bool:
        """Returns whether we have a 'full set' of parameters (i.e. no None values)."""
        return not any(x is None for x in self.parameter_type_to_value.values())

    def set_parameter(self, param_type: _ParameterType, param_value: _ParameterValueType) -> None:
        self.parameter_type_to_value[param_type] = param_value

    def get_parameter(self, param_type: _ParameterType) -> _ParameterValueType | None:
        return self.parameter_type_to_value[param_type]

    def is_parameter_set(self, param_type: _ParameterType) -> bool:
        return self.get_parameter(param_type) is not None

    def _get_parameter_infallibly(self, param_type: _ParameterType) -> _ParameterValueType:
        # PT: For caller convenience, provide infallible accessors to parameters
        maybe_param = self.get_parameter(param_type)
        if maybe_param is None:
            raise RuntimeError(f"Expected {param_type.name} to be available")
        return maybe_param


_OrbitalParameterValueType = Meters | float | SemiCircles | int | Seconds | ReceiverTimestampSeconds


class OrbitalParameterType(Enum):
    # Classical Keplerian orbital parameters
    #
    # Also called 'a'
    SEMI_MAJOR_AXIS = auto()
    # Also called 'e'
    ECCENTRICITY = auto()
    # Also called 'i'
    INCLINATION = auto()
    # Also called 'Omega' or Ω
    LONGITUDE_OF_ASCENDING_NODE = auto()
    # Also called 'omega' or 
    ARGUMENT_OF_PERIGEE = auto()
    # Also called 'M'
    MEAN_ANOMALY_AT_REFERENCE_TIME = auto()

    # Correction parameters
    # Also called 'delta n'
    MEAN_MOTION_DIFFERENCE = auto()

    # Time synchronization parameters
    WEEK_NUMBER = auto()
    EPHEMERIS_REFERENCE_TIME = auto()
    RECEIVER_TIME_AT_LAST_TIMESTAMP = auto()
    GPS_TIME_AT_LAST_TIMESTAMP = auto()

    @property
    def unit(self) -> Type[_OrbitalParameterValueType]:
        return {
            self.SEMI_MAJOR_AXIS: Meters,
            self.ECCENTRICITY: float,
            self.INCLINATION: SemiCircles,
            self.LONGITUDE_OF_ASCENDING_NODE: SemiCircles,
            self.ARGUMENT_OF_PERIGEE: SemiCircles,
            self.MEAN_ANOMALY_AT_REFERENCE_TIME: SemiCircles,
            self.WEEK_NUMBER: int,
            self.EPHEMERIS_REFERENCE_TIME: Seconds,
            self.RECEIVER_TIME_AT_LAST_TIMESTAMP: ReceiverTimestampSeconds,
            self.GPS_TIME_AT_LAST_TIMESTAMP: Seconds,
            self.MEAN_MOTION_DIFFERENCE: SemiCirclesPerSecond,
        }[
            self
        ]  # type: ignore


class OrbitalParameters(ParameterSet[OrbitalParameterType, _OrbitalParameterValueType]):
    """Tracks a 'set' of orbital parameters for a classical 2-body orbit."""

    _PARAMETER_TYPE = OrbitalParameterType

    @property
    def semi_major_axis(self) -> Meters:
        return self._get_parameter_infallibly(OrbitalParameterType.SEMI_MAJOR_AXIS)

    @property
    def eccentricity(self) -> float:
        return self._get_parameter_infallibly(OrbitalParameterType.ECCENTRICITY)

    @property
    def inclination(self) -> Meters:
        return self._get_parameter_infallibly(OrbitalParameterType.INCLINATION)

    @property
    def longitude_of_ascending_node(self) -> Meters:
        return self._get_parameter_infallibly(OrbitalParameterType.LONGITUDE_OF_ASCENDING_NODE)

    @property
    def argument_of_perigee(self) -> Meters:
        return self._get_parameter_infallibly(OrbitalParameterType.ARGUMENT_OF_PERIGEE)

    @property
    def mean_anomaly_at_reference_time(self) -> Meters:
        return self._get_parameter_infallibly(OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME)

    @property
    def mean_motion_difference(self) -> int:
        return self._get_parameter_infallibly(OrbitalParameterType.MEAN_MOTION_DIFFERENCE)

    @property
    def week_number(self) -> int:
        return self._get_parameter_infallibly(OrbitalParameterType.WEEK_NUMBER)

    @property
    def ephemeris_reference_time(self) -> Seconds:
        """Expressed in seconds since start of week"""
        return self._get_parameter_infallibly(OrbitalParameterType.EPHEMERIS_REFERENCE_TIME)

    @property
    def receiver_timestamp_at_last_timestamp(self) -> ReceiverTimestampSeconds:
        return self._get_parameter_infallibly(OrbitalParameterType.RECEIVER_TIME_AT_LAST_TIMESTAMP)

    @property
    def gps_time_at_last_timestamp(self) -> Seconds:
        return self._get_parameter_infallibly(OrbitalParameterType.GPS_TIME_AT_LAST_TIMESTAMP)


# TODO(PT): We should probably have a base class for "decoder events", "world model events", etc., for better typing
class DeterminedSatelliteOrbitEvent(Event):
    def __init__(
        self,
        satellite_id: GpsSatelliteId,
        orbital_parameters: OrbitalParameters,
    ) -> None:
        self.satellite_id = satellite_id
        self.orbital_parameters = orbital_parameters


class GpsWorldModel:
    """Integrates satellite subframes to maintain a model of satellite orbits around Earth"""

    def __init__(self) -> None:
        self.satellite_ids_to_orbital_parameters: dict[GpsSatelliteId, OrbitalParameters] = defaultdict(
            OrbitalParameters
        )

    def handle_subframe_emitted(
        self, satellite_id: GpsSatelliteId, emit_subframe_event: EmitSubframeEvent
    ) -> Sequence[Event]:
        events_to_return = []
        subframe = emit_subframe_event.subframe
        subframe_id = subframe.subframe_id

        orbital_params_for_this_satellite = self.satellite_ids_to_orbital_parameters[satellite_id]

        # Keep track of whether we already had all the orbital parameters for this satellite, so we know whether
        # we've just completed a full set.
        were_orbit_params_already_complete = orbital_params_for_this_satellite.is_complete()

        # Always store the receiver timestamp
        orbital_params_for_this_satellite.set_parameter(
            OrbitalParameterType.RECEIVER_TIME_AT_LAST_TIMESTAMP, emit_subframe_event.receiver_timestamp
        )
        # If we have enough parameters to know the GPS time, store that too
        if orbital_params_for_this_satellite.is_parameter_set(OrbitalParameterType.WEEK_NUMBER):
            gps_week_number = orbital_params_for_this_satellite.week_number
            satellite_time_of_week_in_seconds = emit_subframe_event.handover_word.time_of_week_in_seconds
            gps_satellite_time = (gps_week_number * SECONDS_PER_WEEK) + satellite_time_of_week_in_seconds
            orbital_params_for_this_satellite.set_parameter(
                OrbitalParameterType.GPS_TIME_AT_LAST_TIMESTAMP, gps_satellite_time
            )

        # Extract more orbital parameters, discriminating based on the type of subframe we've just seen
        # Casts because the subframe is currently typed as the subframe base class
        if subframe_id == GpsSubframeId.ONE:
            self._process_subframe1(orbital_params_for_this_satellite, cast(NavigationMessageSubframe1, subframe))
        elif subframe_id == GpsSubframeId.TWO:
            self._process_subframe2(orbital_params_for_this_satellite, cast(NavigationMessageSubframe2, subframe))
        elif subframe_id == GpsSubframeId.THREE:
            self._process_subframe3(orbital_params_for_this_satellite, cast(NavigationMessageSubframe3, subframe))
        elif subframe_id == GpsSubframeId.FOUR:
            self._process_subframe4(orbital_params_for_this_satellite, cast(NavigationMessageSubframe4, subframe))
        elif subframe_id == GpsSubframeId.FIVE:
            self._process_subframe5(orbital_params_for_this_satellite, cast(NavigationMessageSubframe5, subframe))

        # Check whether we've just completed the set of orbital parameters for this satellite
        if not were_orbit_params_already_complete:
            if orbital_params_for_this_satellite.is_complete():
                events_to_return.append(
                    DeterminedSatelliteOrbitEvent(
                        satellite_id=satellite_id,
                        orbital_parameters=orbital_params_for_this_satellite,
                    )
                )

        # Can we carry out a position fix?
        # Calculate a pseudorange
        import math

        if False:
            if time_params_for_this_satellite.is_complete():
                receiver_time_with_bias = emit_subframe_event.receiver_timestamp - UNIX_TIMESTAMP_OF_GPS_EPOCH
                satellite_time_of_week_in_seconds = emit_subframe_event.handover_word.time_of_week_in_seconds
                satellite_time = (
                    time_params_for_this_satellite.week_number * SECONDS_PER_WEEK
                ) + satellite_time_of_week_in_seconds
                print(f"*** Pseudorange for satellite #{satellite_id.id} ***")
                print(f"\tSatellite time:             {satellite_time}")
                print(f"\tReceiver  time: (with bias) {receiver_time_with_bias}")

                pseudo_transit_time = receiver_time_with_bias - satellite_time
                speed_of_light: MetersPerSecond = 299_792_458
                pseudorange = pseudo_transit_time * speed_of_light
                print(f"\tPseudo transit time:        {pseudo_transit_time}")
                print(f"\tPseudorange:                {pseudorange}")

        # Do we have at least 4 satellites with a complete set of orbital and time parameters?
        # If so, we can solve a position and time fix now
        satellites_with_complete_orbital_parameters = {
            sv_id: op for sv_id, op in self.satellite_ids_to_orbital_parameters.items() if op.is_complete()
        }
        if len(satellites_with_complete_orbital_parameters) >= 4:
            # Comparing the timestamp of the HOW to the receiver timestamp gives (travel time + clock bias)
            # TODO(PT): I anticipate this will give bad results because we're taking pseudoranges 6 seconds apart - the satellites will be in very
            # different positions!
            # now_gps_timestamp = emit_subframe_event.receiver_timestamp - UNIX_TIMESTAMP_OF_GPS_EPOCH

            # All the satellite times should be synchronized, so just look at the newest one to determine GPS time
            satellite_time_of_week_in_seconds = emit_subframe_event.handover_word.time_of_week_in_seconds
            gps_week_number = orbital_params_for_this_satellite.week_number
            current_gps_time = (gps_week_number * SECONDS_PER_WEEK) + satellite_time_of_week_in_seconds

            for satellite_id, orbital_params in satellites_with_complete_orbital_parameters.items():
                print(f"*** {satellite_id}")
                ephemeris_reference_time = orbital_params.get_parameter(OrbitalParameterType.EPHEMERIS_REFERENCE_TIME)
                print(f"\tEphemeris reference time {ephemeris_reference_time}")
                mean_motion_difference = orbital_params.get_parameter(OrbitalParameterType.MEAN_MOTION_DIFFERENCE)
                print(f"\tMean motion difference {mean_motion_difference}")

                # The reference time is always expressed as seconds into the current week
                time_to_ephemeris_reference_time = satellite_time_of_week_in_seconds - ephemeris_reference_time
                print(f"\tTime to ephemeris reference time {time_to_ephemeris_reference_time}")

                semi_major_axis = orbital_params.get_parameter(OrbitalParameterType.SEMI_MAJOR_AXIS)
                print(f"\tSemi major axis {semi_major_axis}")
                earth_gravitational_constant = 3.986005 * (10**14)
                computed_mean_motion: RadiansPerSecond = math.sqrt(
                    (earth_gravitational_constant / (semi_major_axis**3))
                )
                print(f"\tComputed mean motion {computed_mean_motion}")

                # Divide by pi to convert radians to semicircles
                corrected_mean_motion = (computed_mean_motion / math.pi) + mean_motion_difference
                print(f"\tCorrected mean motion {corrected_mean_motion}")

                mean_anomaly_at_reference_time = orbital_params.get_parameter(
                    OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME
                )
                mean_anomaly_now = mean_anomaly_at_reference_time + (
                    corrected_mean_motion * time_to_ephemeris_reference_time
                )
                print(f"\tMean anomaly at reference time {mean_anomaly_at_reference_time}")
                print(f"\tMean anomaly now {mean_anomaly_now}")

                # Solve current eccentric anomaly by iteration
                # (Should be in radians)
                eccentric_anomaly_now_estimation = mean_anomaly_now * math.pi
                eccentricity = orbital_params.get_parameter(OrbitalParameterType.ECCENTRICITY)
                for i in range(10):
                    numerator = (
                        mean_anomaly_now
                        - eccentric_anomaly_now_estimation
                        + (eccentricity * math.sin(eccentric_anomaly_now_estimation))
                    )
                    denominator = 1 - (eccentricity * math.cos(eccentric_anomaly_now_estimation))
                    eccentric_anomaly_now_estimation = eccentric_anomaly_now_estimation + (numerator / denominator)
                eccentric_anomaly_now: Radians = eccentric_anomaly_now_estimation
                print(f"\tEccentric anomaly now {eccentric_anomaly_now}")
            raise NotImplementedError()

        if False:
            # Organize the satellite/orbital parameter pairs by the last GPS timestamp we've seen for each.
            # To carry out a fix, we'll need each satellite to be fixed at the same GPS timestamp.
            gps_timestamps_to_satellites_and_orbital_parameters = defaultdict(list)
            for sv_id, orbital_params in satellites_with_complete_orbital_parameters.items():
                gps_timestamp = orbital_params.get_parameter(OrbitalParameterType.GPS_TIME_AT_LAST_TIMESTAMP)
                gps_timestamps_to_satellites_and_orbital_parameters[gps_timestamp].append((sv_id, orbital_params))

            for gps_timestamp, sv_ids_and_params_tups in gps_timestamps_to_satellites_and_orbital_parameters.items():
                if len(sv_ids_and_params_tups) >= 4:
                    print(
                        f"Found 4 full sets of orbital params for GPS timestamp: {gps_timestamp}: {sv_ids_and_params_tups}"
                    )
                else:
                    print(f"Not enough full sets for timestamp {gps_timestamp}: {sv_ids_and_params_tups}")

        # satellites_with_complete_time_parameters = {sv_id: tp for sv_id, tp in satellites_with_complete_orbital_parameters.items() if tp.is_complete()}
        # if len(satellites_with_complete_orbital_parameters) >= 4:
        #    print(f'*** Carrying out a PT fix!')
        #
        #    # First, propagate the current orbit positions for each satellite
        #    receiver_time_with_bias = emit_subframe_event.receiver_timestamp - UNIX_TIMESTAMP_OF_GPS_EPOCH

        return events_to_return

    def _process_subframe1(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe1) -> None:
        orbital_parameters.set_parameter(OrbitalParameterType.WEEK_NUMBER, subframe.week_num)

    def _process_subframe2(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe2) -> None:
        orbital_parameters.set_parameter(
            OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME, subframe.mean_anomaly_at_reference_time
        )
        orbital_parameters.set_parameter(OrbitalParameterType.ECCENTRICITY, subframe.eccentricity)
        # The satellite transmits the square root of the semi-major axis, so square it now.
        orbital_parameters.set_parameter(OrbitalParameterType.SEMI_MAJOR_AXIS, subframe.sqrt_semi_major_axis**2)
        orbital_parameters.set_parameter(
            OrbitalParameterType.MEAN_MOTION_DIFFERENCE, subframe.mean_motion_difference_from_computed_value
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.EPHEMERIS_REFERENCE_TIME, subframe.reference_time_ephemeris
        )

    def _process_subframe3(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe3) -> None:
        orbital_parameters.set_parameter(OrbitalParameterType.INCLINATION, subframe.inclination_angle)
        orbital_parameters.set_parameter(OrbitalParameterType.ARGUMENT_OF_PERIGEE, subframe.argument_of_perigee)
        orbital_parameters.set_parameter(
            OrbitalParameterType.LONGITUDE_OF_ASCENDING_NODE, subframe.longitude_of_ascending_node
        )

    def _process_subframe4(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe4) -> None:
        pass

    def _process_subframe5(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe5) -> None:
        pass
