from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from typing import Callable
from typing import Generic, Sequence, Type, TypeVar, cast
from typing import Iterator
from typing import Self

import math

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
from gypsum.units import GpsSatelliteSeconds
from gypsum.units import GpsSatelliteSecondsIntoWeek
from gypsum.units import MetersPerSecond, Seconds
from gypsum.units import SampleCount

_ParameterType = TypeVar("_ParameterType")
_ParameterValueType = TypeVar("_ParameterValueType")


@dataclass
class EcefCoordinates:
    x: float
    y: float
    z: float


class ParameterSet(Generic[_ParameterType, _ParameterValueType]):
    """Tracks a 'set' of parameters that are progressively fleshed out"""

    # Must be set by subclasses
    # PT: It's a lot more convenient to set this explicitly than trying to pull it out of the TypeVar
    _PARAMETER_TYPE = None

    def __init_subclass__(cls, **kwargs):
        if cls._PARAMETER_TYPE is None:
            raise RuntimeError(f"_PARAMETER_TYPE must be set by subclasses")

    @classmethod
    def __get_validators__(cls) -> Iterator[Callable[..., Any]]:
        yield cls.validate

    @classmethod
    def validate(cls, v: Any, validation_info) -> Self:
        return v
        raise NotImplementedError(v, x)

    def json_dump(self) -> dict[str, Any]:
        return self.parameter_type_to_value

    def __init__(self) -> None:
        self.parameter_type_to_value: dict[_ParameterType, _ParameterValueType | None] = {
            t: None for t in self._PARAMETER_TYPE
        }

    def is_complete(self) -> bool:
        """Returns whether we have a 'full set' of parameters (i.e. no None values)."""
        return not any(x is None for x in self.parameter_type_to_value.values())

    def clear_parameter(self, param_type: _ParameterType) -> None:
        self.parameter_type_to_value[param_type] = None

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
    # Also called Cuc
    CORRECTION_TO_ARGUMENT_OF_LATITUDE_COS = auto()
    # Also called Cus
    CORRECTION_TO_ARGUMENT_OF_LATITUDE_SIN = auto()
    # Also called Crc
    CORRECTION_TO_ORBITAL_RADIUS_COS = auto()
    # Also called Crs
    CORRECTION_TO_ORBITAL_RADIUS_SIN = auto()
    # Also called Cic
    CORRECTION_TO_INCLINATION_ANGLE_COS = auto()
    # Also called Cis
    CORRECTION_TO_INCLINATION_ANGLE_SIN = auto()
    # Also called 'Omega dot'
    RATE_OF_RIGHT_ASCENSION = auto()
    # Also called 'IDOT'
    RATE_OF_INCLINATION_ANGLE = auto()

    # Time synchronization parameters
    WEEK_NUMBER = auto()
    EPHEMERIS_REFERENCE_TIME = auto()
    GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP = auto()

    A_F0 = auto()
    A_F1 = auto()
    A_F2 = auto()
    T_OC = auto()

    @property
    def unit(self) -> Type[_OrbitalParameterValueType]:
        # TODO(PT): Update
        return {
            self.SEMI_MAJOR_AXIS: Meters,
            self.ECCENTRICITY: float,
            self.INCLINATION: SemiCircles,
            self.LONGITUDE_OF_ASCENDING_NODE: SemiCircles,
            self.ARGUMENT_OF_PERIGEE: SemiCircles,
            self.MEAN_ANOMALY_AT_REFERENCE_TIME: SemiCircles,
            self.WEEK_NUMBER: int,
            self.EPHEMERIS_REFERENCE_TIME: Seconds,
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
    def ephemeris_reference_time(self) -> GpsSatelliteSeconds:
        """Expressed in seconds since start of week"""
        return self._get_parameter_infallibly(OrbitalParameterType.EPHEMERIS_REFERENCE_TIME)


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

    def __init__(self, samples_per_prn_transmission: SampleCount) -> None:
        self.satellite_ids_to_orbital_parameters: dict[GpsSatelliteId, OrbitalParameters] = defaultdict(
            OrbitalParameters
        )
        # PT: Not a defaultdict, because it matters whether the satellite tracked in this map.
        self.satellite_ids_to_prn_observations_since_last_handover_timestamp: dict[GpsSatelliteId, int] = {}
        # PT: Update this to a code phase newtype
        self.satellite_ids_to_prn_code_phases: dict[GpsSatelliteId, int] = {}
        self.receiver_current_timestamp: ReceiverTimestampSeconds | None = None
        self.samples_per_prn_transmission = samples_per_prn_transmission

    def handle_processed_1ms_of_antenna_data(self) -> None:
        if self.receiver_current_timestamp is not None:
            self.receiver_current_timestamp += 0.001

    def handle_prn_observed(self, satellite_id: GpsSatelliteId, prn_code_phase) -> None:
        if satellite_id not in self.satellite_ids_to_prn_observations_since_last_handover_timestamp:
            self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id] = 0
        self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id] += 1
        self.satellite_ids_to_prn_code_phases[satellite_id] = prn_code_phase

    def handle_lost_satellite_lock(self, satellite_id: GpsSatelliteId) -> None:
        # We're no longer reliably counting PRNs, so clear our counter state
        del self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id]
        del self.satellite_ids_to_prn_code_phases[satellite_id]
        # And clear the last timestamp for this satellite
        # Otherwise, when we reacquire this satellite, we'll think we have a reliable time reference to work with.
        # Instead, once we start re-tracking this satellite, we'll need to find out from the satellite what its
        # current timestamp is.
        self.satellite_ids_to_orbital_parameters[satellite_id].clear_parameter(
            OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP
        )

    def _can_interrogate_precise_timings_for_satellite(self, satellite_id: GpsSatelliteId) -> bool:
        # We can only rely on precise timings for a satellite if:
        # 1) We're currently counting PRN observations
        # 2) We've received at least one handover word
        # 3) We have enough data from the satellite to interpret the time represented in the handover word's timestamp
        if satellite_id not in self.satellite_ids_to_prn_observations_since_last_handover_timestamp:
            return False

        if satellite_id not in self.satellite_ids_to_prn_code_phases:
            # Should never happen if we're counting PRNs
            raise RuntimeError(f'Expected to have a code phase if we\'re tracking PRNs')

        orbital_parameters = self.satellite_ids_to_orbital_parameters[satellite_id]
        if not orbital_parameters.is_parameter_set(OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP):
            return False

        return True

    def get_pseudorange_for_satellite(self, satellite_id: GpsSatelliteId) -> Seconds | None:
        if not self._can_interrogate_precise_timings_for_satellite(satellite_id):
            return None

        # We need to compare the satellite's timestamp to our receiver's time
        if self.receiver_current_timestamp is None:
            return None

        current_receiver_time = self.receiver_current_timestamp
        gps_system_time_of_week_for_satellite = self._gps_system_time_of_week_for_satellite(satellite_id)

        # Just assume that 1ms always passes for the receiver
        # current_receiver_time = satellite_timestamp_at_last_subframe + (prn_observations_since_last_subframe * 0.001)
        # Add in the current PRN chip phase
        # This represents the 'fractional delay' of the PRN
        # PT: No, this doesn't work, because it'd only tell us the offset within a single millisecond. Where does the
        # other 70ms come from? I guess we really need to measure it.
        # *** It's because other satellites would have seen less ticks!!! ***
        # e.g. 70ms for one satellite and 63ms for another

        time_for_signal_to_arrive = current_receiver_time - gps_system_time_of_week_for_satellite
        #print(f'')
        #print(f'*** Pseudorange for satellite         {satellite_id} ***')
        #print(f'Sat timestamp at last subframe:       {satellite_timestamp_at_last_subframe}')
        #print(f'PRN observations since last subframe: {prn_observations_since_last_subframe}')
        #print(f'Current receiver time:                {current_receiver_time}')
        #print(f'Current satellite time:               {current_satellite_time}')
        #print(f'*** End pseudorange ***')
        #print(f'')

        # PT: I guess this can only tell you the pseudorange at 1ms boundaries, not continuously... doesn't seem like a
        # blocker...
        # TODO(PT): Add code phase
        #code_phase = self.satellite_ids_to_prn_code_phases[satellite_id]
        # TODO(PT): Pass in 2046
        # TODO(PT): Maybe this should be a subtraction..?
        #code_phase_fraction = code_phase / self.samples_per_prn_transmission
        #contribution_from_code_phase = code_phase_fraction * 0.001
        #print(f'Code phase {code_phase}')
        #print(f'Code phase fraction {code_phase_fraction}')
        #print(f'Offset from code phase {contribution_from_code_phase}')
        #time_for_signal_to_arrive += contribution_from_code_phase
        #print(f'Time for signal to arrive:            {time_for_signal_to_arrive} seconds')
        return time_for_signal_to_arrive

    def get_eccentric_anomaly(
        self,
        orbital_params: OrbitalParameters,
        time_of_week: GpsSatelliteSecondsIntoWeek
    ) -> float:
        earth_gravitational_constant = 3.986005e14
        eccentricity = orbital_params.eccentricity
        semi_major_axis = orbital_params.semi_major_axis

        computed_mean_motion = math.sqrt(earth_gravitational_constant / (math.pow(semi_major_axis, 3)))
        mean_motion_difference = orbital_params.get_parameter(OrbitalParameterType.MEAN_MOTION_DIFFERENCE)
        corrected_mean_motion = computed_mean_motion + mean_motion_difference
        # M0
        mean_anomaly_at_reference_time = orbital_params.get_parameter(
            OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME
        )
        ephemeris_reference_time = orbital_params.get_parameter(OrbitalParameterType.EPHEMERIS_REFERENCE_TIME)
        time_from_ephemeris_reference_time = time_of_week - ephemeris_reference_time
        mean_anomaly_now = mean_anomaly_at_reference_time + (
            corrected_mean_motion * time_from_ephemeris_reference_time
        )

        eccentric_anomaly_now_estimation = mean_anomaly_now
        for i in range(10):
            numerator = (
                mean_anomaly_now
                - eccentric_anomaly_now_estimation
                + (eccentricity * math.sin(eccentric_anomaly_now_estimation))
            )
            denominator = 1 - (eccentricity * math.cos(eccentric_anomaly_now_estimation))
            eccentric_anomaly_now_estimation = eccentric_anomaly_now_estimation + (numerator / denominator)
        eccentric_anomaly_now = eccentric_anomaly_now_estimation
        return eccentric_anomaly_now

    def _get_satellite_position_at_time_of_week(self, satellite_id: GpsSatelliteId, satellite_time_of_week: GpsSatelliteSecondsIntoWeek) -> EcefCoordinates:
        orbit_params = self.satellite_ids_to_orbital_parameters[satellite_id]
        if not orbit_params.is_complete():
            raise RuntimeError(f'Expected complete orbital parameters')
        cuc = orbit_params.get_parameter(OrbitalParameterType.CORRECTION_TO_ARGUMENT_OF_LATITUDE_COS)
        cus = orbit_params.get_parameter(OrbitalParameterType.CORRECTION_TO_ARGUMENT_OF_LATITUDE_SIN)
        crc = orbit_params.get_parameter(OrbitalParameterType.CORRECTION_TO_ORBITAL_RADIUS_COS)
        crs = orbit_params.get_parameter(OrbitalParameterType.CORRECTION_TO_ORBITAL_RADIUS_SIN)
        cic = orbit_params.get_parameter(OrbitalParameterType.CORRECTION_TO_INCLINATION_ANGLE_COS)
        cis = orbit_params.get_parameter(OrbitalParameterType.CORRECTION_TO_INCLINATION_ANGLE_SIN)
        i0 = orbit_params.get_parameter(OrbitalParameterType.INCLINATION)
        IDOT = orbit_params.get_parameter(OrbitalParameterType.RATE_OF_INCLINATION_ANGLE)
        eccentricity = orbit_params.get_parameter(OrbitalParameterType.ECCENTRICITY)
        semi_major_axis = orbit_params.get_parameter(OrbitalParameterType.SEMI_MAJOR_AXIS)
        ephemeris_reference_time = orbit_params.get_parameter(OrbitalParameterType.EPHEMERIS_REFERENCE_TIME)
        w = orbit_params.get_parameter(OrbitalParameterType.ARGUMENT_OF_PERIGEE)
        # Omega0
        longitude_of_ascending_node_at_week_start = orbit_params.get_parameter(OrbitalParameterType.LONGITUDE_OF_ASCENDING_NODE)
        # Omega dot
        rate_of_right_ascension = orbit_params.get_parameter(OrbitalParameterType.RATE_OF_RIGHT_ASCENSION)

        earth_rotation_rate = 7.2921151467e-5
        time_from_ephemeris_reference_time = satellite_time_of_week - ephemeris_reference_time
        if time_from_ephemeris_reference_time > 302_400 or time_from_ephemeris_reference_time < -302_400:
            # That is, if tk is greater than 302,400 seconds, subtract 604,800 seconds from tk. If tk is less than -
            # 302,400 seconds, add 604,800 seconds to tk.
            print(f'Time from ephemeris ref time {time_from_ephemeris_reference_time}')
            if time_from_ephemeris_reference_time > 302_400:
                time_from_ephemeris_reference_time -= 604_800
            elif time_from_ephemeris_reference_time < -302_400:
                time_from_ephemeris_reference_time += 604_800
            print(f'Adjusted time from ephemeris ref time {time_from_ephemeris_reference_time}')

        #satellite_time_of_week2 = orbit_params.get_parameter(OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP)
        #satellite_time_of_week = 499758.0
        #print(f'Sat TOW  {satellite_time_of_week}')
        #print(f'Sat TOW2 {satellite_time_of_week2}')

        eccentric_anomaly_now = self.get_eccentric_anomaly(orbit_params, satellite_time_of_week)
        term1 = math.sqrt((1 + eccentricity) / (1 - eccentricity))
        term2 = math.tan(eccentric_anomaly_now / 2)
        true_anomaly = 2 * math.atan(term1 * term2)
        vk = true_anomaly

        argument_of_latitude = vk + w

        duk = (cus * math.sin(2 * argument_of_latitude)) + (cuc * math.cos(2 * argument_of_latitude))
        drk = (crs * math.sin(2 * argument_of_latitude)) + (crc * math.cos(2 * argument_of_latitude))
        dik = (cis * math.sin(2 * argument_of_latitude)) + (cic * math.cos(2 * argument_of_latitude))

        uk = argument_of_latitude + duk
        rk = (semi_major_axis * (1 - (eccentricity * math.cos(eccentric_anomaly_now)))) + drk
        ik = i0 + dik + (IDOT * time_from_ephemeris_reference_time)

        orbital_plane_x = rk * math.cos(uk)
        orbital_plane_y = rk * math.sin(uk)

        omega0 = longitude_of_ascending_node_at_week_start
        omega_at_ref_time = omega0 + ((rate_of_right_ascension - earth_rotation_rate) * time_from_ephemeris_reference_time) - (earth_rotation_rate * ephemeris_reference_time)
        corrected_longitude_of_ascending_node = omega_at_ref_time

        ecef_x = (orbital_plane_x * math.cos(corrected_longitude_of_ascending_node)) - (
                orbital_plane_y * math.cos(ik) * math.sin(corrected_longitude_of_ascending_node))
        ecef_y = (orbital_plane_x * math.sin(corrected_longitude_of_ascending_node)) + (
                orbital_plane_y * math.cos(ik) * math.cos(corrected_longitude_of_ascending_node))
        ecef_z = orbital_plane_y * math.sin(ik)
        return EcefCoordinates(
            ecef_x,
            ecef_y,
            ecef_z
        )

    def _compute_position(self) -> None:
        print('**** _compute_position **** ')
        #print(f'Satellite time of week: {satellite_time_of_week}')
        satellites_with_complete_orbital_parameters = {
            sv_id: op for sv_id, op in self.satellite_ids_to_orbital_parameters.items() if op.is_complete()
        }
        out = []
        positions_at_ref_time = []
        # Filter down to satellites we're actively tracking
        # TODO(PT): Improve this?
        sats = {sv_id: op for sv_id, op in satellites_with_complete_orbital_parameters.items() if sv_id in self.satellite_ids_to_prn_observations_since_last_handover_timestamp and self.satellite_ids_to_prn_observations_since_last_handover_timestamp[sv_id] <= 6000}
        #newest_sat_time = 0
        #for sv_id, op in sats.items():
        #    newest_sat_time = max(newest_sat_time, op.get_parameter(OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP))
        for satellite_id, orbital_params in sats.items():
            print(f"*** {satellite_id}")
            print("{")
            for param_type, param_value in orbital_params.parameter_type_to_value.items():
                print(f'\tOrbitalParameterType.{param_type.name}: {param_value},')
            print("}")
            #print(f'*** Pseudorange for {satellite_id}: {self.get_pseudorange_for_satellite(satellite_id)}')
            pseudo_transmission_time = self.get_pseudorange_for_satellite(satellite_id)
            satellite_time_of_week_now = self._gps_system_time_of_week_for_satellite(satellite_id)

            satellite_pos_now = self._get_satellite_position_at_time_of_week(satellite_id, satellite_time_of_week_now)
            reference_time = orbital_params.get_parameter(OrbitalParameterType.EPHEMERIS_REFERENCE_TIME)
            satellite_pos_at_reference_time = self._get_satellite_position_at_time_of_week(satellite_id, reference_time)
            #print(f'*** Pseudo transmission time for {satellite_id}: {pseudo_transmission_time}')
            #print(f'*** ECF for {satellite_id}:')
            #print(f'{ecf_coords.x}')
            #print(f'{ecf_coords.y}')
            #print(f'{ecf_coords.z}')
            out.append((pseudo_transmission_time, (satellite_pos_now.x, satellite_pos_now.y, satellite_pos_now.z)))
            positions_at_ref_time.append((satellite_pos_at_reference_time.x, satellite_pos_at_reference_time.y, satellite_pos_at_reference_time.z))
            #out.append((tup[0], (ecf_coords.x,ecf_coords.y,ecf_coords.z)))
            #out2 =
            print()
            print()

        from pprint import pprint
        print(f'*** Pseudo transmission times and satellite coordinates now')
        print(out)
        print(f'*** Satellite positions at reference time')
        print(positions_at_ref_time)
        raise NotImplementedError()

    def _gps_system_time_of_week_for_satellite(self, satellite_id: GpsSatelliteId) -> GpsSatelliteSecondsIntoWeek:
        """Since GPS time is intended to be synchronized across all the satellites, this 'should' give the same
        result for any tracked satellite.
        However, the GPS spec (20.3.4.2) specifies that the time emitted by each satellite represents the SV time,
        not the GPS system time. Therefore, there can be small variations due to each SV's clock error and other minor
        effects. Since we want the most exact possible measurement of the SV time, we'll interrogate each SV's emitted
        time directly, rather than making any assumptions about GPS system time.
        """
        if not self._can_interrogate_precise_timings_for_satellite(satellite_id):
            raise RuntimeError(f'Cannot call this now!')
        orbital_params = self.satellite_ids_to_orbital_parameters[satellite_id]
        satellite_time_of_week_at_last_subframe = orbital_params.get_parameter(OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP)
        prn_observations_since_last_subframe = self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id]
        if prn_observations_since_last_subframe > 6000:
            raise RuntimeError(f'More than 6 seconds since we last saw a subframe, giving up! {prn_observations_since_last_subframe}')
        current_satellite_time_of_week = satellite_time_of_week_at_last_subframe
        # Each PRN observation represents 1ms of elapsed time
        current_satellite_time_of_week += prn_observations_since_last_subframe * 0.001

        # TODO(PT): I think it's necessary to include code phase here...
        # TODO(PT): I'm really unsure whether it's appropriate to include the code phase here - maybe that should only be used for pseudoranges, but not for selecting the current GPS time at time of transmission?!
        code_phase = self.satellite_ids_to_prn_code_phases[satellite_id]
        code_phase_fraction = code_phase / self.samples_per_prn_transmission
        contribution_from_code_phase = code_phase_fraction * 0.001
        current_satellite_time_of_week += contribution_from_code_phase

        # Correct the time of last transmission based on the clock correction factors sent by the satellite
        # t can be approximated by tsv
        t = current_satellite_time_of_week
        F = -4.442807633e-10
        e = orbital_params.eccentricity
        # PT: Retain the original sqrt(A)?
        sqrt_A = math.sqrt(orbital_params.semi_major_axis)
        # PT: Circular dependency between current time and current eccentric anomaly...
        # Use an approximation of current time
        Ek = self.get_eccentric_anomaly(orbital_params, current_satellite_time_of_week)
        delta_tr = F * e * sqrt_A * math.sin(Ek)

        af0 = orbital_params.get_parameter(OrbitalParameterType.A_F0)
        af1 = orbital_params.get_parameter(OrbitalParameterType.A_F1)
        af2 = orbital_params.get_parameter(OrbitalParameterType.A_F2)
        toc = orbital_params.get_parameter(OrbitalParameterType.T_OC)
        delta_sv_time = af0 + (af1 * (t - toc)) + (af2 * math.pow(t - toc, 2)) + delta_tr
        print(f'*** Delta SV time {delta_sv_time}')

        corrected_current_satellite_time_of_week = current_satellite_time_of_week - delta_sv_time

        return corrected_current_satellite_time_of_week

    def handle_subframe_emitted(
        self, satellite_id: GpsSatelliteId, emit_subframe_event: EmitSubframeEvent
    ) -> Sequence[Event]:
        events_to_return = []
        subframe = emit_subframe_event.subframe
        subframe_id = subframe.subframe_id

        # We've just observed a handover timestamp, so reset our counter tracking how many PRNs we've observed since
        # then.
        self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id] = 0

        orbital_params_for_this_satellite = self.satellite_ids_to_orbital_parameters[satellite_id]

        # Keep track of whether we already had all the orbital parameters for this satellite, so we know whether
        # we've just completed a full set.
        were_orbit_params_already_complete = orbital_params_for_this_satellite.is_complete()

        # If we have enough parameters to know the satellite's time, store it
        if orbital_params_for_this_satellite.is_parameter_set(OrbitalParameterType.WEEK_NUMBER):
            gps_week_number = orbital_params_for_this_satellite.week_number
            # The HOW gives the timestamp of the leading edge of the *next* subframe.
            # But since we just finished processing this subframe in full, we *are* at the leading edge of the
            # next subframe. Therefore, we don't need to do any adjustment to this timestamp.
            satellite_time_of_week_in_seconds = emit_subframe_event.handover_word.time_of_week_in_seconds
            # gps_satellite_time = (gps_week_number * SECONDS_PER_WEEK) + satellite_time_of_week_in_seconds
            orbital_params_for_this_satellite.set_parameter(
                OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP, satellite_time_of_week_in_seconds
            )
            # If we've never synchronized the receiver clock, synchronize it now.
            # As an initial estimate of our clock offset, set our clock to be the GPS system time, plus a
            # typical (but currently unmeasured) transmission delay. This should give us a receiver time estimate
            # that's not massively different from reality (as transmission times are in the realm of 60-80ms).
            # This guess will then be refined when we carry out our position fix. The magnitude of error here
            # shouldn't be too bad, as GPS should be able to cope with clock bias upwards of several seconds.
            if self.receiver_current_timestamp is None:
                self.receiver_current_timestamp = satellite_time_of_week_in_seconds + (0.001 * 68)

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
        # TODO(PT): Improve
        sats_ready = {
            sv_id: op for sv_id, op in satellites_with_complete_orbital_parameters.items() if self.satellite_ids_to_prn_observations_since_last_handover_timestamp[sv_id] <= 6000
        }
        if len(sats_ready) >= 4:
            self._compute_position()

        return events_to_return

    def _process_subframe1(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe1) -> None:
        orbital_parameters.set_parameter(OrbitalParameterType.WEEK_NUMBER, subframe.week_num)
        orbital_parameters.set_parameter(OrbitalParameterType.A_F0, subframe.a_f0)
        orbital_parameters.set_parameter(OrbitalParameterType.A_F1, subframe.a_f1)
        orbital_parameters.set_parameter(OrbitalParameterType.A_F2, subframe.a_f2)
        orbital_parameters.set_parameter(OrbitalParameterType.T_OC, subframe.t_oc)

    def _process_subframe2(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe2) -> None:
        orbital_parameters.set_parameter(
            OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME, subframe.mean_anomaly_at_reference_time
        )
        orbital_parameters.set_parameter(OrbitalParameterType.ECCENTRICITY, subframe.eccentricity)
        # The satellite transmits the square root of the semi-major axis, so square it now.
        orbital_parameters.set_parameter(OrbitalParameterType.SEMI_MAJOR_AXIS, math.pow(subframe.sqrt_semi_major_axis, 2))
        orbital_parameters.set_parameter(
            OrbitalParameterType.MEAN_MOTION_DIFFERENCE, subframe.mean_motion_difference_from_computed_value
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.EPHEMERIS_REFERENCE_TIME, subframe.reference_time_ephemeris
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_ARGUMENT_OF_LATITUDE_COS, subframe.correction_to_latitude_cos
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_ARGUMENT_OF_LATITUDE_SIN, subframe.correction_to_latitude_sin
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_ORBITAL_RADIUS_SIN, subframe.correction_to_orbital_radius_sin
        )

    def _process_subframe3(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe3) -> None:
        orbital_parameters.set_parameter(OrbitalParameterType.INCLINATION, subframe.inclination_angle)
        orbital_parameters.set_parameter(OrbitalParameterType.ARGUMENT_OF_PERIGEE, subframe.argument_of_perigee)
        orbital_parameters.set_parameter(
            OrbitalParameterType.LONGITUDE_OF_ASCENDING_NODE, subframe.longitude_of_ascending_node
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_INCLINATION_ANGLE_COS, subframe.correction_to_inclination_angle_cos
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_INCLINATION_ANGLE_SIN, subframe.correction_to_inclination_angle_sin
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.RATE_OF_RIGHT_ASCENSION, subframe.rate_of_right_ascension
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.RATE_OF_INCLINATION_ANGLE, subframe.rate_of_inclination_angle
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_ORBITAL_RADIUS_COS, subframe.correction_to_orbital_radius_cos
        )

    def _process_subframe4(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe4) -> None:
        pass

    def _process_subframe5(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe5) -> None:
        pass
