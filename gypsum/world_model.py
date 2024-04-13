import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from typing import Callable
from typing import Generic, Sequence, Type, TypeVar, cast
from typing import Iterator
from typing import Self
from typing import Tuple

import math
import numpy as np

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.constants import ONE_MILLISECOND
from gypsum.constants import SPEED_OF_LIGHT
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
from gypsum.satellite_signal_processing_pipeline import GpsSatelliteSignalProcessingPipeline
from gypsum.units import GpsSatelliteSeconds
from gypsum.units import GpsSatelliteSecondsIntoWeek
from gypsum.units import Seconds
from gypsum.units import SampleCount


_PI = 3.1415926535898

_logger = logging.getLogger(__name__)

_ParameterType = TypeVar("_ParameterType")
_ParameterValueType = TypeVar("_ParameterValueType")


def _get_lat_long(x, y, z):
    # Constants for WGS84 ellipsoid
    a = 6378.137  # semi-major axis in kilometers
    b = 6356.7523142  # semi-minor axis in kilometers
    e_sq = (a ** 2 - b ** 2) / a ** 2  # square of eccentricity

    # Convert to geographic coordinates
    p = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan(z * a / (p * b))

    # Latitude (phi) in radians
    phi = math.atan((z + e_sq * b * math.pow(math.sin(theta), 3)) / (p - e_sq * a * math.pow(math.cos(theta), 3)))
    # Longitude (lambda) in radians
    lambda_ = math.atan2(y, x)

    # Convert radians to degrees
    latitude = math.degrees(phi)
    longitude = math.degrees(lambda_)
    # Altitude above the ellipsoid
    N = a ** 2 / math.sqrt(a ** 2 * math.cos(phi) ** 2 + b ** 2 * math.sin(phi) ** 2)
    altitude = p / math.cos(phi) - N
    return latitude, longitude, altitude


@dataclass
class EcefCoordinates:
    x: float
    y: float
    z: float

    def __hash__(self):
        return hash(self.x) + hash(self.y) + hash(self.z)

    #def __repr__(self):
    #    return f'({self.x=:.2f}, {self.y=:.2f}, {self.z=:.2f})'

    def __str__(self):
        return f'({self.x=:.2f}, {self.y=:.2f}, {self.z=:.2f})'

    @classmethod
    def zero(cls) -> 'EcefCoordinates':
        return cls(x=0, y=0, z=0)


@dataclass
class ReceiverSolution:
    clock_bias: Seconds
    receiver_pos: EcefCoordinates


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
    SQRT_SEMI_MAJOR_AXIS = auto()
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
    RECEIVER_TIMESTAMP_AT_LAST_HOW_TIMESTAMP = auto()
    PRN_TIMESTAMP_OF_LEADING_EDGE_OF_TOW = auto()

    A_F0 = auto()
    A_F1 = auto()
    A_F2 = auto()
    T_OC = auto()
    ESTIMATED_GROUP_DELAY_DIFFERENTIAL = auto()

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
        self.samples_per_prn_transmission = samples_per_prn_transmission
        self.satellite_ids_to_orbital_parameters: dict[GpsSatelliteId, OrbitalParameters] = defaultdict(
            OrbitalParameters
        )
        # PT: Not a defaultdict, because it matters whether the satellite tracked in this map.
        self.satellite_ids_to_prn_observations_since_last_handover_timestamp: dict[GpsSatelliteId, int] = {}
        # PT: Update this to a code phase newtype
        self.satellite_ids_to_prn_code_phases: dict[GpsSatelliteId, int] = {}
        self.receiver_seconds_since_startup: ReceiverTimestampSeconds = 0
        self.receiver_clock_slide: Seconds | None = None
        self.last_receiver_prn_timestamp_by_satellite_id: dict[GpsSatelliteId, ReceiverTimestampSeconds] = {}
        self.satellite_ids_to_receiver_timestamp_and_prn_counts_since_last_how = defaultdict(dict)

        self.satellites_to_observed_prn_counts: dict[GpsSatelliteId, int] = defaultdict(int)
        self.receiver_timestamp_to_satellite_prn_counts: dict[ReceiverTimestampSeconds, dict[GpsSatelliteId, int]] = defaultdict(dict)

    def handle_processed_1ms(self, receiver_timestmap: ReceiverTimestampSeconds) -> None:
        from copy import deepcopy
        self.receiver_timestamp_to_satellite_prn_counts[receiver_timestmap] = deepcopy(self.satellites_to_observed_prn_counts)

    def handle_prn_observed(
        self,
        satellite_id: GpsSatelliteId,
        prn_code_phase,
        start_time: ReceiverTimestampSeconds,
        end_time: ReceiverTimestampSeconds
    ) -> None:
        if satellite_id not in self.satellite_ids_to_prn_observations_since_last_handover_timestamp:
            self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id] = 0
        self.satellite_ids_to_receiver_timestamp_and_prn_counts_since_last_how[satellite_id][start_time] = self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id]
        self.satellite_ids_to_receiver_timestamp_and_prn_counts_since_last_how[satellite_id][end_time] = self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id] + 1
        self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id] += 1
        self.satellite_ids_to_prn_code_phases[satellite_id] = prn_code_phase
        self.last_receiver_prn_timestamp_by_satellite_id[satellite_id] = start_time

        self.satellites_to_observed_prn_counts[satellite_id] += 1

    def handle_lost_satellite_lock(self, satellite_id: GpsSatelliteId, start_time: ReceiverTimestampSeconds) -> None:
        _logger.info(f'handle_lost_satellite_lock({satellite_id})')
        # We're no longer reliably counting PRNs, so clear our counter state
        del self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id]
        del self.last_receiver_prn_timestamp_by_satellite_id[satellite_id]
        if start_time in self.satellite_ids_to_receiver_timestamp_and_prn_counts_since_last_how[satellite_id]:
            del self.satellite_ids_to_receiver_timestamp_and_prn_counts_since_last_how[satellite_id][start_time]
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
        if satellite_id not in self.last_receiver_prn_timestamp_by_satellite_id:
            return False

        if satellite_id not in self.satellite_ids_to_prn_code_phases:
            # Should never happen if we're counting PRNs
            raise RuntimeError(f'Expected to have a code phase if we\'re tracking PRNs')

        orbital_parameters = self.satellite_ids_to_orbital_parameters[satellite_id]
        needed_params = [
            OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP,
            # For clock correction factor
            OrbitalParameterType.ECCENTRICITY,
            OrbitalParameterType.SQRT_SEMI_MAJOR_AXIS,
            OrbitalParameterType.A_F0,
            OrbitalParameterType.A_F1,
            OrbitalParameterType.A_F2,
            OrbitalParameterType.T_OC,
            OrbitalParameterType.ESTIMATED_GROUP_DELAY_DIFFERENTIAL,
        ]
        for needed_param in needed_params:
            if not orbital_parameters.is_parameter_set(needed_param):
                return False

        return True

    def get_pseudorange_for_satellite(
        self,
        satellite_id: GpsSatelliteId,
        receiver_timestamp: ReceiverTimestampSeconds,
        tracker: GpsSatelliteSignalProcessingPipeline,
    ) -> Seconds | None:
        if not self._can_interrogate_precise_timings_for_satellite(satellite_id):
            return None

        # We need to compare the satellite's timestamp to our receiver's time
        current_receiver_time = self.receiver_clock_slide + receiver_timestamp

        # I noticed that this says satellite 32 was 3ms ahead or so ahead of satellite 32 at timestamp 499758 or so, which seems like way too much?
        gps_system_time_of_week_for_satellite = self._gps_observed_system_time_of_week_for_satellite(satellite_id, receiver_timestamp, tracker)
        time_for_signal_to_arrive = current_receiver_time - gps_system_time_of_week_for_satellite
        return time_for_signal_to_arrive

    def get_eccentric_anomaly(
        self,
        orbital_params: OrbitalParameters,
        time_from_ephemeris_reference_time: GpsSatelliteSecondsIntoWeek
    ) -> float:
        earth_gravitational_constant = 3.986004418e14
        eccentricity = orbital_params.eccentricity
        semi_major_axis = math.pow(orbital_params.get_parameter(OrbitalParameterType.SQRT_SEMI_MAJOR_AXIS), 2)

        computed_mean_motion = math.sqrt(earth_gravitational_constant) / math.sqrt(math.pow(semi_major_axis, 3))
        mean_motion_difference = orbital_params.get_parameter(OrbitalParameterType.MEAN_MOTION_DIFFERENCE)
        corrected_mean_motion = computed_mean_motion + mean_motion_difference
        # M0
        mean_anomaly_at_reference_time = orbital_params.get_parameter(
            OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME
        )
        mean_anomaly_now = mean_anomaly_at_reference_time + (
            corrected_mean_motion * time_from_ephemeris_reference_time
        )

        # Mk = Ek - esinEk
        # Mk - Ek = -esinEk
        # -Ek = -esinEk - Mk
        # Ek = esinEk - Mk
        eccentric_anomaly_now_estimation = mean_anomaly_now
        for i in range(7):
            # Kepler's equation
            eccentric_anomaly_now_estimation = mean_anomaly_now + (eccentricity * math.sin(eccentric_anomaly_now_estimation))
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
            # > That is, if tk is greater than 302,400 seconds, subtract 604,800 seconds from tk. If tk is less than -
            # 302,400 seconds, add 604,800 seconds to tk.
            logging.debug(f'Time from ephemeris ref time {time_from_ephemeris_reference_time}')
            if time_from_ephemeris_reference_time > 302_400:
                time_from_ephemeris_reference_time -= 604_800
            elif time_from_ephemeris_reference_time < -302_400:
                time_from_ephemeris_reference_time += 604_800
            logging.debug(f'Adjusted time from ephemeris ref time {time_from_ephemeris_reference_time}')

        eccentric_anomaly_now = self.get_eccentric_anomaly(orbit_params, time_from_ephemeris_reference_time)

        # True anomaly
        # Different equation in doc
        true_anomaly = math.atan2(
            math.sqrt(1 - (eccentricity * eccentricity)) * math.sin(eccentric_anomaly_now),
            math.cos(eccentric_anomaly_now) - eccentricity
        )
        vk = true_anomaly

        argument_of_latitude = vk + w

        duk = (cus * math.sin(2 * argument_of_latitude)) + (cuc * math.cos(2 * argument_of_latitude))
        drk = (crs * math.sin(2 * argument_of_latitude)) + (crc * math.cos(2 * argument_of_latitude))
        dik = (cis * math.sin(2 * argument_of_latitude)) + (cic * math.cos(2 * argument_of_latitude))

        uk = argument_of_latitude + duk
        rk = (semi_major_axis * (1 - (eccentricity * math.cos(eccentric_anomaly_now)))) + drk
        ik = i0 + (IDOT * time_from_ephemeris_reference_time) + dik

        orbital_plane_x = rk * math.cos(uk)
        orbital_plane_y = rk * math.sin(uk)

        omega0 = longitude_of_ascending_node_at_week_start
        omega_at_ref_time = (
            omega0 + (
                (rate_of_right_ascension - earth_rotation_rate) * time_from_ephemeris_reference_time
            ) - (earth_rotation_rate * ephemeris_reference_time)
        )
        corrected_longitude_of_ascending_node = omega_at_ref_time

        ecef_x = (
            (orbital_plane_x * math.cos(corrected_longitude_of_ascending_node))
            - (orbital_plane_y * math.cos(ik) * math.sin(corrected_longitude_of_ascending_node))
        )
        ecef_y = (
            (orbital_plane_x * math.sin(corrected_longitude_of_ascending_node))
            + (orbital_plane_y * math.cos(ik) * math.cos(corrected_longitude_of_ascending_node))
        )
        ecef_z = orbital_plane_y * math.sin(ik)
        return EcefCoordinates(
            ecef_x,
            ecef_y,
            ecef_z
        )

    def _compute_solution_residuals(
        self,
        solution: ReceiverSolution,
        sats_x: list[float],
        sats_y: list[float],
        sats_z: list[float],
        sats_t: list[float],
    ) -> np.ndarray:
        # Computes the difference between the squared distances from the guess to each satellites,
        # and the squared distance that light would travel in the time error.
        return np.array([
            (
                (solution.receiver_pos.x - x) ** 2
                + (solution.receiver_pos.y - y) ** 2
                + (solution.receiver_pos.z - z) ** 2
                - ((SPEED_OF_LIGHT * (time - solution.clock_bias)) ** 2)
            )
            for x, y, z, time in zip(sats_x, sats_y, sats_z, sats_t)
        ])

    def _compute_jacobian_matrix(
        self,
        solution: ReceiverSolution,
        sats_x: list[float],
        sats_y: list[float],
        sats_z: list[float],
        sats_t: list[float],
    ) -> np.ndarray:
        # Computes the Jacobian matrix, necessary for Newton's method.
        return np.array([
            [
                2 * (solution.receiver_pos.x - x),
                2 * (solution.receiver_pos.y - y),
                2 * (solution.receiver_pos.z - z),
                2 * (math.pow(SPEED_OF_LIGHT, 2) * (time - solution.clock_bias)),
            ]
            for x, y, z, time in zip(sats_x, sats_y, sats_z, sats_t)
        ])

    def _solve_position_via_newtons_method(
        self,
        clock_and_ecef: list[Tuple[Seconds, EcefCoordinates]],
        guess: ReceiverSolution,
    ) -> ReceiverSolution:
        sats_x = [tup[1].x for tup in clock_and_ecef]
        sats_y = [tup[1].y for tup in clock_and_ecef]
        sats_z = [tup[1].z for tup in clock_and_ecef]
        sats_t = [tup[0] for tup in clock_and_ecef]
        residuals = self._compute_solution_residuals(guess, sats_x, sats_y, sats_z, sats_t)
        jacobian = self._compute_jacobian_matrix(guess, sats_x, sats_y, sats_z, sats_t)

        for i in range(20):
            V = np.linalg.solve(jacobian, -residuals)

            guess.receiver_pos.x += V[0]
            guess.receiver_pos.y += V[1]
            guess.receiver_pos.z += V[2]
            guess.clock_bias += V[3]

            residuals = self._compute_solution_residuals(guess, sats_x, sats_y, sats_z, sats_t)
            jacobian = self._compute_jacobian_matrix(guess, sats_x, sats_y, sats_z, sats_t)

        return guess

    def _get_pseudorange_and_satellite_position(
        self,
        satellite_id: GpsSatelliteId,
        receiver_timestamp: ReceiverTimestampSeconds,
        tracker: GpsSatelliteSignalProcessingPipeline,
    ) -> Tuple[Seconds, EcefCoordinates]:
        pseudo_transmission_time = self.get_pseudorange_for_satellite(satellite_id, receiver_timestamp, tracker)
        observed_satellite_time = self._gps_observed_system_time_of_week_for_satellite(satellite_id, receiver_timestamp, tracker)

        # PT: This needs to be **at transmission time**, so subtract the pseudorange (*note* this includes our clock bias!)
        # PT: We actually want the satellite time at transmission time... is this the same thing? I think it is
        satellite_pos_now = self._get_satellite_position_at_time_of_week(satellite_id, observed_satellite_time)
        return pseudo_transmission_time, satellite_pos_now

    def attempt_position_fix(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        trackers: dict[GpsSatelliteId, GpsSatelliteSignalProcessingPipeline],
    ) -> ReceiverSolution | None:
        # Do we have at least 4 satellites with a complete set of orbital and time parameters?
        # If so, we can solve a position and time fix now
        satellites_with_complete_orbital_parameters = {
            sv_id: op for sv_id, op in self.satellite_ids_to_orbital_parameters.items() if op.is_complete()
        }
        if len(satellites_with_complete_orbital_parameters) < 4:
            return None

        # TODO(PT): Improve
        # TODO(PT): Make sure we have 4 sats with the same HOW timestamp, or is this sufficient?
        sats_ready = {
            sv_id: op for sv_id, op in satellites_with_complete_orbital_parameters.items()
            if self.satellite_ids_to_prn_observations_since_last_handover_timestamp[sv_id] <= 6000
        }
        if len(sats_ready) < 4:
            return None

        return self._compute_position(receiver_timestamp, list(sats_ready.keys()), trackers)

    def _compute_position(
        self,
        receiver_timestamp: ReceiverTimestampSeconds,
        satellite_ids: list[GpsSatelliteId],
        trackers: dict[GpsSatelliteId, GpsSatelliteSignalProcessingPipeline],
    ) -> ReceiverSolution:
        logging.info('**** _compute_position **** ')
        logging.info(self.satellite_ids_to_prn_code_phases)
        logging.info(self.satellite_ids_to_prn_observations_since_last_handover_timestamp)
        logging.info(self.last_receiver_prn_timestamp_by_satellite_id)
        logging.info(self.receiver_clock_slide)
        guess = ReceiverSolution(
            clock_bias=0,
            receiver_pos=EcefCoordinates.zero(),
        )
        for i in range(5):
            logging.info(f'*** Attemping a round {i} {receiver_timestamp=}...')
            satellite_id_to_pseudoranges_and_satellite_coordinates = {}
            for satellite_id in satellite_ids:
                satellite_id_to_pseudoranges_and_satellite_coordinates[satellite_id] = self._get_pseudorange_and_satellite_position(
                    satellite_id,
                    receiver_timestamp,
                    None#trackers[satellite_id]
                )
            logging.info(f'Iteration {i}: {list(satellite_id_to_pseudoranges_and_satellite_coordinates.values())}')
            guess = self._solve_position_via_newtons_method(
                list(satellite_id_to_pseudoranges_and_satellite_coordinates.values()),
                guess,
            )
            clock_bias, receiver_position = guess.clock_bias, guess.receiver_pos

            x, y, z = receiver_position.x, receiver_position.y, receiver_position.z
            logging.info(f'\tClock bias: {clock_bias}s')
            logging.info(f'\tReceiver position: ({x:.6f}, {y:.6f}, {z:.6f})')

            lat, lon, alt = _get_lat_long(x, y, z)
            logging.info(f'Latitude: {lat}')
            logging.info(f'Longitude: {lon}')
            logging.info(f'{lat}, {lon}')
            logging.info(f'Altitude: {alt}')
            self.receiver_clock_slide -= clock_bias

        return guess

    def _gps_observed_system_time_of_week_for_satellite(
        self,
        satellite_id: GpsSatelliteId,
        receiver_timestamp: ReceiverTimestampSeconds,
        tracker: GpsSatelliteSignalProcessingPipeline,
    ) -> GpsSatelliteSecondsIntoWeek:
        """Since GPS time is intended to be synchronized across all the satellites, this 'should' give the same
        result for any tracked satellite.
        However, the GPS spec (20.3.4.2) specifies that the time emitted by each satellite represents the SV time,
        not the GPS system time. Therefore, there can be small variations due to each SV's clock error and other minor
        effects. Since we want the most exact possible measurement of the SV time, we'll interrogate each SV's emitted
        time directly, rather than making any assumptions about GPS system time.
        Note that this does not directly say anything about what the current time *at the satellite* is.
        Instead, this is very specifically about the latest time that the receiver has seen from the satellite/
        the latest observed timestamp, using PRN ticks as a counter.
        """
        if not self._can_interrogate_precise_timings_for_satellite(satellite_id):
            raise RuntimeError(f'Cannot call this now!')

        # TODO(PT): Track receiver timestamp : PRN count per satellite?
        # TODO(PT): Keep debugging here - it looks like there's a slide of 20 seconds?!

        # Start with the last timestamped HOW that we saw
        orbital_params = self.satellite_ids_to_orbital_parameters[satellite_id]
        satellite_time_of_week_at_last_subframe = orbital_params.get_parameter(OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP)

        # Add in the number of (1ms) PRN ticks since the last subframe
        # TODO(PT): Shouldn't the PRN observation count be ~60-80 ms different per satellite..?
        # TODO(PT): It looks like satellite 32 is in the lead? We're determining everything relative to sat 28, but the RECEIVER_TIMESTAMP_AT_LAST_HOW_TIMESTAMP
        # is greater at sat 32 than the receiver timestamp at the subframe event - maybe investigate further?
        # TODO(PT): We might be computing the TOW relative to a given subframe that's NOT exactly where we are up to in the stream...
        # I think this needs to take a receiver timestamp...
        # TODO(PT): Can we count the # of PRNs at the sample # of the subframe, and the # of PRNs at the time we're computing the receiver pos? Then just subtract them
        # PT: Also need to look up the PRN code phase *at the timestamp*!
        prn_observations_since_last_subframe_at_timestamp = self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id]
        # Each PRN observation represents 1ms of elapsed time
        current_satellite_time_of_week = satellite_time_of_week_at_last_subframe
        current_satellite_time_of_week += ONE_MILLISECOND * prn_observations_since_last_subframe_at_timestamp

        # TODO(PT): This is commented out because the pseudosymbol timestamps themselves now include the PRN code delay
        # prn_code_phase = self.satellite_ids_to_prn_code_phases[satellite_id]
        # time_delay_from_code_phase = ONE_MILLISECOND * (prn_code_phase / self.samples_per_prn_transmission)
        # current_satellite_time_of_week += time_delay_from_code_phase

        # Correct the time of last transmission based on the clock correction factors sent by the satellite
        # Algorithm specified by GPS 20.3.3.3.3.1.
        # Circular dependency between Ek and delta Tr, so compute them iteratively
        delta_sv_time = 0
        orbital_params = self.satellite_ids_to_orbital_parameters[satellite_id]
        for i in range(10):
            t = current_satellite_time_of_week
            F = -4.442807633e-10
            e = orbital_params.eccentricity
            sqrt_A = orbital_params.get_parameter(OrbitalParameterType.SQRT_SEMI_MAJOR_AXIS)
            # PT: Circular dependency between current time and current eccentric anomaly...
            # Use an approximation of current time
            ephemeris_reference_time = orbital_params.get_parameter(OrbitalParameterType.EPHEMERIS_REFERENCE_TIME)
            time_from_ephemeris_reference_time = current_satellite_time_of_week - ephemeris_reference_time
            Ek = self.get_eccentric_anomaly(orbital_params, time_from_ephemeris_reference_time - delta_sv_time)
            delta_tr = F * e * sqrt_A * math.sin(Ek)

            af0 = orbital_params.get_parameter(OrbitalParameterType.A_F0)
            af1 = orbital_params.get_parameter(OrbitalParameterType.A_F1)
            af2 = orbital_params.get_parameter(OrbitalParameterType.A_F2)
            toc = orbital_params.get_parameter(OrbitalParameterType.T_OC)
            tgd = orbital_params.get_parameter(OrbitalParameterType.ESTIMATED_GROUP_DELAY_DIFFERENTIAL)
            delta_sv_time = af0 + (af1 * (t - toc)) + (math.pow(af2 * (t - toc), 2)) + delta_tr - tgd

        current_satellite_time_of_week -= delta_sv_time

        return current_satellite_time_of_week

    def handle_subframe_emitted(
        self, satellite_id: GpsSatelliteId, emit_subframe_event: EmitSubframeEvent
    ) -> Sequence[Event]:
        events_to_return = []
        subframe = emit_subframe_event.subframe
        subframe_id = subframe.subframe_id

        # We've just observed a handover timestamp, so reset our counter tracking how many PRNs we've observed since
        # then.
        self.satellite_ids_to_prn_observations_since_last_handover_timestamp[satellite_id] = 0
        self.satellite_ids_to_receiver_timestamp_and_prn_counts_since_last_how[satellite_id][emit_subframe_event.receiver_timestamp] = 0
        self.satellite_ids_to_receiver_timestamp_and_prn_counts_since_last_how[satellite_id][emit_subframe_event.trailing_edge_receiver_timestamp] = 0

        orbital_params_for_this_satellite = self.satellite_ids_to_orbital_parameters[satellite_id]

        # Keep track of whether we already had all the orbital parameters for this satellite, so we know whether
        # we've just completed a full set.
        were_orbit_params_already_complete = orbital_params_for_this_satellite.is_complete()

        # The HOW gives the timestamp of the leading edge of the *next* subframe.
        # But since we just finished processing this subframe in full, we *are* at the leading edge of the
        # next subframe. Therefore, we don't need to do any adjustment to this timestamp.
        satellite_time_of_week_in_seconds = emit_subframe_event.handover_word.time_of_week_in_seconds
        orbital_params_for_this_satellite.set_parameter(
            OrbitalParameterType.GPS_TIME_OF_WEEK_AT_LAST_TIMESTAMP, satellite_time_of_week_in_seconds
        )
        orbital_params_for_this_satellite.set_parameter(
            OrbitalParameterType.RECEIVER_TIMESTAMP_AT_LAST_HOW_TIMESTAMP, emit_subframe_event.trailing_edge_receiver_timestamp
        )
        logging.info(
            f'*** Got a subframe from {satellite_id} with a TOW for the '
            f'next PRN timestamped at {emit_subframe_event.trailing_edge_receiver_timestamp}'
        )
        orbital_params_for_this_satellite.set_parameter(
            OrbitalParameterType.PRN_TIMESTAMP_OF_LEADING_EDGE_OF_TOW, emit_subframe_event.trailing_edge_receiver_timestamp
        )
        # If we've never synchronized the receiver clock, synchronize it now.
        # As an initial estimate of our clock offset, set our clock to be the GPS system time, plus a
        # typical (but currently unmeasured) transmission delay. This should give us a receiver time estimate
        # that's not massively different from reality (as transmission times are in the realm of 60-80ms).
        # This guess will then be refined when we carry out our position fix. The magnitude of error here
        # shouldn't be too bad, as GPS should be able to cope with clock bias upwards of several seconds.
        if self.receiver_clock_slide is None or True:
            # Account for our delay in processing this subframe
            # TODO(PT): It might be necessary to add 1ms to this, since the trailing edge excludes the next PRN
            self.receiver_clock_slide = satellite_time_of_week_in_seconds - emit_subframe_event.trailing_edge_receiver_timestamp
            # Why is the observed TOW a good half MS behind where it should be? The error shouldn't be that high
            # It seems like this condition should hold:
            # emit_subframe_event.trailing_edge_receiver_timestamp + self.receiver_clock_slide == satellite_time_of_week_in_seconds
            # The trailing edge of a PRN should be exactly equivalent to the leading edge of the next PRN

            # Satellite TOW: The timestamp at the start of the next subframe
            # Receiver timestamp: How many seconds of listening it took to find the start of this subframe
            # Slide: How much we add to the receiver timestamp to get to the start of the next subframe

            logging.info('**** Subframe timing!')
            logging.info(f'\tSat time of week              {satellite_time_of_week_in_seconds}')
            logging.info(f'\tReceiver time slide           {self.receiver_clock_slide}')
            logging.info(f'\tTimestamped subframe          {emit_subframe_event.receiver_timestamp}')
            logging.info(f'\tTrailing edge                 {emit_subframe_event.trailing_edge_receiver_timestamp}')
            # TODO(PT): Debug next, why is the timestamp at 4.09 but the trailing edge is at 10.089!
        logging.info(f'*** Subframe for {satellite_id} at SV time {satellite_time_of_week_in_seconds}, Rx {self.receiver_clock_slide + emit_subframe_event.trailing_edge_receiver_timestamp}')
        # TODO(PT): Are we emitting subframes at 'random' times unrelated to when the PRNs tick in, due to
        # queueing in the subsystems?
        # Perhaps we need to work with the 'receiver timestamp' stamped in the subframe event.
        # There can be a 'receivr ticks since startup' that represents the receiver true passage of time, and another
        # 'receiver time slide' that's informed by the message HOW. That way we can compute the receiver time for a
        # given subframe on-the-fly

        # If we have enough parameters to know the satellite's time, store it
        # if orbital_params_for_this_satellite.is_parameter_set(OrbitalParameterType.WEEK_NUMBER):
        #    gps_week_number = orbital_params_for_this_satellite.week_number
        #    gps_satellite_time = (gps_week_number * SECONDS_PER_WEEK) + satellite_time_of_week_in_seconds

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

        if self._can_interrogate_precise_timings_for_satellite(satellite_id):
            logging.info(f'*** Received a subframe at {satellite_time_of_week_in_seconds}')

        return events_to_return

    def _process_subframe1(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe1) -> None:
        orbital_parameters.set_parameter(OrbitalParameterType.WEEK_NUMBER, subframe.week_num)
        orbital_parameters.set_parameter(OrbitalParameterType.A_F0, subframe.a_f0)
        orbital_parameters.set_parameter(OrbitalParameterType.A_F1, subframe.a_f1)
        orbital_parameters.set_parameter(OrbitalParameterType.A_F2, subframe.a_f2)
        orbital_parameters.set_parameter(OrbitalParameterType.T_OC, subframe.t_oc)
        orbital_parameters.set_parameter(OrbitalParameterType.ESTIMATED_GROUP_DELAY_DIFFERENTIAL, subframe.estimated_group_delay_differential)

    def _process_subframe2(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe2) -> None:
        orbital_parameters.set_parameter(
            OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME, subframe.mean_anomaly_at_reference_time * _PI
        )
        orbital_parameters.set_parameter(OrbitalParameterType.ECCENTRICITY, subframe.eccentricity)
        # The satellite transmits the square root of the semi-major axis, so square it now.
        orbital_parameters.set_parameter(OrbitalParameterType.SQRT_SEMI_MAJOR_AXIS, subframe.sqrt_semi_major_axis)
        orbital_parameters.set_parameter(OrbitalParameterType.SEMI_MAJOR_AXIS, math.pow(subframe.sqrt_semi_major_axis, 2))
        orbital_parameters.set_parameter(
            OrbitalParameterType.MEAN_MOTION_DIFFERENCE, subframe.mean_motion_difference_from_computed_value * _PI
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
        orbital_parameters.set_parameter(OrbitalParameterType.INCLINATION, subframe.inclination_angle * _PI)
        orbital_parameters.set_parameter(OrbitalParameterType.ARGUMENT_OF_PERIGEE, subframe.argument_of_perigee * _PI)
        orbital_parameters.set_parameter(
            OrbitalParameterType.LONGITUDE_OF_ASCENDING_NODE, subframe.longitude_of_ascending_node * _PI
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_INCLINATION_ANGLE_COS, subframe.correction_to_inclination_angle_cos
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_INCLINATION_ANGLE_SIN, subframe.correction_to_inclination_angle_sin
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.RATE_OF_RIGHT_ASCENSION, subframe.rate_of_right_ascension * _PI
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.RATE_OF_INCLINATION_ANGLE, subframe.rate_of_inclination_angle * _PI
        )
        orbital_parameters.set_parameter(
            OrbitalParameterType.CORRECTION_TO_ORBITAL_RADIUS_COS, subframe.correction_to_orbital_radius_cos
        )

    def _process_subframe4(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe4) -> None:
        pass

    def _process_subframe5(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe5) -> None:
        pass
