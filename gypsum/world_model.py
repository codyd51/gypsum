from collections import defaultdict
from enum import Enum
from enum import auto
from typing import Type
from typing import cast

from gypsum.events import Event
from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.navigation_message_decoder import EmitSubframeEvent
from gypsum.navigation_message_parser import GpsSubframeId
from gypsum.navigation_message_parser import Meters
from gypsum.navigation_message_parser import NavigationMessageSubframe1
from gypsum.navigation_message_parser import NavigationMessageSubframe2
from gypsum.navigation_message_parser import NavigationMessageSubframe3
from gypsum.navigation_message_parser import NavigationMessageSubframe4
from gypsum.navigation_message_parser import NavigationMessageSubframe5
from gypsum.navigation_message_parser import SemiCircles


class OrbitalParameterType(Enum):
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

    @property
    def unit(self) -> Type:
        return {
            self.SEMI_MAJOR_AXIS: Meters,
            self.ECCENTRICITY: float,
            self.INCLINATION: SemiCircles,
            self.LONGITUDE_OF_ASCENDING_NODE: SemiCircles,
            self.ARGUMENT_OF_PERIGEE: SemiCircles,
            self.MEAN_ANOMALY_AT_REFERENCE_TIME: SemiCircles,
        }[self]


_OrbitalParameterType = Meters | float | SemiCircles


class OrbitalParameters:
    """Tracks a 'set' of orbital parameters for a classical 2-body orbit."""
    def __init__(self) -> None:
        self.parameter_type_to_value = {t: None for t in OrbitalParameterType}

    def is_complete(self) -> bool:
        """Returns whether we have a 'full set' of parameters to describe a body's orbit."""
        return not any(x is None for x in self.parameter_type_to_value.values())

    # PT: For caller convenience, provide infallible accessors to orbital parameters

    def _get_parameter_infallibly(self, param_type: OrbitalParameterType) -> _OrbitalParameterType:
        maybe_param = self.parameter_type_to_value[param_type]
        if maybe_param is None:
            raise RuntimeError(f'Expected {param_type.name} to be available')
        return maybe_param

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
        self.satellite_ids_to_orbital_parameters: dict[GpsSatelliteId, OrbitalParameters] = defaultdict(OrbitalParameters)

    def handle_subframe_emitted(self, satellite_id: GpsSatelliteId, emit_subframe_event: EmitSubframeEvent) -> list[Event]:
        events_to_return = []
        subframe = emit_subframe_event.subframe
        subframe_id = subframe.subframe_id

        orbital_params_for_this_satellite = self.satellite_ids_to_orbital_parameters[satellite_id]
        # Keep track of whether we already had all the orbital parameters for this satellite, so we know whether
        # we've just completed a full set.
        were_orbit_params_already_complete = orbital_params_for_this_satellite.is_complete()

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

        return events_to_return

    def _process_subframe1(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe1) -> None:
        pass

    def _process_subframe2(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe2) -> None:
        orbital_parameters.parameter_type_to_value[OrbitalParameterType.MEAN_ANOMALY_AT_REFERENCE_TIME] = subframe.mean_anomaly_at_reference_time
        orbital_parameters.parameter_type_to_value[OrbitalParameterType.ECCENTRICITY] = subframe.eccentricity
        # The satellite transmits the square root of the semi-major axis, so square it now.
        orbital_parameters.parameter_type_to_value[OrbitalParameterType.SEMI_MAJOR_AXIS] = subframe.sqrt_semi_major_axis ** 2

    def _process_subframe3(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe3) -> None:
        orbital_parameters.parameter_type_to_value[OrbitalParameterType.INCLINATION] = subframe.inclination_angle
        orbital_parameters.parameter_type_to_value[OrbitalParameterType.ARGUMENT_OF_PERIGEE] = subframe.argument_of_perigee
        orbital_parameters.parameter_type_to_value[OrbitalParameterType.LONGITUDE_OF_ASCENDING_NODE] = subframe.longitude_of_ascending_node

    def _process_subframe4(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe4) -> None:
        pass

    def _process_subframe5(self, orbital_parameters: OrbitalParameters, subframe: NavigationMessageSubframe5) -> None:
        pass
