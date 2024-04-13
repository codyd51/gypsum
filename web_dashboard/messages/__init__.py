import pydantic

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.world_model import OrbitalParameters


class GpsReceiverState(pydantic.BaseModel):
    receiver_timestamp: ReceiverTimestampSeconds
    dashboard_figures: list[str]
    tracked_satellite_count: int
    processed_subframe_count: int
    satellite_ids_to_orbital_parameters: dict[GpsSatelliteId, OrbitalParameters]
    # PT: Should be inferred from the pipelines?..
    tracked_satellite_ids: list[GpsSatelliteId]
    satellite_ids_eligible_for_acquisition: list[GpsSatelliteId]
    satellite_ids_ineligible_for_acquisition: list[GpsSatelliteId]
    position_fixes: list[str]


class SetCurrentReceiverStateRequest(pydantic.BaseModel):
    current_state: GpsReceiverState
