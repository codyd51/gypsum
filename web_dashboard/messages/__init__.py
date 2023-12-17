import pydantic

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
from gypsum.gps_ca_prn_codes import GpsSatelliteId


class GpsReceiverState(pydantic.BaseModel):
    receiver_timestamp: ReceiverTimestampSeconds
    satellite_ids_eligible_for_acquisition: list[GpsSatelliteId]
    dashboard_figures: list[str]


class SetCurrentReceiverStateRequest(pydantic.BaseModel):
    current_state: GpsReceiverState
