import json
import logging
from dataclasses import asdict
import datetime
from dataclasses import dataclass

import falcon

from web_dashboard.messages import GpsReceiverState
from web_dashboard.messages import SetCurrentReceiverStateRequest
from web_dashboard.templates import JINJA_ENVIRONMENT

_logger = logging.getLogger(__name__)


@dataclass
class DashboardContext:
    """Context common to all templates"""
    generated_at: str
    state: GpsReceiverState

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)


@dataclass
class GpsReceiverDashboardStateProvider:
    last_seen_gps_state: GpsReceiverState | None = None

    def handle_state_update(self, state: GpsReceiverState) -> None:
        self.last_seen_gps_state = state
        # Always insert a first entry to the position fixes
        self.last_seen_gps_state.position_fixes.insert(0, "<Awaiting first position fix>")

    def get_state(self) -> GpsReceiverState | None:
        return self.last_seen_gps_state


class GpsReceiverDashboardResource:
    def __init__(self, state_provider: GpsReceiverDashboardStateProvider):
        self.state_provider = state_provider

    def on_get(self, request: falcon.Request, response: falcon.Response) -> None:
        state = self.state_provider.get_state()
        if state is None:
            response.text = "No data from receiver yet"
            return

        self.handle_on_get(state, request, response)

    def handle_on_get(self, state: GpsReceiverState, _request: falcon.Request, response: falcon.Response) -> None:
        ...


class GpsReceiverDashboard(GpsReceiverDashboardResource):
    def handle_on_get(self, state: GpsReceiverState, _request: falcon.Request, response: falcon.Response) -> None:
        now = datetime.datetime.utcnow()
        context = DashboardContext(
            generated_at=now.strftime("%B %d, %Y at %H:%M"),
            state=state,
        )
        response.content_type = falcon.MEDIA_HTML
        response.text = JINJA_ENVIRONMENT.get_template("dashboard.html.jinja2").render(asdict(context))

    def on_post(self, request: falcon.Request, _response: falcon.Response) -> None:
        # _logger.info("Handling update from receiver...")

        raw_post_data = json.loads(request.media)
        update = SetCurrentReceiverStateRequest(**raw_post_data)

        # Push the update to the state storage
        self.state_provider.handle_state_update(update.current_state)


class GpsReceiverDashboardTrackerVisualizers(GpsReceiverDashboardResource):
    def handle_on_get(self, state: GpsReceiverState, _request: falcon.Request, response: falcon.Response) -> None:
        now = datetime.datetime.utcnow()
        context = DashboardContext(
            generated_at=now.strftime("%H:%M:%S"),
            state=state,
        )
        response.content_type = falcon.MEDIA_HTML
        response.text = JINJA_ENVIRONMENT.get_template("tracker_visualizers.html.jinja2").render(asdict(context))


class GpsReceiverStats(GpsReceiverDashboardResource):
    def handle_on_get(self, state: GpsReceiverState, _request: falcon.Request, response: falcon.Response) -> None:
        now = datetime.datetime.utcnow()
        context = DashboardContext(
            generated_at=now.strftime("%H:%M:%S"),
            state=state,
        )
        response.content_type = falcon.MEDIA_HTML
        response.text = JINJA_ENVIRONMENT.get_template("receiver_stats.html.jinja2").render(asdict(context))


class GpsReceiverSatelliteInfos(GpsReceiverDashboardResource):
    def handle_on_get(self, state: GpsReceiverState, _request: falcon.Request, response: falcon.Response) -> None:
        now = datetime.datetime.utcnow()
        context = DashboardContext(
            generated_at=now.strftime("%H:%M:%S"),
            state=state,
        )
        response.content_type = falcon.MEDIA_HTML
        response.text = JINJA_ENVIRONMENT.get_template("satellite_infos.html.jinja2").render(asdict(context))
