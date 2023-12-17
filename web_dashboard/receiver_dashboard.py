import json
import logging
from dataclasses import asdict
import datetime
from dataclasses import dataclass

import falcon

from web_dashboard.messages import GpsReceiverState
from web_dashboard.messages import SetCurrentReceiverStateRequest
from web_dashboard.templates import JINJA_ENVIRONMENT
from web_dashboard.templates import TemplateContext

_logger = logging.getLogger(__name__)


@dataclass
class DashboardContext:
    """Context common to all templates"""
    generated_at: str
    state: GpsReceiverState

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)


class GpsReceiverDashboard:
    def __init__(self):
        self.falcon_app = falcon.App()
        self.resp = ""
        self.last_seen_gps_state = None

    def on_get(self, _request: falcon.Request, response: falcon.Response) -> None:
        _logger.info(f"Serving dashboard page {self.last_seen_gps_state}...")

        if self.last_seen_gps_state is None:
            response.text = "No data from receiver yet"
            return

        now = datetime.datetime.utcnow()
        context = DashboardContext(
            generated_at=now.strftime("%B %d, %Y at %H:%M"),
            state=self.last_seen_gps_state,
        )
        print(context)
        print(context.state)
        response.content_type = falcon.MEDIA_HTML
        response.text = JINJA_ENVIRONMENT.get_template("dashboard.html.jinja2").render(asdict(context))

    def on_post(self, request: falcon.Request, response: falcon.Response) -> None:
        _logger.info("Handling update from receiver...")

        raw_post_data = json.loads(request.media)
        update = SetCurrentReceiverStateRequest(**raw_post_data)
        self.last_seen_gps_state = update.current_state
