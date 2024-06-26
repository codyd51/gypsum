"""PT: This module is meant to run as a separate process from the main receiver application.
This allows the receiver to run without worrying about yielding control flow to the webserver.
The receiver will periodically POST updates to this process, which will display it.
"""
import logging
import os
from pathlib import Path

import falcon

from web_dashboard.receiver_dashboard import GpsReceiverDashboard
from web_dashboard.receiver_dashboard import GpsReceiverDashboardStateProvider
from web_dashboard.receiver_dashboard import GpsReceiverDashboardTrackerVisualizers
from web_dashboard.receiver_dashboard import GpsReceiverSatelliteInfos
from web_dashboard.receiver_dashboard import GpsReceiverStats

_logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


_APP = falcon.App()
# WSGI expects a global called `application`
application = _APP

# PT: Global state in lieu of a database...
_STATE_PROVIDER = GpsReceiverDashboardStateProvider()


def main():
    _logger.info(f'Starting up under gunicorn...')
    _APP.add_static_route("/static", Path(__file__).parent / "static")
    _APP.add_route("/", GpsReceiverDashboard(_STATE_PROVIDER))
    _APP.add_route("/tracker_visualizers", GpsReceiverDashboardTrackerVisualizers(_STATE_PROVIDER))
    _APP.add_route("/receiver_stats", GpsReceiverStats(_STATE_PROVIDER))
    _APP.add_route("/satellite_infos", GpsReceiverSatelliteInfos(_STATE_PROVIDER))


# Note that __name__ != __main__ here, since we're running via gunicorn
_is_running_under_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
if _is_running_under_gunicorn:
    main()
