import logging

import math
import logging
import numpy as np

from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.tracker import GpsSatelliteTrackingParameters
from gypsum.utils import Seconds

import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


_UPDATE_PERIOD: Seconds = 1.0
_RESET_DISPLAYED_DATA_PERIOD: Seconds = 5.0


class GpsSatelliteTrackerVisualizer:
    def __init__(self, satellite_id: GpsSatelliteId) -> None:
        self._timestamp_of_last_dashboard_update = 0
        self._timestamp_of_last_graph_reset = 0

        # Enable interactive mode if not already done
        if not plt.isinteractive():
            plt.ion()
            plt.autoscale(enable=True)

        self.visualizer_figure = plt.figure(figsize=(11, 6))
        self.visualizer_figure.suptitle(f"Satellite #{satellite_id.id} Tracking Dashboard")
        self.grid_spec = plt.GridSpec(nrows=2, ncols=4, figure=self.visualizer_figure)

        self.freq_ax = self.visualizer_figure.add_subplot(self.grid_spec[0])
        self.constellation_ax = self.visualizer_figure.add_subplot(self.grid_spec[1])
        self.phase_errors_ax = self.visualizer_figure.add_subplot(self.grid_spec[2])
        self.i_ax = self.visualizer_figure.add_subplot(self.grid_spec[3])
        self.q_ax = self.visualizer_figure.add_subplot(self.grid_spec[4])
        self.iq_angle_ax = self.visualizer_figure.add_subplot(self.grid_spec[5])
        self.carrier_phase_ax = self.visualizer_figure.add_subplot(self.grid_spec[6])

        self._redraw_subplot_titles()

        # All done, request tight layout
        self.grid_spec.tight_layout(self.visualizer_figure)

    def _redraw_subplot_titles(self):
        """Unfortunately, plt.Axes.clear() also erases the subplot title.
        Therefore, every time we clear an axis, we have to redraw its title.
        """
        self.freq_ax.set_title("Beat Frequency (Hz)")
        self.constellation_ax.set_title("IQ Constellation")
        self.phase_errors_ax.set_title("Carrier Phase Error")
        self.i_ax.set_title("I Component")
        self.q_ax.set_title("Q Component")
        self.iq_angle_ax.set_title("IQ Angle (Rad)")
        self.carrier_phase_ax.set_title("Carrier Phase (Rad)")

    def step(self, seconds_since_start: Seconds, current_tracking_params: GpsSatelliteTrackingParameters) -> None:
        if seconds_since_start - self._timestamp_of_last_dashboard_update < _UPDATE_PERIOD:
            # It hasn't been long enough since our last GUI update
            return

        if seconds_since_start - self._timestamp_of_last_graph_reset >= _RESET_DISPLAYED_DATA_PERIOD:
            self._timestamp_of_last_graph_reset = seconds_since_start

            # Reset the graphs that clear periodically (so the old data doesn't clutter things up).
            self.constellation_ax.clear()
            self.phase_errors_ax.clear()

        # Time to update the GUI
        self._timestamp_of_last_dashboard_update = seconds_since_start

        locked_state = "Locked" if current_tracking_params.is_locked() else "Unlocked"
        last_few_phase_errors = list(current_tracking_params.carrier_wave_phase_errors)[-250:]
        variance = np.var(last_few_phase_errors)
        _logger.info(f'Seconds since start: {seconds_since_start} ({locked_state}), Variance {variance:.2f}')

        params = current_tracking_params
        self.freq_ax.plot(params.doppler_shifts[::10])

        points = np.array(params.correlation_peaks_rolling_buffer)
        points_on_left_pole = points[points.real < 0]
        points_on_right_pole = points[points.real >= 0]
        left_point = np.mean(points_on_left_pole)
        right_point = np.mean(points_on_right_pole)
        angle = 180 - (((np.arctan2(left_point.imag, left_point.real) / math.tau) * 360) % 180)
        rotation = angle
        if angle > 90:
            rotation = angle - 180
        _logger.info(f'Angle {angle:.2f} Rotation {rotation:.2f} Doppler {params.current_doppler_shift:.2f}')

        self.constellation_ax.scatter(np.real(points), np.imag(points))
        self.constellation_ax.scatter([left_point.real, right_point.real], [left_point.imag, right_point.imag])
        self.i_ax.clear()
        self.i_ax.plot(np.real(points))

        self.q_ax.clear()
        #self.q_ax.plot(self._qs)
        self.q_ax.plot(np.imag(points))

        self.iq_angle_ax.clear()
        self.iq_angle_ax.plot(params.correlation_peak_angles)
        #self.iq_angles = []

        self.carrier_phase_ax.clear()
        #self.carrier_phase_ax.plot(self.carrier_phases)
        self.carrier_phase_ax.plot(params.carrier_wave_phases)
        #self.carrier_phases = []

        self.phase_errors_ax.plot(params.carrier_wave_phase_errors)

        # We've just erased some of our axes titles via plt.Axes.clear(), so redraw them.
        self._redraw_subplot_titles()
        plt.pause(0.001)
