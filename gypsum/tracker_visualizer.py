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
        # Enable interactive mode if not already done
        if not plt.isinteractive():
            plt.ion()
            plt.autoscale(enable=True)

        self.visualizer_figure = plt.figure(figsize=(9, 6))
        self.visualizer_figure.suptitle(f"Satellite #{satellite_id.id} Tracker Dashboard")
        grid_spec = plt.GridSpec(3, 3, figure=self.visualizer_figure)

        self.freq_ax = self.visualizer_figure.add_subplot(grid_spec[0], title="Beat Frequency (Hz)")
        self.constellation_ax = self.visualizer_figure.add_subplot(grid_spec[1], title="IQ Constellation")
        self.samples_ax = self.visualizer_figure.add_subplot(grid_spec[2], title="Samples")
        self.phase_errors_ax = self.visualizer_figure.add_subplot(grid_spec[3], title="Carrier Phase Error")
        self.i_ax = self.visualizer_figure.add_subplot(grid_spec[4], title="I")
        self.q_ax = self.visualizer_figure.add_subplot(grid_spec[5], title="Q")
        self.iq_angle_ax = self.visualizer_figure.add_subplot(grid_spec[6], title="IQ Angle")
        self.carrier_phase_ax = self.visualizer_figure.add_subplot(grid_spec[6], title="Carrier Phase")
        self.visualizer_figure.show()

        self._timestamp_of_last_dashboard_update = 0
        self._timestamp_of_last_graph_reset = 0

    def step(self, seconds_since_start: Seconds, current_tracking_params: GpsSatelliteTrackingParameters) -> None:
        if seconds_since_start - self._timestamp_of_last_dashboard_update < _UPDATE_PERIOD:
            # It hasn't been long enough since our last GUI update
            return

        if seconds_since_start - self._timestamp_of_last_graph_reset >= _RESET_DISPLAYED_DATA_PERIOD:
            self._timestamp_of_last_graph_reset = seconds_since_start

            # Reset the graphs that clear periodically (so the old data doesn't clutter things up).
            self.constellation_ax.clear()

        # Time to update the GUI
        self._timestamp_of_last_dashboard_update = seconds_since_start

        locked_state = "Locked" if current_tracking_params.is_locked() else "Unlocked"
        last_few_phase_errors = current_tracking_params.carrier_wave_phase_errors[-250:]
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
        self._is = []

        self.q_ax.clear()
        #self.q_ax.plot(self._qs)
        self.q_ax.plot(np.imag(points))
        self._qs = []

        self.iq_angle_ax.clear()
        #self.iq_angle_ax.plot(self.iq_angles)
        self.iq_angles = []

        self.carrier_phase_ax.clear()
        #self.carrier_phase_ax.plot(self.carrier_phases)
        self.carrier_phase_ax.plot(params.carrier_wave_phases)
        self.carrier_phases = []

        self.carrier = []
        self.mixed = []

        self.phase_errors_ax.plot(params.carrier_wave_phase_errors)

        plt.pause(0.001)
