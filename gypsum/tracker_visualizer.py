from enum import Enum
from enum import auto

import math
import logging
import numpy as np
from matplotlib.axes import Axes

from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.navigation_bit_intergrator import NavigationBitIntegratorHistory
from gypsum.navigation_message_decoder import NavigationMessageDecoderHistory
from gypsum.tracker import BitValue
from gypsum.tracker import GpsSatelliteTrackingParameters
from gypsum.units import Seconds

import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


_UPDATE_PERIOD: Seconds = 1.0
_RESET_DISPLAYED_DATA_PERIOD: Seconds = 5.0


class GraphTypeEnum(Enum):
    # Note that the ordering of this enum defines the layout of the graphs in the dashboard
    DOPPLER_SHIFT = auto()
    IQ_CONSTELLATION = auto()
    CARRIER_PHASE_ERROR = auto()
    I_COMPONENT = auto()
    Q_COMPONENT = auto()
    IQ_ANGLE = auto()
    CARRIER_PHASE = auto()
    PSEUDOSYMBOLS = auto()
    BIT_PHASE = auto()
    SUBFRAME_PHASE = auto()
    TRACK_DURATION = auto()
    BIT_HEALTH = auto()

    @property
    def presentation_name(self) -> str:
        return {
            GraphTypeEnum.DOPPLER_SHIFT: "Beat Frequency (Hz)",
            GraphTypeEnum.IQ_CONSTELLATION: "IQ Constellation",
            GraphTypeEnum.CARRIER_PHASE_ERROR: "Carrier Phase Error",
            GraphTypeEnum.I_COMPONENT: "I Component",
            GraphTypeEnum.Q_COMPONENT: "Q Component",
            GraphTypeEnum.IQ_ANGLE: "IQ Angle (Rad)",
            GraphTypeEnum.CARRIER_PHASE: "Carrier Phase (Rad)",
            GraphTypeEnum.PSEUDOSYMBOLS: "Pseudosymbols",
            GraphTypeEnum.BIT_PHASE: "Bit Phase (PSymbols)",
            GraphTypeEnum.SUBFRAME_PHASE: "Subframe Phase (Bits)",
            GraphTypeEnum.TRACK_DURATION: "Track Duration (Sec)",
            GraphTypeEnum.BIT_HEALTH: "Bit Health",
        }[self]


class GpsSatelliteTrackerVisualizer:
    def __init__(self, satellite_id: GpsSatelliteId) -> None:
        self._timestamp_of_last_dashboard_update = 0
        self._timestamp_of_last_graph_reset = 0

        # Enable interactive mode if not already done
        if not plt.isinteractive():
            plt.ion()

        self.visualizer_figure = plt.figure(figsize=(11, 6))
        self.visualizer_figure.suptitle(f"Satellite #{satellite_id.id} Tracking Dashboard")
        self.grid_spec = plt.GridSpec(nrows=3, ncols=4, figure=self.visualizer_figure)

        grid_spec_idx_iterator = iter(range(len(GraphTypeEnum)))
        self.graph_type_to_graphs = {
            t: self.visualizer_figure.add_subplot(self.grid_spec[next(grid_spec_idx_iterator)])
            for t in GraphTypeEnum
        }

        self._redraw_subplot_titles()

        # All done, request tight layout
        self.grid_spec.tight_layout(self.visualizer_figure)

    def _redraw_subplot_titles(self):
        """Unfortunately, plt.Axes.clear() also erases the subplot title.
        Therefore, every time we clear an axis, we have to redraw its title.
        """
        for graph_type, graph in self.graph_type_to_graphs.items():
            graph.set_title(graph_type.presentation_name)

    def graph_for_type(self, t: GraphTypeEnum) -> Axes:
        return self.graph_type_to_graphs[t]

    def step(
        self,
        seconds_since_start: Seconds,
        current_tracking_params: GpsSatelliteTrackingParameters,
        bit_integrator_history: NavigationBitIntegratorHistory,
        navigation_message_decoder_history: NavigationMessageDecoderHistory,
    ) -> None:
        if seconds_since_start - self._timestamp_of_last_dashboard_update < _UPDATE_PERIOD:
            # It hasn't been long enough since our last GUI update
            return

        if seconds_since_start - self._timestamp_of_last_graph_reset >= _RESET_DISPLAYED_DATA_PERIOD:
            self._timestamp_of_last_graph_reset = seconds_since_start

            # Reset the graphs that clear periodically (so the old data doesn't clutter things up).
            self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION).clear()
            self.graph_for_type(GraphTypeEnum.CARRIER_PHASE_ERROR).clear()

        # Time to update the GUI
        self._timestamp_of_last_dashboard_update = seconds_since_start

        locked_state = "Locked" if current_tracking_params.is_locked() else "Unlocked"
        last_few_phase_errors = list(current_tracking_params.carrier_wave_phase_errors)[-250:]
        variance = np.var(last_few_phase_errors) if len(last_few_phase_errors) >= 2 else 0
        _logger.info(f'Seconds since start: {seconds_since_start} ({locked_state}), Variance {variance:.2f}')

        params = current_tracking_params
        self.graph_for_type(GraphTypeEnum.DOPPLER_SHIFT).plot(params.doppler_shifts[::10])

        points = np.array(params.correlation_peaks_rolling_buffer)
        self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION).scatter(np.real(points), np.imag(points))

        if len(points) > 2:
            # Draw the 'average' / mean point of each pole
            points_on_left_pole = points[points.real < 0]
            points_on_right_pole = points[points.real >= 0]

            left_point = np.mean(points_on_left_pole) if len(points_on_left_pole) >= 2 else 0
            right_point = np.mean(points_on_right_pole) if len(points_on_right_pole) >= 2 else 0

            angle = 180 - (((np.arctan2(left_point.imag, left_point.real) / math.tau) * 360) % 180)
            rotation = angle
            if angle > 90:
                rotation = angle - 180
            _logger.info(f'Angle {angle:.2f} Rotation {rotation:.2f} Doppler {params.current_doppler_shift:.2f}')
            self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION).scatter([left_point.real, right_point.real], [left_point.imag, right_point.imag])

        self.graph_for_type(GraphTypeEnum.I_COMPONENT).clear()
        self.graph_for_type(GraphTypeEnum.I_COMPONENT).plot(np.real(points))

        self.graph_for_type(GraphTypeEnum.Q_COMPONENT).clear()
        self.graph_for_type(GraphTypeEnum.Q_COMPONENT).plot(np.imag(points))

        self.graph_for_type(GraphTypeEnum.IQ_ANGLE).clear()
        self.graph_for_type(GraphTypeEnum.IQ_ANGLE).plot(params.correlation_peak_angles)

        self.graph_for_type(GraphTypeEnum.CARRIER_PHASE).clear()
        self.graph_for_type(GraphTypeEnum.CARRIER_PHASE).plot(params.carrier_wave_phases)

        self.graph_for_type(GraphTypeEnum.CARRIER_PHASE_ERROR).clear()
        self.graph_for_type(GraphTypeEnum.CARRIER_PHASE_ERROR).plot(params.carrier_wave_phase_errors)

        self.graph_for_type(GraphTypeEnum.PSEUDOSYMBOLS).clear()
        self.graph_for_type(GraphTypeEnum.PSEUDOSYMBOLS).plot(
            [x.pseudosymbol.as_val() for x in bit_integrator_history.last_seen_pseudosymbols]
        )

        self.graph_for_type(GraphTypeEnum.BIT_PHASE).clear()
        lock_state = f"Unlocked" if not bit_integrator_history.is_bit_phase_locked else "Locked"
        bit_phase_status_message = f"{bit_integrator_history.determined_bit_phase} ({lock_state})"
        self.graph_for_type(GraphTypeEnum.BIT_PHASE).text(0.0, 0.4, bit_phase_status_message, fontsize=20)

        self.graph_for_type(GraphTypeEnum.SUBFRAME_PHASE).clear()
        if navigation_message_decoder_history.determined_subframe_phase is None:
            subframe_phase_status_message = f"Unknown"
        else:
            subframe_phase_status_message = f"{navigation_message_decoder_history.determined_subframe_phase}"
        self.graph_for_type(GraphTypeEnum.SUBFRAME_PHASE).text(0.0, 0.4, subframe_phase_status_message, fontsize=20)

        self.graph_for_type(GraphTypeEnum.TRACK_DURATION).clear()
        # TODO(PT): This is the offset from startup, not track start...
        track_duration_text = f'{seconds_since_start:.2f}'
        self.graph_for_type(GraphTypeEnum.TRACK_DURATION).text(0.0, 0.4, track_duration_text, fontsize=20)

        self.graph_for_type(GraphTypeEnum.BIT_HEALTH).clear()
        # Bit health represents the proportion of the previous period of bits that were resolved with confidence
        if len(bit_integrator_history.last_emitted_bits) == 0:
            bit_health_text = "No bits seen yet"
        else:
            bit_health = (len([x for x in bit_integrator_history.last_emitted_bits if x != BitValue.UNKNOWN]) / len(bit_integrator_history.last_emitted_bits)) * 100
            bit_health_text = f"{bit_health}%"
        self.graph_for_type(GraphTypeEnum.BIT_HEALTH).text(0.0, 0.4, bit_health_text, fontsize=20)

        # We've just erased some of our axes titles via plt.Axes.clear(), so redraw them.
        self._redraw_subplot_titles()
        plt.pause(0.001)
