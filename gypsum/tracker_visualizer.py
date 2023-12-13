from dataclasses import dataclass
from enum import Enum
from enum import auto

import math
import logging
import numpy as np
from matplotlib.axes import Axes

from gypsum.constants import PSEUDOSYMBOLS_PER_NAVIGATION_BIT
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


@dataclass
class GraphAttributes:
    display_axes: bool = False
    is_text_only: bool = False
    # The below fields are only relevant if this is a 'text-only' graph
    # Expressed as a hex RGB color, prefixed with '#'
    background_color: str | None = None

    @staticmethod
    def spacer() -> 'GraphAttributes':
        return GraphAttributes(is_text_only=True, display_axes=False)

    @staticmethod
    def text(background_color: str) -> 'GraphAttributes':
        return GraphAttributes(is_text_only=True, display_axes=False, background_color=background_color)

    @staticmethod
    def with_axes() -> 'GraphAttributes':
        return GraphAttributes(display_axes=True, is_text_only=False)

    @staticmethod
    def without_axes() -> 'GraphAttributes':
        return GraphAttributes(display_axes=False, is_text_only=False)


class GraphTypeEnum(Enum):
    DOPPLER_SHIFT = auto()
    CARRIER_PHASE = auto()
    BIT_PHASE = auto()
    SUBFRAME_PHASE = auto()

    Q_COMPONENT = auto()
    CARRIER_PHASE_ERROR = auto()
    TRACK_DURATION = auto()
    BIT_HEALTH = auto()

    I_COMPONENT = auto()
    IQ_CONSTELLATION = auto()
    EMITTED_SUBFRAMES = auto()
    FAILED_BITS = auto()

    PSEUDOSYMBOLS = auto()
    IQ_ANGLE = auto()
    BITS = auto()
    PRN_CODE_PHASE = auto()
    CORRELATION_STRENGTH = auto()
    @property
    def attributes(self) -> GraphAttributes:
        return {
            GraphTypeEnum.CARRIER_PHASE: GraphAttributes.with_axes(),
            GraphTypeEnum.DOPPLER_SHIFT: GraphAttributes.with_axes(),
            GraphTypeEnum.BITS: GraphAttributes.without_axes(),
            GraphTypeEnum.CARRIER_PHASE_ERROR: GraphAttributes.without_axes(),
            GraphTypeEnum.IQ_ANGLE: GraphAttributes.without_axes(),
            GraphTypeEnum.IQ_CONSTELLATION: GraphAttributes.without_axes(),
            GraphTypeEnum.I_COMPONENT: GraphAttributes.without_axes(),
            GraphTypeEnum.PSEUDOSYMBOLS: GraphAttributes.without_axes(),
            GraphTypeEnum.Q_COMPONENT: GraphAttributes.without_axes(),
            GraphTypeEnum.BIT_HEALTH: GraphAttributes.text(background_color="#ffe7a6"),
            GraphTypeEnum.BIT_PHASE: GraphAttributes.text(background_color="#acdffc"),
            GraphTypeEnum.CORRELATION_STRENGTH: GraphAttributes.text(background_color="#ffe7a6"),
            GraphTypeEnum.EMITTED_SUBFRAMES: GraphAttributes.text(background_color="#c4fcac"),
            GraphTypeEnum.FAILED_BITS: GraphAttributes.text(background_color="#ffe7a6"),
            GraphTypeEnum.PRN_CODE_PHASE: GraphAttributes.text(background_color="#acdffc"),
            GraphTypeEnum.SUBFRAME_PHASE: GraphAttributes.text(background_color="#acdffc"),
            GraphTypeEnum.TRACK_DURATION: GraphAttributes.text(background_color="#c4fcac"),
            GraphTypeEnum.SPACER1: GraphAttributes.spacer(),
            GraphTypeEnum.SPACER2: GraphAttributes.spacer(),
            GraphTypeEnum.SPACER3: GraphAttributes.spacer(),
        }[self]

    @classmethod
    def layout_order(cls) -> list[list['GraphTypeEnum']]:
        """Defines the ordering of the graphs in the dashboard"""
        return [
            [
                cls.DOPPLER_SHIFT,
                cls.CARRIER_PHASE,
                cls.BIT_HEALTH,
                cls.FAILED_BITS,
            ],
            [
                cls.I_COMPONENT,
                cls.IQ_CONSTELLATION,
                cls.CORRELATION_STRENGTH,
                cls.SPACER1,
            ],
            [
                cls.PSEUDOSYMBOLS,
                cls.IQ_ANGLE,
                cls.EMITTED_SUBFRAMES,
                cls.TRACK_DURATION,
            ],
            [
                cls.BITS,
                cls.SPACER2,
                cls.BIT_PHASE,
                cls.PRN_CODE_PHASE,
            ],
            [
                cls.Q_COMPONENT,
                cls.CARRIER_PHASE_ERROR,
                cls.SUBFRAME_PHASE,
                cls.SPACER3,
            ],
        ]

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
            GraphTypeEnum.BITS: "Bits",
            GraphTypeEnum.BIT_PHASE: "Bit Phase (PSymbols)",
            GraphTypeEnum.SUBFRAME_PHASE: "Subframe Phase (Bits)",
            GraphTypeEnum.TRACK_DURATION: "Track Duration (Sec)",
            GraphTypeEnum.BIT_HEALTH: "Bit Health",
            GraphTypeEnum.SPACER: "",
            GraphTypeEnum.EMITTED_SUBFRAMES: "Emitted Subframes",
            GraphTypeEnum.FAILED_BITS: "Failed Bits",
        }[self]


class GpsSatelliteTrackerVisualizer:
    def __init__(self, satellite_id: GpsSatelliteId, should_display: bool = True) -> None:
        self.should_display = should_display
        self._timestamp_of_last_dashboard_update = 0
        self._timestamp_of_last_graph_reset = 0

        if not should_display:
            return

        # Enable interactive mode if not already done
        if not plt.isinteractive():
            plt.ion()

        self.visualizer_figure = plt.figure(figsize=(11, 6))
        self.visualizer_figure.suptitle(f"Satellite #{satellite_id.id} Tracking Dashboard")
        self.grid_spec = plt.GridSpec(nrows=5, ncols=4, figure=self.visualizer_figure)

        grid_spec_idx_iterator = iter(range(len(GraphTypeEnum)))
        # Initialize the graphs in the order specified
        self.graph_type_to_graphs = {}
        layout_order = GraphTypeEnum.layout_order()
        for row in layout_order:
            for graph_type in row:
                self.graph_type_to_graphs[graph_type] = self.visualizer_figure.add_subplot(self.grid_spec[next(grid_spec_idx_iterator)])

        self._redraw_subplot_titles()

        # All done, request tight layout
        self.grid_spec.tight_layout(self.visualizer_figure)

    def _redraw_subplot_titles(self):
        """Unfortunately, plt.Axes.clear() also erases the subplot title.
        Therefore, every time we clear an axis, we have to redraw its title.
        """
        for graph_type, graph in self.graph_type_to_graphs.items():
            graph.set_title(graph_type.presentation_name)

        # Certain graph types are text-only, and we don't need the ticks/frame that pyplot provides by default.
        graphs_with_no_frames = [x for x in GraphTypeEnum if x.attributes.is_text_only]
        for graph_type in graphs_with_no_frames:
            self.graph_for_type(graph_type).axis('off')

        # Certain graph types aren't worth showing the axis labels, as the magnitudes aren't too important and they
        # clutter the UI.
        graphs_without_axes_labels = [x for x in GraphTypeEnum if not x.attributes.display_axes]
        for graph_type in graphs_without_axes_labels:
            self.graph_for_type(graph_type).get_xaxis().set_visible(False)
            self.graph_for_type(graph_type).get_yaxis().set_visible(False)

    def graph_for_type(self, t: GraphTypeEnum) -> Axes:
        return self.graph_type_to_graphs[t]

    def draw_text(self, t: GraphTypeEnum, s: str):
        background_color = t.attributes.background_color
        if background_color is None:
            raise ValueError(f'No background color set for {t}')

        self.graph_for_type(t).text(
            0.5,
            0.25,
            s,
            fontsize=20,
            bbox={
                'edgecolor': '#000000',
                "facecolor": background_color,
                "boxstyle": 'round',
                "pad": 0.2
            },
            ha="center"
        )

    def step(
        self,
        seconds_since_start: Seconds,
        current_tracking_params: GpsSatelliteTrackingParameters,
        bit_integrator_history: NavigationBitIntegratorHistory,
        navigation_message_decoder_history: NavigationMessageDecoderHistory,
    ) -> None:
        if not self.should_display:
            return

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

        # locked_state = "Locked" if current_tracking_params.is_locked() else "Unlocked"
        # last_few_phase_errors = list(current_tracking_params.carrier_wave_phase_errors)[-250:]
        # variance = np.var(last_few_phase_errors) if len(last_few_phase_errors) >= 2 else 0
        # _logger.info(f'Seconds since start: {seconds_since_start} ({locked_state}), Variance {variance:.2f}')

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
            # _logger.info(f'Angle {angle:.2f} Rotation {rotation:.2f} Doppler {params.current_doppler_shift:.2f}')
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

        self.graph_for_type(GraphTypeEnum.BITS).clear()
        bits_as_runs = []
        for bit in bit_integrator_history.last_emitted_bits:
            bits_as_runs.extend([0.5 if bit == BitValue.UNKNOWN else bit.as_val() for _ in range(PSEUDOSYMBOLS_PER_NAVIGATION_BIT)])
        self.graph_for_type(GraphTypeEnum.BITS).plot(bits_as_runs)

        self.graph_for_type(GraphTypeEnum.BIT_PHASE).clear()
        bit_phase_status_message = f"{bit_integrator_history.determined_bit_phase}"
        self.draw_text(GraphTypeEnum.BIT_PHASE, bit_phase_status_message)

        self.graph_for_type(GraphTypeEnum.SUBFRAME_PHASE).clear()
        if navigation_message_decoder_history.determined_subframe_phase is None:
            subframe_phase_status_message = f"Unknown"
        else:
            subframe_phase_status_message = f"{navigation_message_decoder_history.determined_subframe_phase}"
        self.draw_text(GraphTypeEnum.SUBFRAME_PHASE, subframe_phase_status_message)

        self.graph_for_type(GraphTypeEnum.TRACK_DURATION).clear()
        # TODO(PT): This is the offset from startup, not track start...
        track_duration_text = f'{int(seconds_since_start)}'
        self.draw_text(GraphTypeEnum.TRACK_DURATION, track_duration_text)

        self.graph_for_type(GraphTypeEnum.BIT_HEALTH).clear()
        # Bit health represents the proportion of the previous period of bits that were resolved with confidence
        if len(bit_integrator_history.last_emitted_bits) == 0:
            bit_health_text = "No bits seen yet"
        else:
            bit_health = int((len([x for x in bit_integrator_history.last_emitted_bits if x != BitValue.UNKNOWN]) / len(bit_integrator_history.last_emitted_bits)) * 100)
            bit_health_text = f"{bit_health}%"
        self.draw_text(GraphTypeEnum.BIT_HEALTH, bit_health_text)

        self.graph_for_type(GraphTypeEnum.EMITTED_SUBFRAMES).clear()
        emitted_subframes_text = f"{navigation_message_decoder_history.emitted_subframe_count}"
        self.draw_text(GraphTypeEnum.EMITTED_SUBFRAMES, emitted_subframes_text)

        self.graph_for_type(GraphTypeEnum.FAILED_BITS).clear()
        failed_bits_text = f'{bit_integrator_history.failed_bit_count}'
        self.draw_text(GraphTypeEnum.FAILED_BITS, failed_bits_text)

        # We've just erased some of our axes titles via plt.Axes.clear(), so redraw them.
        self._redraw_subplot_titles()
        plt.pause(0.001)
