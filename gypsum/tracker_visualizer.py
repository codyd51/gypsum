import base64
import io
import logging
import math
from dataclasses import dataclass
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from gypsum.constants import PSEUDOSYMBOLS_PER_NAVIGATION_BIT
from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.navigation_bit_intergrator import NavigationBitIntegratorHistory
from gypsum.navigation_message_decoder import NavigationMessageDecoderHistory
from gypsum.tracker import BitValue, GpsSatelliteTrackingParameters
from gypsum.units import Seconds
from gypsum.utils import get_iq_constellation_circularity
from gypsum.utils import get_iq_constellation_rotation

_logger = logging.getLogger(__name__)


_UPDATE_PERIOD: Seconds = 1.0
_RESET_DISPLAYED_DATA_PERIOD: Seconds = 5.0


@dataclass
class GraphAttributes:
    display_x_axis: bool = False
    display_y_axis: bool = False
    is_text_only: bool = False
    # The below fields are only relevant if this is a 'text-only' graph
    # Expressed as a hex RGB color, prefixed with '#'
    background_color: str | None = None

    @staticmethod
    def spacer() -> "GraphAttributes":
        return GraphAttributes(
            is_text_only=True,
            display_x_axis=False,
            display_y_axis=False,
        )

    @staticmethod
    def text(background_color: str) -> "GraphAttributes":
        return GraphAttributes(
            is_text_only=True,
            display_x_axis=False,
            display_y_axis=False,
            background_color=background_color,
        )

    @staticmethod
    def with_axes() -> "GraphAttributes":
        return GraphAttributes(
            display_x_axis=True,
            display_y_axis=True,
            is_text_only=False,
        )

    @staticmethod
    def with_y_axis() -> "GraphAttributes":
        return GraphAttributes(
            display_x_axis=False,
            display_y_axis=True,
            is_text_only=False,
        )

    @staticmethod
    def without_axes() -> "GraphAttributes":
        return GraphAttributes(
            display_x_axis=False,
            display_y_axis=False,
            is_text_only=False,
        )


class GraphTypeEnum(Enum):
    DOPPLER_SHIFT = auto()
    CARRIER_PHASE = auto()
    BIT_PHASE = auto()
    SUBFRAME_PHASE = auto()

    CARRIER_PHASE_ERROR = auto()
    TRACK_DURATION = auto()
    BIT_HEALTH = auto()

    IQ_COMPONENTS = auto()
    IQ_CONSTELLATION = auto()
    EMITTED_SUBFRAMES = auto()
    FAILED_BITS = auto()

    PSEUDOSYMBOLS = auto()
    IQ_ANGLE = auto()
    BITS = auto()
    PRN_CODE_PHASE = auto()
    CORRELATION_STRENGTH = auto()

    IQ_CONSTELLATION_CIRCULARITY = auto()
    IQ_CONSTELLATION_ROTATION = auto()

    SPACER3 = auto()

    @property
    def attributes(self) -> GraphAttributes:
        return {
            GraphTypeEnum.CARRIER_PHASE: GraphAttributes.with_y_axis(),
            GraphTypeEnum.DOPPLER_SHIFT: GraphAttributes.with_y_axis(),
            GraphTypeEnum.BITS: GraphAttributes.without_axes(),
            GraphTypeEnum.CARRIER_PHASE_ERROR: GraphAttributes.with_y_axis(),
            GraphTypeEnum.IQ_ANGLE: GraphAttributes.without_axes(),
            GraphTypeEnum.IQ_CONSTELLATION: GraphAttributes.without_axes(),
            GraphTypeEnum.IQ_COMPONENTS: GraphAttributes.with_y_axis(),
            GraphTypeEnum.PSEUDOSYMBOLS: GraphAttributes.without_axes(),
            GraphTypeEnum.BIT_HEALTH: GraphAttributes.text(background_color="#ffe7a6"),
            GraphTypeEnum.BIT_PHASE: GraphAttributes.text(background_color="#acdffc"),
            GraphTypeEnum.CORRELATION_STRENGTH: GraphAttributes.text(background_color="#ffe7a6"),
            GraphTypeEnum.EMITTED_SUBFRAMES: GraphAttributes.text(background_color="#c4fcac"),
            GraphTypeEnum.FAILED_BITS: GraphAttributes.text(background_color="#ffe7a6"),
            GraphTypeEnum.PRN_CODE_PHASE: GraphAttributes.text(background_color="#acdffc"),
            GraphTypeEnum.SUBFRAME_PHASE: GraphAttributes.text(background_color="#acdffc"),
            GraphTypeEnum.TRACK_DURATION: GraphAttributes.text(background_color="#c4fcac"),
            GraphTypeEnum.IQ_CONSTELLATION_CIRCULARITY: GraphAttributes.text(background_color="#c4fcac"),
            GraphTypeEnum.IQ_CONSTELLATION_ROTATION: GraphAttributes.text(background_color="#c4fcac"),
            GraphTypeEnum.SPACER3: GraphAttributes.spacer(),
        }[self]

    @classmethod
    def layout_order(cls) -> list[list["GraphTypeEnum"]]:
        """Defines the ordering of the graphs in the dashboard"""
        return [
            [
                cls.DOPPLER_SHIFT,
                cls.CARRIER_PHASE,
                cls.BIT_HEALTH,
                cls.FAILED_BITS,
            ],
            [
                cls.IQ_COMPONENTS,
                cls.IQ_CONSTELLATION,
                cls.CORRELATION_STRENGTH,
                cls.IQ_CONSTELLATION_ROTATION,
            ],
            [
                cls.PSEUDOSYMBOLS,
                cls.IQ_ANGLE,
                cls.EMITTED_SUBFRAMES,
                cls.TRACK_DURATION,
            ],
            [
                cls.BITS,
                cls.SPACER3,
                cls.BIT_PHASE,
                cls.PRN_CODE_PHASE,
            ],
            [
                cls.CARRIER_PHASE_ERROR,
                cls.SUBFRAME_PHASE,
                cls.IQ_CONSTELLATION_CIRCULARITY,
            ],
        ]

    @property
    def presentation_name(self) -> str:
        return {
            self.DOPPLER_SHIFT: "Beat Frequency (Hz)",
            self.IQ_CONSTELLATION: "IQ Constellation",
            self.CARRIER_PHASE_ERROR: "Carrier Phase Error",
            self.IQ_COMPONENTS: "IQ Components",
            self.IQ_ANGLE: "IQ Angle (Rad)",
            self.CARRIER_PHASE: "Carrier Phase (Rad)",
            self.PSEUDOSYMBOLS: "Pseudosymbols",
            self.BITS: "Bits",
            self.BIT_PHASE: "Bit Phase",
            self.SUBFRAME_PHASE: "Subframe Phase",
            self.TRACK_DURATION: "Track Duration",
            self.BIT_HEALTH: "Bit Health",
            self.PRN_CODE_PHASE: "PRN Code Phase",
            self.EMITTED_SUBFRAMES: "Emitted Subframes",
            self.FAILED_BITS: "Failed Bits",
            self.CORRELATION_STRENGTH: "PRN Correlation Strength",
            self.IQ_CONSTELLATION_CIRCULARITY: "IQ Circularity",
            self.IQ_CONSTELLATION_ROTATION: "IQ Rotation",
            self.SPACER3: "",
        }[self]


class GpsSatelliteTrackerVisualizer:
    def __init__(self, satellite_id: GpsSatelliteId, should_render: bool = True, should_present: bool = False) -> None:
        self.should_render = should_render
        self.should_present = should_present
        self._timestamp_of_last_dashboard_update = 0

        if not should_render:
            return

        if should_present:
            # Enable interactive mode if not already done
            if not plt.isinteractive():
                plt.ion()

        # PT: Disable the matplotlib toolbar.
        # Unfortunately, I don't know of a good way to do this on a per-figure basis, rather than globally.
        plt.rcParams['toolbar'] = 'None'
        self.visualizer_figure = plt.figure(figsize=(11, 7))
        title = f"Satellite #{satellite_id.id} Tracking Dashboard"
        self.visualizer_figure.suptitle(title, fontweight="bold")

        self.grid_spec = plt.GridSpec(nrows=6, ncols=4, figure=self.visualizer_figure)

        grid_spec_idx_iterator = iter(range(len(GraphTypeEnum)))
        # Initialize the graphs in the order specified
        self.graph_type_to_graphs = {}
        layout_order = GraphTypeEnum.layout_order()
        for row in layout_order:
            for graph_type in row:
                self.graph_type_to_graphs[graph_type] = self.visualizer_figure.add_subplot(
                    self.grid_spec[next(grid_spec_idx_iterator)]
                )

        self._redraw_subplot_titles()

        # All done, request tight layout
        self.grid_spec.tight_layout(self.visualizer_figure)

        # Each step, the dashboard will be re-rendered and persisted here.
        self.rendered_dashboard_png_base64: str = ''

    def _redraw_subplot_titles(self):
        """Unfortunately, plt.Axes.clear() also erases the subplot title.
        Therefore, every time we clear an axis, we have to redraw its title.
        """
        for graph_type, graph in self.graph_type_to_graphs.items():
            graph.set_title(graph_type.presentation_name)

        # Certain graph types are text-only, and we don't need the ticks/frame that pyplot provides by default.
        graphs_with_no_frames = [x for x in GraphTypeEnum if x.attributes.is_text_only]
        for graph_type in graphs_with_no_frames:
            self.graph_for_type(graph_type).axis("off")

        # Certain graph types aren't worth showing the axis labels, as the magnitudes aren't too important and they
        # clutter the UI.
        graph_without_x_axis_labels = [x for x in GraphTypeEnum if not x.attributes.display_x_axis]
        for graph_type in graph_without_x_axis_labels:
            self.graph_for_type(graph_type).get_xaxis().set_visible(False)

        graph_without_y_axis_labels = [x for x in GraphTypeEnum if not x.attributes.display_y_axis]
        for graph_type in graph_without_y_axis_labels:
            self.graph_for_type(graph_type).get_yaxis().set_visible(False)

    def graph_for_type(self, t: GraphTypeEnum) -> Axes:
        return self.graph_type_to_graphs[t]

    def draw_text(self, t: GraphTypeEnum, s: str):
        background_color = t.attributes.background_color
        if background_color is None:
            raise ValueError(f"No background color set for {t}")

        self.graph_for_type(t).text(
            0.5,
            0.25,
            s,
            fontsize=20,
            bbox={"edgecolor": "#000000", "facecolor": background_color, "boxstyle": "round", "pad": 0.2},
            ha="center",
        )

    def step(
        self,
        seconds_since_start: Seconds,
        current_tracking_params: GpsSatelliteTrackingParameters,
        bit_integrator_history: NavigationBitIntegratorHistory,
        navigation_message_decoder_history: NavigationMessageDecoderHistory,
    ) -> None:
        if not self.should_render:
            return

        if (
            self._timestamp_of_last_dashboard_update != 0
            and seconds_since_start - self._timestamp_of_last_dashboard_update < _UPDATE_PERIOD
        ):
            # It hasn't been long enough since our last GUI update
            return

        # Time to update the GUI
        self._timestamp_of_last_dashboard_update = seconds_since_start

        # locked_state = "Locked" if current_tracking_params.is_locked() else "Unlocked"
        # last_few_phase_errors = list(current_tracking_params.carrier_wave_phase_errors)[-250:]
        # variance = np.var(last_few_phase_errors) if len(last_few_phase_errors) >= 2 else 0
        # _logger.info(f'Seconds since start: {seconds_since_start} ({locked_state}), Variance {variance:.2f}')

        params = current_tracking_params
        self.graph_for_type(GraphTypeEnum.DOPPLER_SHIFT).clear()
        self.graph_for_type(GraphTypeEnum.DOPPLER_SHIFT).plot(params.doppler_shifts[::10])

        correlation_peaks = np.array(params.correlation_peaks_rolling_buffer)
        points_i = np.real(correlation_peaks)
        points_q = np.imag(correlation_peaks)
        self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION).clear()
        self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION).scatter(points_i, points_q)

        iq_constellation_rotation = get_iq_constellation_rotation(correlation_peaks)
        if iq_constellation_rotation is not None:
            self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION_ROTATION).clear()
            self.draw_text(GraphTypeEnum.IQ_CONSTELLATION_ROTATION, f'{iq_constellation_rotation:.2f}°')

            # Draw the mean point of each pole
            peaks_on_left_pole = correlation_peaks[correlation_peaks.real < 0]
            peaks_on_right_pole = correlation_peaks[correlation_peaks.real >= 0]
            left_pole = np.mean(peaks_on_left_pole) if len(peaks_on_left_pole) >= 2 else 0
            right_pole = np.mean(peaks_on_right_pole) if len(peaks_on_right_pole) >= 2 else 0
            self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION).scatter(
                [left_pole.real, right_pole.real], [left_pole.imag, right_pole.imag]
            )

        iq_constellation_circularity = get_iq_constellation_circularity(correlation_peaks)
        if iq_constellation_circularity is not None:
            self.graph_for_type(GraphTypeEnum.IQ_CONSTELLATION_CIRCULARITY).clear()
            self.draw_text(GraphTypeEnum.IQ_CONSTELLATION_CIRCULARITY, f'{iq_constellation_circularity:.2f}%')

        self.graph_for_type(GraphTypeEnum.IQ_COMPONENTS).clear()
        # Draw Q first so it stays in the background
        self.graph_for_type(GraphTypeEnum.IQ_COMPONENTS).plot(np.imag(correlation_peaks), color="#7f7f7f")
        self.graph_for_type(GraphTypeEnum.IQ_COMPONENTS).plot(np.real(correlation_peaks), color='#1f77b4')

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
            bits_as_runs.extend(
                [0.5 if bit == BitValue.UNKNOWN else bit.as_val() for _ in range(PSEUDOSYMBOLS_PER_NAVIGATION_BIT)]
            )
        self.graph_for_type(GraphTypeEnum.BITS).plot(bits_as_runs)

        self.graph_for_type(GraphTypeEnum.BIT_PHASE).clear()
        bit_phase_status_message = f"{bit_integrator_history.determined_bit_phase} pseudosymbols"
        self.draw_text(GraphTypeEnum.BIT_PHASE, bit_phase_status_message)

        self.graph_for_type(GraphTypeEnum.SUBFRAME_PHASE).clear()
        if navigation_message_decoder_history.determined_subframe_phase is None:
            subframe_phase_status_message = f"Unknown"
        else:
            subframe_phase_status_message = f"{navigation_message_decoder_history.determined_subframe_phase} bits"
        self.draw_text(GraphTypeEnum.SUBFRAME_PHASE, subframe_phase_status_message)

        self.graph_for_type(GraphTypeEnum.TRACK_DURATION).clear()
        # TODO(PT): This is the offset from startup, not track start...
        track_duration_text = f"{int(seconds_since_start)} seconds"
        self.draw_text(GraphTypeEnum.TRACK_DURATION, track_duration_text)

        self.graph_for_type(GraphTypeEnum.BIT_HEALTH).clear()
        # Bit health represents the proportion of the previous period of bits that were resolved with confidence
        if len(bit_integrator_history.last_emitted_bits) == 0:
            bit_health_text = "No bits seen yet"
        else:
            bit_health = int(
                (
                    len([x for x in bit_integrator_history.last_emitted_bits if x != BitValue.UNKNOWN])
                    / len(bit_integrator_history.last_emitted_bits)
                )
                * 100
            )
            bit_health_text = f"{bit_health}% success"
        self.draw_text(GraphTypeEnum.BIT_HEALTH, bit_health_text)

        self.graph_for_type(GraphTypeEnum.EMITTED_SUBFRAMES).clear()
        emitted_subframes_text = f"{navigation_message_decoder_history.emitted_subframe_count} subframes"
        self.draw_text(GraphTypeEnum.EMITTED_SUBFRAMES, emitted_subframes_text)

        self.graph_for_type(GraphTypeEnum.FAILED_BITS).clear()
        failed_bits_text = f"{bit_integrator_history.failed_bit_count} bits"
        self.draw_text(GraphTypeEnum.FAILED_BITS, failed_bits_text)

        self.graph_for_type(GraphTypeEnum.PRN_CODE_PHASE).clear()
        prn_code_phase_text = f"{current_tracking_params.current_prn_code_phase_shift} chips"
        self.draw_text(GraphTypeEnum.PRN_CODE_PHASE, prn_code_phase_text)

        self.graph_for_type(GraphTypeEnum.CORRELATION_STRENGTH).clear()
        mean_correlation_strength = np.mean(np.array(params.correlation_peak_strengths_rolling_buffer))
        correlation_strength_text = f"{mean_correlation_strength:.2f}"
        self.draw_text(GraphTypeEnum.CORRELATION_STRENGTH, correlation_strength_text)

        # We've just erased some of our axes titles via plt.Axes.clear(), so redraw them.
        self._redraw_subplot_titles()

        # Raster the figure to a base64-encoded image, so it can be rendered in the dashboard webserver.
        if self.should_render:
            pixel_buffer = io.BytesIO()
            self.visualizer_figure.savefig(pixel_buffer, format="png")
            pixel_buffer.seek(0)
            figure_as_png = pixel_buffer.getvalue()
            pixel_buffer.close()
            self.rendered_dashboard_png_base64 = base64.b64encode(figure_as_png).decode('utf-8')

        # Update the GUI loop if we're presenting directly from pyplot
        if self.should_present:
            plt.pause(0.001)

    def handle_satellite_dropped(self) -> None:
        plt.close(self.visualizer_figure)
