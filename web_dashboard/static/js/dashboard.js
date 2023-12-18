import { DoubleBufferedIFrame } from './double_buffered_iframe.js';

addEventListener("load", (event) => {
    let receiverStatsIFrame = new DoubleBufferedIFrame(
        document.getElementById("receiver_stats_spacer"),
        document.getElementById("receiver_stats_iframe_container"),
        document.getElementById("receiver_stats1"),
        document.getElementById("receiver_stats2"),
    );
    receiverStatsIFrame.enqueueDelayedSwap();

    let trackerVisualizerIFrame = new DoubleBufferedIFrame(
        document.getElementById("tracker_visualizers_spacer"),
        document.getElementById("tracker_visualizers_iframe_container"),
        document.getElementById("tracker_visualizers1"),
        document.getElementById("tracker_visualizers2"),
    );
    trackerVisualizerIFrame.enqueueDelayedSwap();

    let satelliteInfosIFrame = new DoubleBufferedIFrame(
        document.getElementById("satellite_infos_spacer"),
        document.getElementById("satellite_infos_iframe_container"),
        document.getElementById("satellite_infos1"),
        document.getElementById("satellite_infos2"),
    );
    satelliteInfosIFrame.enqueueDelayedSwap();
});