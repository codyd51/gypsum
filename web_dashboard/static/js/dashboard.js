import { DoubleBufferedIFrame } from './double_buffered_iframe.js';

addEventListener("load", (event) => {
    let trackerVisualizerIFrame = new DoubleBufferedIFrame(
        document.getElementById("tracker-visualizers-spacer"),
        document.getElementById("tracker-visualizers-iframe-container"),
        document.getElementById("tracker-visualizers1"),
        document.getElementById("tracker-visualizers2"),
    );
    trackerVisualizerIFrame.enqueueDelayedSwap();
});