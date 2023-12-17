// Double-buffered iframes prevent a white flicker when refreshing an iframe's contents.
export class DoubleBufferedIFrame {
    constructor (spacer_div, iframe_container_div, iframe1, iframe2) {
        this.spacer = spacer_div;
        this.container = iframe_container_div;
        this.active_iframe = iframe1;
        this.active_iframe.style.visibility = "visible";

        this.inactive_iframe = iframe2;
        this.active_iframe.style.visibility = "hidden";

        // Prime the first refresh
        this.swapIFrames();
    }

    enqueueDelayedSwap() {
        window.setInterval(
            this.swapIFrames.bind(this),
            1000
        );
    }

    swapIFrames() {
        // Swap the active and inactive iframe references.
        let tmp = this.active_iframe;
        this.active_iframe = this.inactive_iframe;
        this.inactive_iframe = tmp;

        // Swap out the displayed iframe.
        // This is to prevent a white flicker while refreshing the iframe.
        let contentWindow = this.inactive_iframe.contentWindow.location;
        this.container.removeChild(this.inactive_iframe);
        this.active_iframe.style.visibility = "visible";
        this.active_iframe.style.maxHeight = "inherit";
        this.container.appendChild(this.inactive_iframe);
        this.active_iframe.style.height = this.active_iframe.contentWindow.document.documentElement.scrollHeight + 'px';

        // Ensure we take up the correct amount of space in the document flow
        this.spacer.style.height = this.active_iframe.style.height;
        this.inactive_iframe.style.visibility = "hidden";

        // Refresh the iframe that's just been swapped off-screen
        contentWindow.reload();
    }
}
