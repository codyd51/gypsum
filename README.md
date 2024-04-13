<p align="center">
  <img src="./readme_images/gypsum_logo_header.png" width="800">
</p>

gypsum is a homebrew GPS receiver, written in Python. gypsum can carry out a position fix in less than 1 minute of antenna readings from cold start. 

gypsum implements a GPS signal processing stack with no dependencies aside from `numpy`. 

<p align="center">
  <img src="./readme_images/sat_tracker.png" width="600">
</p>

gypsum turns any commodity SDR into a GPS receiver. I primarily use an [RTL-SDR](https://www.rtl-sdr.com/buy-rtl-sdr-dvb-t-dongles/), and have had success with a [HackRF](https://greatscottgadgets.com/hackrf/one/) as well. Either of these can be paired with any patch antenna sensitive to 1.57542MHz.

[This project comes with a 4-part writeup](https://axleos.com/building-a-gps-receiver-part-1-hearing-whispers/) on implementing a GPS receiver from scratch.

https://github.com/codyd51/gypsum/assets/4972184/e72151fe-994e-4e5a-95b4-19e5c91d2b20

gypsum ships with a web-based dashboard that allows the user to monitor signal quality, track progress, position fix history, and satellite tracking pipeline state.

## Using gypsum

The most convenient way to try out gypsum is to use a file containing saved antenna samples. This allows off-the-air development and signal replays.

I've uploaded a sample antenna recording to the [Releases section] of the repo. Download [this file] and place it in `./gypsum/vendored_signals/`. gypsum needs information on what these files contain, so currently their info is hard-coded [here](). In the future, we could introduce a bespoke file format that includes the recording parameters and the antenna samples in-band.

```bash
# Install gypsum's dependencies
$ pip install -r requirements.txt
# If you want to use the web-based tracking dashboard
$ pip install -r requirements-webapp.txt

# Run gypsum against the cached antenna samples
# (And limit the satellite search scope for speed) 
$ python3 gypsum-cli.py --only_acquire_satellite_ids 25 28 31 32 --present_web_ui

# (In another shell)
# Launch the webserver to observe gypsum
$ gunicorn -b :8080 --timeout 0 web_dashboard:application
```

## License

MIT 
