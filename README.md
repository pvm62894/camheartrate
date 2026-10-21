# Camera Heart Rate

Point a webcam at your face, palm or arm and read your heart rate, with no contact sensors. It uses remote photoplethysmography (rPPG): the tiny colour change of skin on each heartbeat.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_dsp.py                # DSP self-test, must print 11/11 passed
uvicorn app:app --reload          # then open http://localhost:8000
```

To start without a camera (for example, if a lecture-room machine has none), run `RPPG_SOURCE=synthetic uvicorn app:app --reload`. To start on a particular camera, give part of its name: `RPPG_CAMERA=iphone uvicorn app:app --reload`.

On first webcam use the app downloads Google's MediaPipe face and hand landmark models (~3.6 MB and ~7.8 MB) into `models/`. On macOS the terminal running uvicorn needs camera permission: *System Settings → Privacy & Security → Camera*.

## Using it

The page shows the camera, the patch of skin being read (outlined), your heart rate, and a one-minute graph of it.

* **Show some skin.** It picks a region on its own: a palm held up to the camera first, then your face, then any other exposed skin such as a forearm.
* **Tap to choose.** Click any skin in the picture to read from exactly there. *Pick skin automatically* undoes that.
* **Hold still for 5–10 s.** The outline is dashed while it measures and turns solid once it has a reading. With the screen light off it also flushes red on every beat.
* **Screen light** (on by default) turns the page white so your display lights your skin with steady, neutral light. Turn display brightness all the way up, use *Full screen*, and sit within about an arm's length. Under the light the beat animation stops, since a flickering screen would add a fake pulse to the light it casts.
* **Cameras.** It picks each camera's best mode up to 1080p, preferring frame rate. On a MacBook Pro the built-in FaceTime HD camera tops out at 1080p **30 fps** at every resolution. An iPhone used as a Continuity Camera runs 1080p **60 fps**, and when one is nearby a *Use … camera* link appears. Keep macOS video effects (Portrait, Studio Light, Center Stage in Control Center) off: they repaint skin tone and brightness, which is the very thing being measured.

Demo controls are on the keyboard only: `1` `2` `3` switch the method (POS, CHROM, raw green), `S` toggles the built-in simulation, and in the simulation `↑` `↓` change the injected heart rate and `←` `→` the amount of head movement.

## The physics: why skin colour carries your pulse

Each heartbeat pushes a pressure wave through the arterioles in the skin, so the volume of blood in the dermis rises and falls once per beat. Haemoglobin absorbs light strongly, with an absorption peak in green (~540–577 nm). More blood means slightly less reflected light. On a webcam the effect is about **0.1–0.5 % of the pixel value**, far below what the eye can see but well within reach of a statistical estimator.

(The flushed look after exercise is a different, slower effect: sustained vasodilation that raises baseline skin redness. rPPG tracks the tiny *beat-to-beat* change on top of that baseline.)

### Dichromatic reflection model

Following Wang et al. (2017), the colour of skin pixel *k* at frame *t* is

$$
\mathbf{C}_k(t) = I(t)\,\big(\mathbf{v}_s(t) + \mathbf{v}_d(t)\big) + \mathbf{v}_n(t)
$$

| term | meaning |
|---|---|
| $I(t)$ | luminance intensity: light source level, distance/angle to the camera. **Head motion and auto-exposure act here.** |
| $\mathbf{v}_s(t)$ | specular reflection: light mirrored off the skin surface, the colour of the light source, no pulse information. Changes with motion. |
| $\mathbf{v}_d(t) = \mathbf{u}_d d_0 + \mathbf{u}_p p(t)$ | diffuse reflection: light scattered inside the skin. $\mathbf{u}_d d_0$ is the stationary skin tone; $\mathbf{u}_p p(t)$ is the **pulsatile component** we want. |
| $\mathbf{v}_n(t)$ | camera sensor noise (shot + read noise, quantisation). |

The task is to recover $p(t)$ while rejecting the much larger $I(t)$ and $\mathbf{v}_s(t)$ variations.

### Spatial averaging

Averaging $N$ ROI pixels reduces independent sensor noise by $\sqrt{N}$. A forehead+cheeks ROI of ~15,000 pixels gives a ~122× reduction, which lifts a sub-1-LSB pulse above the 8-bit noise floor. `/api/telemetry` reports `roi_pixels` and `noise_reduction` live.

### Skin regions (`rppg_regions.py`, `rppg_vision.py`)

* **Face.** MediaPipe FaceLandmarker tracks the 468-point face mesh. Three convex-hull ROIs follow it: the forehead (landmarks `10, 338, 297, 332, 299, 337, 151, 108, 69, 103, 67, 109`) and both malar/cheek regions (`117, 118, 101, 36, 205, 187, 123` and `346, 347, 330, 266, 425, 411, 352`). Eyes, brows, lips and nostrils are left out because blinking, talking and hair add large non-cardiac changes. The three patches are measured separately and fused (see the DSP section).
* **Palm.** MediaPipe HandLandmarker; the ROI is the hull of the wrist, thumb base and four knuckles (`0, 1, 5, 9, 13, 17`), shrunk 15 % toward its centre. Palms are very well perfused.
* **Any other skin.** Colour segmentation in YCrCb. Skin of every tone clusters tightly in the (Cr, Cb) chroma plane, because melanin and haemoglobin mostly change brightness, not hue. A fixed box is used until a face or palm has been seen; after that a Gaussian fitted to *your* skin replaces it, which rejects wood and walls much better. Blobs are cleaned with morphology, the face and hand are cut out, and a border is eroded off so edge pixels that flicker between skin and background are not averaged.

Landmarks pass through a One Euro filter (Casiez et al., 2012): it smooths hard while you are still and lets go while you move, so the outline neither shakes nor lags. Every pixel of jitter would otherwise move the ROI edge over different pixels and add noise. Pixels with any channel at 250 or above are skipped, because a saturated pixel has no pulse left in it.

Region changes restart the measurement (each patch has its own colour and pulse amplitude), so automatic switching needs the new choice to persist for ~1 s and rides through brief detection dropouts. "Skin" blobs covering more than 40 % of the frame are rejected as walls or colour casts.

### Capture and threads (`rppg_camera.py`, `rppg_pipeline.py`)

On macOS frames come straight from AVFoundation (via PyObjC). The chosen format and a fixed frame duration are set on both the device and the capture connection; without the connection setting, Continuity Camera silently delivers 30 fps despite accepting a 60 fps format. Each frame carries the camera's own presentation timestamp, so sample timing never depends on Python scheduling. Elsewhere OpenCV is asked for 1920×1080 at 60 fps.

Three threads: *capture* (frames into a hand-off queue), *measure* (every frame: average each skin patch inside a mask cropped to the region's bounding box, feed the DSP; this keeps up with 60 fps at 1080p), and *track* (newest frame only, 30 Hz: landmarks, skin blobs, region choice, the next mask, and the JPEG for the page). `/api/telemetry` reports `camera_fps` (what the camera delivers) and `fps` (what was measured).

## The DSP pipeline (`rppg_dsp.py`)

A 15 s rolling buffer of `(t, R̄, Ḡ, B̄)` samples, one triple per skin patch. Every 0.25 s (after 5 s of data) the heart rate is re-estimated.

1. **Uniform resampling.** Webcam frame times jitter and frames drop. `scipy.interpolate.interp1d` linearly resamples the raw traces onto an exact grid at the camera rate (30 or 60 Hz) that ends at the newest sample.
2. **Pulse extraction.** One of three algorithms, selectable live:
   * **POS (default).** For each 1.6 s sliding window (48 samples at 30 Hz, 96 at 60 Hz): temporally normalise $C_n = C/\mu_C$, which maps the skin tone to $[1,1,1]$. Then project onto the plane orthogonal to it:
     $S_1 = G_n - B_n$, $S_2 = G_n + B_n - 2R_n$,
     $h = S_1 + \frac{\sigma(S_1)}{\sigma(S_2)+10^{-8}}\,S_2$.
     Zero-mean each $h$ and overlap-add. **Why it works:** after normalisation an intensity change $I(t)$ moves all channels by the same factor, i.e. along $[1,1,1]$, and both projection axes are orthogonal to $[1,1,1]$, so it vanishes. Specular changes project onto $S_1$ and $S_2$ with *opposite* signs while the pulse projects with the *same* sign, so the $\alpha = \sigma_1/\sigma_2$ tuning cancels the first and reinforces the second.
   * **CHROM.** $X = 3R_n - 2G_n$, $Y = 1.5R_n + G_n - 1.5B_n$ (both cancel white-light intensity for a standard skin tone). Both are bandpassed, and $S = X_f - \alpha Y_f$ with $\alpha = \sigma(X_f)/\sigma(Y_f)$ per half-overlapping window.
   * **GREEN.** The normalised green channel alone. This is the baseline with no motion rejection.
3. **Detrend + bandpass.** `scipy.signal.detrend` (linear), then a zero-phase 4th-order Butterworth bandpass, `filtfilt`, 0.75–3.5 Hz (45–210 BPM).
4. **Spectral HR.** Welch PSD per patch (Hann, 10 s segments, 50 % overlap, zero-padded to ~0.45 BPM bins).
   * **Patch fusion.** Each patch's spectrum is normalised; a quality-weighted average finds the rate the patches agree on; then each patch is re-weighted by the share of its power at that rate. A shadowed cheek, or a patch with hair moving over it, ends up with almost no weight. In the self-test one pooled colour reads 110 bpm where fusion reads the true 70.
   * **Peak choice.** The tallest in-band peak, except that a peak within ±12 bpm of the current reading wins unless the other is more than twice as strong (continuity), and a clear peak at half the frequency wins over its 2nd harmonic (a real heart rate has nothing at half its rate). Refined by parabolic interpolation: $\text{BPM}_{raw} = 60 f_{max}$.
5. **Signal quality.**
   SNR = power within ±0.15 Hz of $f_{max}$ and of $2f_{max}$, divided by the remaining in-band power.
   SQI (%) = $100 \cdot S/(S+N)$. In-band noise alone measures ~20–37 %, so the status levels are **Locked** at ≥ 50 %, **Weak signal** at 40–50 %, and **High motion noise** below 40 %, with 3 points of hysteresis so the label does not flicker. The page says "too much movement" only when the region is actually moving; otherwise it asks for more light.
6. **Readout.** The first number appears after a solid lock (SQI ≥ 50 %) or after 4 s of moderate-quality estimates that agree within ±2.5 bpm. After that: estimates below 40 % SQI never move it; it follows the median of the last 2 s of good estimates; a change of more than 12 bpm must persist for 2.5 s; and it moves by an EMA whose gain grows with SQI ($\alpha = 0.03 + 0.37\,\text{SQI}^2$). In the self-test a 3 s burst of blood-coloured motion that swung the old readout by 55 bpm now moves it by 0.1.
7. **HRV (RMSSD).** When SQI > 50 %, `find_peaks` locates systolic peaks (min distance 0.6 × beat period, prominence 0.3σ), sub-sample refined. Inter-beat intervals deviating > 25 % from the median are rejected, and RMSSD is computed from the successive differences.

### Suggested lecture demo

1. Start on the webcam with POS. Sit still under steady light, wait for the solid outline, and compare with a pulse oximeter or a finger on the wrist. Hold up a palm and watch it switch.
2. Press `S` for the simulation, then `→` until movement is ~80 %. The status line shows the true rate.
3. Press `1` (POS): the reading stays within ~1 bpm of the truth.
4. Press `3` (raw green): the reading jumps to the nodding rhythm, often 50–90 bpm off.
5. Raw green can even report a confident lock on the wrong rate. Signal quality measures *periodicity*, and a rhythmic nod is periodic too. Rejecting it takes the colour-space projection, not the spectrum.

The full DSP output (BVP waveform, spectrum, RGB traces, SQI, RMSSD) is still streamed in every telemetry message and available at `/api/telemetry`.

## Files

| file | role |
|---|---|
| `rppg_camera.py` | camera capture: AVFoundation on macOS (best mode, pinned frame rate, hardware timestamps), OpenCV elsewhere |
| `rppg_vision.py` | MediaPipe face and hand landmarks, One Euro landmark smoothing, skin segmentation |
| `rppg_regions.py` | face / palm / skin-blob ROIs, automatic choice with hysteresis, tap-to-select |
| `rppg_dsp.py` | resampling, POS / CHROM / GREEN, filtering, Welch HR, patch fusion, SQI, readout logic, RMSSD |
| `rppg_synthetic.py` | dichromatic-model skin trace generator (pulse with dicrotic notch, sinus arrhythmia, motion, specular, noise) and its stand-in frame |
| `rppg_pipeline.py` | measure and track threads; sends the mirrored outline as polygons |
| `app.py` | FastAPI server, `/ws` WebSocket (binary JPEG + JSON telemetry, JSON commands back) |
| `static/index.html` | the app page: camera, skin outline, heart rate, one-minute graph (plain HTML/CSS/canvas) |
| `test_dsp.py` | 11 self-tests: 72.0 ± 1.0 BPM through motion spikes (30 and 60 Hz), full 60–160 range, harmonic guard, patch fusion, motion bursts, steady first reading |

## Limitations

* **Not a medical device.** Expect a few BPM of error at rest, more under motion.
* **Lighting** matters most: use bright, steady, diffuse light. Flickering LED or fluorescent light can show up as a spectral peak.
* **Auto-exposure / white balance.** Neither the MacBook camera nor Continuity Camera lets macOS lock them. POS's per-window normalisation absorbs most exposure steps, and steady light (the screen light) keeps the camera from hunting.
* **Skin tone.** Higher melanin absorbs more light, so the pulsatile signal gets weaker and SNR drops. This is a documented bias in rPPG and in contact pulse oximetry.
* **HRV** from a webcam is indicative only. Peak-timing noise inflates RMSSD; in the synthetic tests it reads roughly 1.3–2× the injected value.
* **Prior art.** Camera-based pulse measurement is an established research field (references below). This project is a clean, real-time implementation of those methods, built for demonstration.

## References

* W. Verkruysse, L. O. Svaasand, J. S. Nelson. *Remote plethysmographic imaging using ambient light.* Optics Express, 2008.
* M.-Z. Poh, D. J. McDuff, R. W. Picard. *Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.* Optics Express, 2010.
* G. de Haan, V. Jeanne. *Robust pulse rate from chrominance-based rPPG.* IEEE Trans. Biomed. Eng., 2013.
* W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan. *Algorithmic principles of remote PPG.* IEEE Trans. Biomed. Eng., 2017 (POS).
