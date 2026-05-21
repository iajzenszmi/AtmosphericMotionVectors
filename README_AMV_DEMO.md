# Atmospheric Motion Vector Synthetic Model Demo

Files:
- `setup_run_amv_demo.sh`: installs dependencies and runs the demo.
- `amv_model_demo.py`: complete Python AMV simulator/tracker/visualiser.

Run:

```bash
chmod +x setup_run_amv_demo.sh
./setup_run_amv_demo.sh
```

Interactive display, useful in Pydroid:

```bash
./setup_run_amv_demo.sh --show
```

Outputs go to `output_amv/`, including PNG visualisations, CSV vectors, and a run summary.

Main features:
- synthetic cloud image simulation
- known true wind field
- frame-to-frame advection
- normalised cross-correlation template tracking
- sub-pixel peak refinement
- quality control
- brightness-temperature cloud-top height assignment
- CSV and PNG outputs
