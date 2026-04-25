# ONEAT Event Detection Apps

Spatio-temporal event detection (e.g. mitosis) in 3D+T microscopy data using trained DenseVollNet models from KapoorLabs-Lightning.

Three deployment options are provided, each suited to a different use case.

## Streamlit Apps

### Remote App (`streamlit/remote_app.py`)

Submits prediction jobs to Jean Zay HPC via SSH + SLURM. The app runs on a lightweight VM; all GPU work happens on the cluster.

**Requirements:**
- sshfs mount of the Jean Zay lustre filesystem at `/home/debian/jean-zay/demo`
- SSH key access to the Jean Zay login node
- All scripts (`submit_job.sh`, `demo_predict.py`, `run_demo_prediction.slurm`) on the mount at `/home/debian/jean-zay/demo/`

**Usage:**
```bash
cd apps/streamlit
streamlit run remote_app.py
```

**Features:**
- **Optional file upload** -- a "Use demo files" checkbox (on by default) uses pre-placed raw and segmentation TIFs from the mount (`uploads/raw_demo_default.tif`, `uploads/seg_demo_default.tif`). Uncheck to upload custom files.
- **Model selection** -- scans `/home/debian/jean-zay/demo/models/` for model directories. Each directory should contain a `.ckpt` checkpoint and a `.json` architecture config. The model name, architecture parameters, and training config are displayed in expandable sidebar sections.
- **Memory-safe previews** -- default images are displayed as a single mid-Z slice at t=0 via page-level TIF access (`tifffile.TiffFile`), never loading the full volume into memory. The detection overlay viewer also reads one slice at a time.
- **Job tracking** -- polls a `status.txt` file on the mount to show live progress (queued, loading model, predicting, postprocessing, done).

**Adding models:**

Place model directories in the `models/` folder on the Jean Zay mount. Each directory becomes a selectable entry in the app. Expected structure:

```
models/
  oneat_mitosis_model_adam_heavy/
    epoch=249-step=19500.ckpt    # checkpoint
    adam_heavy_aug.json           # architecture config (input_channels, growth_rate, etc.)
    training_config.json          # full training parameters (optional, displayed in UI)
    adam_heavy_aug.npz            # optional extra files
```

The prediction script on Jean Zay (`demo_predict.py`) loads the model using its hardcoded `DEFAULT_MODEL_DIR` on lustre. The models shown in the app are for display and selection only -- the actual checkpoint path resolution happens server-side.

### Local App (`streamlit/local_app.py`)

Runs prediction locally using PyTorch. Requires a GPU (or CPU, slower) and the full `kapoorlabs_lightning` package installed.

**Usage:**
```bash
cd apps/streamlit
streamlit run local_app.py
```

**Features:**
- Upload raw + segmentation TIF files
- Select a `.ckpt` checkpoint from the `streamlit/models/` directory
- Configure model architecture, NMS, and normalization parameters in the sidebar
- Results displayed as a table with CSV download and a detection overlay viewer

**Adding models:**

Place `.ckpt` files directly in `apps/streamlit/models/`.

## Hugging Face App (`huggingface/app.py`)

Gradio-based app for deployment on Hugging Face Spaces.

**Usage:**
```bash
cd apps/huggingface
python app.py
```

See `huggingface/README.md` for the Spaces configuration.

## Architecture

All three apps use the same prediction pipeline from `kapoorlabs_lightning`:

1. Build a `DenseVollNet` network matching the trained architecture
2. Load weights via `OneatActionModule.load_from_checkpoint()`
3. Create an `OneatPredictionDataset` from raw + segmentation TIFs
4. Run `Trainer.predict()` to get raw detections
5. Apply `nms_space_time()` for non-maximum suppression
6. Output a CSV with columns: `t, z, y, x, event_name, cell_id`
