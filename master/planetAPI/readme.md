# Planet Imagery Workflow

This workflow has three main steps: **lookup**, **order**, and **download**.

> **Note:** The Planet API has rate limits and quota usage.  
> The **lookup** and **order** steps can be run serially.  
> The **download** step is set up for SLURM parallel runs. Use a small number of parallel tasks (about 4–5) to avoid throttling.

## Setup (`config.py`)

Before running the workflow, update `config.py` to define the run name and paths:

- **run**: Label for the current processing run. Changing this creates a new output folder.
- **shp_path**: Path to the input shapefile. Each feature will be processed separately.
- **lookup_path**: Directory where .npy metadata files from the lookup step will be saved.
- **order_path**: Directory where .json order URL files will be saved.
- **download_path**: Directory where downloaded imagery will be stored.

---

## 1) `planet_lookup.py`

Queries the Planet API for all available imagery intersecting each feature in the input shapefile over the specified time period.

- **Input:** Shapefile features and time window  
- **Output:** One `.npy` metadata file per feature (saved to the lookup path)  
- **Purpose:** Identifies all candidate scenes before placing any orders  
- **Note:** This step does **not** use quota. It only retrieves metadata.

---

## 2) `planet_order.py --order`

Uses the `.npy` files from the lookup step to place orders for all available images.

- **Input:** Lookup `.npy` files  
- **Output:** One `.json` file per feature containing Planet download URLs (saved to the order path)  
- **Note:** This step **uses Planet quota** when orders are submitted.

---

## 3) `planet_order.py --download`

Downloads imagery using the URLs created in the order step.

- **Before downloading:** Make sure orders are complete by running: check_order_status.py
- **Execution:** Run as SLURM array jobs with ~4–5 parallel tasks  
- **Output:** Images saved per feature ID in the download directory  
- **Note:** Downloads may take time depending on volume and network speed.

---

## Notes and Tips

- **Rate limits:** Too many parallel downloads can slow things down or cause failed requests. A small number of parallel jobs (4–5) works well.
- **Quota usage:** Quota is used during the **order** step, not during lookup or download.
- **Order completion:** Orders are processed asynchronously. Always check status before downloading.
- **Restarting downloads:** The download step skips files that already exist, so it is safe to rerun if interrupted.
- **Storage:** Planet imagery can be large. Make sure there is enough disk space before ordering and downloading.
