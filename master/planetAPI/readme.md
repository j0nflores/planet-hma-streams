# Planet Imagery Workflow

There are three main steps in the workflow: **lookup**, **order**, and **download**.

> **Important:** The Planet API enforces rate limits and quota usage. Avoid excessive parallelization.  
> The **lookup** and **order** steps are run serially.  
> The **download** step is configured for SLURM parallel runs; use a modest level of parallelism (recommended: ~4–5 concurrent tasks) to avoid throttling and failed requests.

---

## 1) `planet_lookup.py`

Queries the Planet API for all available imagery intersecting each feature in the input shapefile over the specified time period.

- **Input:** Shapefile features + time window  
- **Output:** One `.npy` metadata file per unique feature (saved to the lookup path)
- **Purpose:** Identifies all candidate scenes before placing any orders
- **Note:** This step does **not** consume quota. It only retrieves metadata.

---

## 2) `planet_order.py --order`

Uses the `.npy` files produced in the lookup step to place orders for all available images.

- **Input:** Lookup `.npy` files  
- **Output:** One `.json` file per feature containing the Planet download URLs (saved to the order path)
- **Important:** This step **consumes Planet quota** when orders are submitted, not when downloaded.
- **Tip:** If rerunning, ensure you are not re-ordering the same scenes unnecessarily.

---

## 3) `planet_order.py --download`

Downloads imagery using the URLs generated in the order step.

- **Prerequisite:** Confirm orders are complete by running `check_order_status.py` before downloading.
- **Execution:** Run as SLURM array jobs with ~4–5 parallel tasks to balance throughput and avoid API throttling.
- **Output:** Image files saved per feature ID in the download directory.
- **Note:** Downloads can take a significant amount of time depending on volume and network conditions.

---

# Planet API Notes & Best Practices

### Rate Limits
- Planet limits the number of simultaneous requests per user/IP.
- Too many parallel downloads can result in:
  - HTTP 429 errors (Too Many Requests)
  - Slower transfer speeds
  - Temporary throttling
- Recommended:
  - 4–5 concurrent SLURM tasks
  - Avoid launching large arrays all at once

### Quota Usage
- Quota is consumed at the **order stage**, not during lookup or download.
- Re-running the order step can double-charge quota if not careful.
- Always verify what has already been ordered before submitting new orders.

### Order Completion
- Orders are asynchronous and may take time to process.
- Always run: check_order_status.py

before downloading.
- Download will fail if orders are not yet ready.

### Retry Safety
- The download script is designed to skip files that already exist.
- If a job crashes, it can safely be restarted without re-downloading completed scenes.

### Parallelization Strategy
- Lookup: serial is sufficient  
- Order: serial recommended (quota-sensitive step)  
- Download: parallelize moderately (network-bound)

### Storage Considerations
- Planet imagery can be large; ensure sufficient disk space before ordering/downloading.