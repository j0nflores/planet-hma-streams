#Modified from https://github.com/scikit-image

import numpy as np

def threshold_otsu(image=None, nbins=256, *, hist=None):
    """Return threshold value based on Otsu's method.
    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored.
    Parameters
    ----------
    image : (N, M[, ..., P]) ndarray, optional
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities. If no hist provided,
        this function will compute it from the image.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        warn(f'threshold_otsu is expected to work correctly only for '
             f'grayscale images; image shape {image.shape} looks like '
             f'that of an RGB image.')

    # Check if the image has more than one intensity value; if not, return that
    # value
    if image is not None:
        first_pixel = image.reshape(-1)[0]
        if np.all(image == first_pixel):
            return first_pixel

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold

def histogram(image, nbins=256, source_range='image', normalize=False, *,
              channel_axis=None):
    """Return histogram of image.
    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.
    If `channel_axis` is not set, the histogram is computed on the flattened
    image. For color or multichannel images, set ``channel_axis`` to use a
    common binning for all channels. Alternatively, one may apply the function
    separately on each channel to obtain a histogram for each color channel
    with separate binning.
    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    source_range : string, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    Returns
    -------
    hist : array
        The values of the histogram. When ``channel_axis`` is not None, hist
        will be a 2D array where the first axis corresponds to channels.
    bin_centers : array
        The values at the center of the bins.
    See Also
    --------
    cumulative_distribution
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([ 93585, 168559]), array([0. , 0.5, 1. ]))
    >>> exposure.histogram(image, nbins=2)
    (array([ 93585, 168559]), array([0.25, 0.75]))
    """
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4 and channel_axis is None:
        utils.warn('This might be a color image. The histogram will be '
                   'computed on the flattened image. You can instead '
                   'apply this function to each color channel, or set '
                   'channel_axis.')

    if channel_axis is not None:
        channels = sh[-1]
        hist = []

        # compute bins based on the raveled array
        if np.issubdtype(image.dtype, np.integer):
            # here bins corresponds to the bin centers
            bins = _bincount_histogram_centers(image, source_range)
        else:
            # determine the bin edges for np.histogram
            hist_range = _get_numpy_hist_range(image, source_range)
            bins = _get_bin_edges(image, nbins, hist_range)

        for chan in range(channels):
            h, bc = _histogram(image[..., chan], bins, source_range, normalize)
            hist.append(h)
        # Convert to numpy arrays
        bin_centers = np.asarray(bc)
        hist = np.stack(hist, axis=0)
    else:
        hist, bin_centers = _histogram(image, nbins, source_range, normalize)

    return hist, bin_centers


def _histogram(image, bins, source_range, normalize):
    """
    Parameters
    ----------
    image : ndarray
        Image for which the histogram is to be computed.
    bins : int or ndarray
        The number of histogram bins. For images with integer dtype, an array
        containing the bin centers can also be provided. For images with
        floating point dtype, this can be an array of bin_edges for use by
        ``np.histogram``.
    source_range : string, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    """

    image = image.flatten()
    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        bin_centers = bins if isinstance(bins, np.ndarray) else None
        hist, bin_centers = _bincount_histogram(
            image, source_range, bin_centers
        )
    else:
        hist_range = _get_numpy_hist_range(image, source_range)
        hist, bin_edges = np.histogram(image, bins=bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers

def _validate_image_histogram(image, hist, nbins=None, normalize=False):
    """Ensure that either image or hist were given, return valid histogram.
    If hist is given, image is ignored.
    Parameters
    ----------
    image : array or None
        Grayscale image.
    hist : array, 2-tuple of array, or None
        Histogram, either a 1D counts array, or an array of counts together
        with an array of bin centers.
    nbins : int, optional
        The number of bins with which to compute the histogram, if `hist` is
        None.
    normalize : bool
        If hist is not given, it will be computed by this function. This
        parameter determines whether the computed histogram is normalized
        (i.e. entries sum up to 1) or not.
    Returns
    -------
    counts : 1D array of float
        Each element is the number of pixels falling in each intensity bin.
    bin_centers : 1D array
        Each element is the value corresponding to the center of each intensity
        bin.
    Raises
    ------
    ValueError : if image and hist are both None
    """
    if image is None and hist is None:
        raise Exception("Either image or hist must be provided.")

    if hist is not None:
        if isinstance(hist, (tuple, list)):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)

        if counts[0] == 0 or counts[-1] == 0:
            # Trim histogram from both ends by removing starting and
            # ending zeroes as in histogram(..., source_range="image")
            cond = counts > 0
            start = np.argmax(cond)
            end = cond.size - np.argmax(cond[::-1])
            counts, bin_centers = counts[start:end], bin_centers[start:end]
    else:
        counts, bin_centers = histogram(
            image[~np.isnan(image)].reshape(-1), nbins, source_range='image', normalize=normalize
            )
    return counts.astype('float32', copy=False), bin_centers



def _bincount_histogram_centers(image, source_range):
    """Compute bin centers for bincount-based histogram."""
    if source_range not in ['image', 'dtype']:
        raise ValueError(
            f'Incorrect value for `source_range` argument: {source_range}'
        )
    if source_range == 'image':
        image_min = int(image.min().astype(np.int64))
        image_max = int(image.max().astype(np.int64))
    elif source_range == 'dtype':
        image_min, image_max = dtype_limits(image, clip_negative=False)
    bin_centers = np.arange(image_min, image_max + 1)
    return bin_centers

def _get_numpy_hist_range(image, source_range):
    if source_range == 'image':
        hist_range = None
    elif source_range == 'dtype':
        hist_range = dtype_limits(image, clip_negative=False)
    else:
        ValueError('Wrong value for the `source_range` argument')
    return hist_range