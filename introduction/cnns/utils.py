import numpy as np


def conv_(img, conv_filter):
    """Perform 2D convolution between a single-channel image and a single filter.

    :param img: 2D NumPy array (H x W) single-channel image patch
    :param conv_filter: 2D NumPy array (F x F) square filter with odd F
    :return: 2D NumPy array containing the valid (no padding) convolution

    Notes
    - This function iterates over valid center positions where the filter fully
      fits inside the image. It multiplies element-wise the F x F region and
      sums the result. The working `result` buffer keeps the same shape as the
      input image, but the returned `final_result` is trimmed so only the valid
      convolution outputs are returned (same convention as 'valid' in many
      libraries).
    - Uses integer indices computed with floor/ceil to support odd-sized
      square filters.
    """

    filter_size = conv_filter.shape[1]
    # result uses the same shape as the input image so we can write by center
    # coordinates (r, c) and then trim the borders at the end.
    # Create result buffer with the same shape as the input image. Use the
    # raw tuple `img.shape` rather than nesting it in another tuple.
    result = np.zeros(img.shape)

    # Integer padding (half the filter size). For odd filter sizes this equals
    # floor(F/2) and is used to compute start/end indices for slicing.
    pad = int(filter_size // 2)

    # Looping through the image to apply the convolution operation. We iterate
    # only over center positions where the full filter window is inside img.
    for r in np.arange(filter_size / 2.0, img.shape[0] - filter_size / 2.0 + 1).astype(
        np.uint16
    ):
        for c in np.arange(
            filter_size / 2.0, img.shape[1] - filter_size / 2.0 + 1
        ).astype(np.uint16):
            # Extract the current region (F x F) centered at (r, c). We compute
            # start/end using floor/ceil to handle the symmetric window around
            # the integer center indices for odd-sized filters.
            # Use integer pad to compute the F x F region centered at (r, c).
            curr_region = img[
                r - pad : r + (pad + 1),
                c - pad : c + (pad + 1),
            ]
            # Element-wise multiplication between the current region and the filter
            # followed by summation to produce a single scalar response.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)
            # Save the computed value at the center coordinate in the working
            # result buffer. Later we trim the buffer to return only valid
            # convolution outputs.
            result[r, c] = conv_sum

    # Trim border values where the filter did not fully overlap the image.
    # Use the original image shape for computing the trimmed region bounds.
    final_result = result[
        pad : img.shape[0] - pad,
        pad : img.shape[1] - pad,
    ]
    return final_result


def conv(img, conv_filter):
    """Apply a bank of filters to an image and return feature maps.

    :param img: H x W x C or H x W NumPy array (C is optional for single-channel)
    :param conv_filter: K x F x F or K x F x F x C NumPy array where K is the number
      of filters and F is the filter (kernel) size. If filters have a channel
      dimension it must match the image channel count C.
    :return: NumPy array of shape (H - F + 1, W - F + 1, K) containing
      the valid convolution outputs for each filter.

    Notes
    - This function performs basic sanity checks on shapes and ensures filters
      are square and odd-sized. It supports multi-channel images and filters
      by convolving each filter channel with the corresponding image channel
      and summing the results to produce a single feature map per filter.
    - The implementation is straightforward and prints the filter index as it
      processes each filter (useful for small examples).
    """

    # Check if number of image channels matches the filter depth
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            exit()

    # Check if filter dimensions are equal (square filters expected)
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print(
            "Error: Filter must be a square matrix, i.e. number of rows and columns must match."
        )
        exit()

    # Check if filter dimensions are odd (this code assumes odd-sized kernels)
    if conv_filter.shape[1] % 2 == 0:
        print(
            "Error: Filter must have an odd size, i.e. number of rows and columns must be odd."
        )
        exit()

    # Prepare the output feature maps buffer. For each filter we will produce a
    # (H-F+1) x (W-F+1) valid convolution map.
    feature_maps = np.zeros(
        (
            img.shape[0] - conv_filter.shape[1] + 1,
            img.shape[1] - conv_filter.shape[1] + 1,
            conv_filter.shape[0],
        )
    )

    # Convolve the image by each filter in the bank.
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        # Extract this filter (could be single-channel F x F or F x F x C)
        curr_filter = conv_filter[filter_num, :]

        # If the filter has multiple channels, convolve each channel with the
        # corresponding image channel and sum the results to form one feature map.
        if len(curr_filter.shape) > 2:
            # Start with the response from the first channel
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
            # Accumulate responses from remaining channels
            for ch_num in range(1, curr_filter.shape[-1]):
                conv_map = conv_map + conv_(
                    img[:, :, ch_num], curr_filter[:, :, ch_num]
                )
        else:
            # Single-channel filter: directly convolve the full image
            conv_map = conv_(img, curr_filter)

        # Store the computed feature map for this filter
        feature_maps[:, :, filter_num] = conv_map

    return feature_maps


def pooling(feature_map, size=2, stride=2):
    """Perform max-pooling over a feature map.

    :param feature_map: H x W x K NumPy array containing K feature maps
    :param size: pooling window size (assumed square)
    :param stride: step size between windows
    :return: pooled feature maps with shape roughly ((H-size)/stride+1,
      (W-size)/stride+1, K)

    Notes
    - This is a straightforward max-pooling implementation that visits each
      non-overlapping (or overlapping depending on stride) window and takes the
      maximum. The output size formula matches the discrete stepping over the
      input with the provided stride.
    """

    # Preparing the output of the pooling operation. Compute output dims using
    # the standard sliding-window formula.
    pool_out = np.zeros(
        (
            np.uint16((feature_map.shape[0] - size + 1) / stride + 1),
            np.uint16((feature_map.shape[1] - size + 1) / stride + 1),
            feature_map.shape[-1],
        )
    )

    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        # Slide the pooling window top-left corner by 'stride' steps
        for r in np.arange(0, feature_map.shape[0] - size + 1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1] - size + 1, stride):
                # Take the maximum value in the current size x size window
                pool_out[r2, c2, map_num] = np.max(
                    [feature_map[r : r + size, c : c + size, map_num]]
                )
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out


def relu(feature_map):
    """Apply element-wise ReLU (Rectified Linear Unit) to a feature map.

    :param feature_map: NumPy array of arbitrary shape (commonly H x W x K)
    :return: NumPy array of same shape where negative values are replaced by 0
    """

    # Allocate output buffer with same shape and populate by elementwise max
    # with zero. This loop-based approach mirrors the style of the rest of
    # this educational module; in production prefer vectorized operations.
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out
