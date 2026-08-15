"""
ilastik Pixel Classification, written to be read.

This is a teaching copy. It runs a real trained .ilp project on a real image
and produces the same probabilities as the optimised
pixel_classification_standalone.py next to it -- but every optimisation has
been taken out, so the shape of the algorithm is on the surface. Convolutions
are literal loops. The random forest is walked one pixel and one tree at a
time. Nothing is cached, batched, or vectorised.

That makes it slow. Genuinely slow: seconds per thousand pixels, where the
optimised version does a megapixel image in under a minute. Run it on a small
crop. Slowness is the price of every step being visible, and here that is the
point.

    # what it predicts, on a small patch that straddles a cell edge
    python ilastik_pc.py pc.ilp image.png --region 64 96 128 160

    # what it did to reach one pixel's answer, in full
    python ilastik_pc.py pc.ilp image.png --region 64 96 128 160 --explain 14 13

The pixel in that second command is a good one to start on: the forest splits
51/49 there, so the trees genuinely disagree and you can watch two of them
reach opposite leaves from the same feature vector.

The --explain mode is the reason this file exists: it prints the whole
derivation for a single pixel -- every feature value with the name of the
filter that produced it, then each tree's descent through its decision nodes,
comparison by comparison, down to the leaf. That is the thing you can follow
by hand on a whiteboard.

SCOPE, deliberately narrow, so the structure stays legible:
  - 2D images only (the real thing also does 3D volumes)
  - no ComputeIn2d / thin-z handling
  - no time axis
  - inference only; training is a different program entirely
For any of that, read pixel_classification_standalone.py, which handles the
full matrix of cases and is correspondingly harder to read.

Dependencies: numpy, h5py. (numpy is used for array storage and 1-line
arithmetic, not to hide any of the algorithm.)


THE WHOLE ALGORITHM, IN ONE PARAGRAPH

A trained ilastik project holds two things: a list of image filters the user
ticked, and a random forest trained on their output. To classify a pixel, you
run every selected filter over the image, collect that pixel's value from each
resulting filter output into a vector of numbers -- its "feature vector" --
and drop that vector down each tree in the forest. Each tree steers left or
right at every node by comparing one feature against a threshold, and lands on
a leaf holding class probabilities. Average the leaves across all trees and
you have the pixel's answer. Everything below is detail on those two steps.
"""

import argparse
import math
import sys

import h5py
import numpy as np


###############################################################################
#
#  PART 1 -- GAUSSIAN KERNELS
#
#  Every filter here is built out of one primitive: a 1D Gaussian, or one of
#  its derivatives, sampled at integer offsets. Get this right and the filters
#  are just combinations of it.
#
###############################################################################


PRESMOOTHING_WINDOW = 3.5  # how many sigmas wide the presmoothing kernel is
FEATURE_WINDOW = 2.0       # ...and the feature filters' kernels


def gaussian_at(x, sigma, order):
    """
    The Gaussian, or its 1st or 2nd derivative, evaluated at x.

    order=0 is the bell curve itself; order=1 and 2 are its derivatives, which
    is how the filters below detect edges (1st) and ridges (2nd).
    """
    bell = math.exp(-0.5 * (x / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)

    if order == 0:
        return bell
    elif order == 1:
        return -(x / sigma**2) * bell
    elif order == 2:
        return ((x**2 - sigma**2) / sigma**4) * bell
    else:
        raise ValueError(f"only orders 0-2 are needed here, got {order}")


def build_kernel(sigma, order, window):
    """
    Sample gaussian_at() at integer offsets, then correct for the fact that we
    truncated an infinitely wide function.

    The correction is the fiddly part, and it is not optional -- vigra (the C++
    library the real ilastik uses) does exactly this, so a kernel built without
    it produces subtly different numbers from a real trained project.

      order 0: rescale so the samples sum to 1, else smoothing would brighten
               or darken the image.
      order >0: first subtract the mean. A derivative kernel should sum to
               zero -- flat input must give zero response -- but truncation
               leaves a small DC offset. Then rescale so the kernel's discrete
               moment matches the true derivative's.
    """
    radius = int(round(window * sigma))
    offsets = list(range(-radius, radius + 1))
    samples = [gaussian_at(x, sigma, order) for x in offsets]

    if order == 0:
        total = sum(samples)
        return [s / total for s in samples]

    mean = sum(samples) / len(samples)
    samples = [s - mean for s in samples]

    moment = sum(((-x) ** order) * s for x, s in zip(offsets, samples)) / math.factorial(order)
    return [s / moment for s in samples]


###############################################################################
#
#  PART 2 -- CONVOLUTION
#
#  One loop, written out. The real implementation calls
#  scipy.ndimage.correlate1d, which does this in C.
#
###############################################################################


def mirror_index(i, length):
    """
    Where to read when a kernel hangs off the edge of the image.

    Reflect at the border WITHOUT repeating the edge pixel, so for a row
    [a b c d] index -1 reads b and index 4 reads c. This particular choice
    matters: it is what vigra calls BORDER_TREATMENT_REFLECT, and picking a
    different one shifts values all along the image border.
    """
    while i < 0 or i >= length:
        if i < 0:
            i = -i
        if i >= length:
            i = 2 * (length - 1) - i
    return i


def convolve_axis(image, kernel, axis):
    """
    Slide `kernel` along one axis of a 2D image, one output pixel at a time.

    Separability is why this is enough: a 2D Gaussian blur is a horizontal
    pass followed by a vertical one, which costs 2k operations per pixel
    instead of k*k. Every filter below is built from passes like this.
    """
    height, width = image.shape
    radius = (len(kernel) - 1) // 2
    out = np.zeros((height, width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            total = 0.0

            for tap, weight in enumerate(kernel):
                offset = tap - radius

                if axis == 0:
                    total += weight * image[mirror_index(y + offset, height), x]
                else:
                    total += weight * image[y, mirror_index(x + offset, width)]

            out[y, x] = total

    return out


def gaussian_derivative(image, sigma, window, y_order, x_order):
    """
    Smooth (or differentiate) along y, then along x.

    (y_order, x_order) = (0, 0) is a plain blur, (0, 1) is the derivative in x,
    (2, 0) the second derivative in y, and so on. Every filter in Part 3 is
    some combination of these.
    """
    out = convolve_axis(image, build_kernel(sigma, y_order, window), axis=0)
    out = convolve_axis(out, build_kernel(sigma, x_order, window), axis=1)
    return out


###############################################################################
#
#  PART 3 -- THE SIX FILTERS
#
#  These are the entries in ilastik's feature-selection grid. Each takes an
#  image and a scale, and returns one or two images of the same size.
#
#  Returning TWO is not a quirk: the eigenvalue filters describe how strongly
#  and in what direction the image curves, which needs two numbers per pixel
#  in 2D (three in 3D). Those become two separate feature columns.
#
###############################################################################


def gaussian_smoothing(image, scale):
    """Plain blur. Answers: how bright is this neighbourhood?"""
    return [gaussian_derivative(image, scale, FEATURE_WINDOW, 0, 0)]


def laplacian_of_gaussian(image, scale):
    """Sum of second derivatives. Answers: is this a blob, and light or dark?"""
    d2y = gaussian_derivative(image, scale, FEATURE_WINDOW, 2, 0)
    d2x = gaussian_derivative(image, scale, FEATURE_WINDOW, 0, 2)
    return [d2y + d2x]


def gaussian_gradient_magnitude(image, scale):
    """Length of the gradient. Answers: how strong is the edge here?"""
    dy = gaussian_derivative(image, scale, FEATURE_WINDOW, 1, 0)
    dx = gaussian_derivative(image, scale, FEATURE_WINDOW, 0, 1)
    return [np.sqrt(dy**2 + dx**2)]


def difference_of_gaussians(image, scale):
    """
    Blur twice, subtract. Answers: what detail lives at roughly this size?

    The 0.66 ratio between the two blurs is ilastik's choice, not a law of
    nature -- see OpPixelFeaturesPresmoothed.
    """
    wide = gaussian_derivative(image, scale, FEATURE_WINDOW, 0, 0)
    narrow = gaussian_derivative(image, scale * 0.66, FEATURE_WINDOW, 0, 0)
    return [wide - narrow]


def eigenvalues_2x2(a, b, c):
    """
    Eigenvalues of the symmetric 2x2 matrix [[a, b], [b, c]], larger first.

    Closed form, from the quadratic formula -- for 2x2 there is no need for a
    general eigensolver, and writing it out keeps the arithmetic visible:

        mean = (a + c) / 2
        spread = sqrt(((a - c) / 2)^2 + b^2)
        eigenvalues = mean +/- spread

    Larger-first ordering is vigra's convention, and the forest was trained on
    columns in that order, so it is not ours to change.
    """
    mean = (a + c) / 2.0
    spread = np.sqrt(((a - c) / 2.0) ** 2 + b**2)
    return [mean + spread, mean - spread]


def structure_tensor_eigenvalues(image, scale):
    """
    Blur the products of first derivatives, then take eigenvalues. Answers:
    is there a consistent orientation around here, or is it a corner?

    Note the scales are used the other way round from what the parameter names
    suggest: the gradients are taken at scale*0.5 and the products smoothed at
    scale. That is not a slip -- fastfilters, the library real ilastik uses,
    swaps innerScale and outerScale between its C header and its Python
    binding, and trained projects carry the consequences. See
    nd_filters.structure_tensor_eigenvalues for the full account.
    """
    gradient_scale = scale * 0.5
    smoothing_scale = scale

    dy = gaussian_derivative(image, gradient_scale, FEATURE_WINDOW, 1, 0)
    dx = gaussian_derivative(image, gradient_scale, FEATURE_WINDOW, 0, 1)

    yy = gaussian_derivative(dy * dy, smoothing_scale, FEATURE_WINDOW, 0, 0)
    yx = gaussian_derivative(dy * dx, smoothing_scale, FEATURE_WINDOW, 0, 0)
    xx = gaussian_derivative(dx * dx, smoothing_scale, FEATURE_WINDOW, 0, 0)

    return eigenvalues_2x2(yy, yx, xx)


def hessian_of_gaussian_eigenvalues(image, scale):
    """
    Eigenvalues of the matrix of second derivatives. Answers: how does the
    image curve here, and how much -- a ridge, a valley, a saddle?
    """
    dyy = gaussian_derivative(image, scale, FEATURE_WINDOW, 2, 0)
    dyx = gaussian_derivative(image, scale, FEATURE_WINDOW, 1, 1)
    dxx = gaussian_derivative(image, scale, FEATURE_WINDOW, 0, 2)

    return eigenvalues_2x2(dyy, dyx, dxx)


FILTERS = {
    "GaussianSmoothing": gaussian_smoothing,
    "LaplacianOfGaussian": laplacian_of_gaussian,
    "GaussianGradientMagnitude": gaussian_gradient_magnitude,
    "DifferenceOfGaussians": difference_of_gaussians,
    "StructureTensorEigenvalues": structure_tensor_eigenvalues,
    "HessianOfGaussianEigenvalues": hessian_of_gaussian_eigenvalues,
}


###############################################################################
#
#  PART 4 -- BUILDING THE FEATURE STACK
#
#  The one genuine surprise in the whole pipeline lives here.
#
###############################################################################


def presmoothing_split(scale):
    """
    ilastik does NOT run a filter at scale 5 directly. It blurs the image
    first, then runs the filter at a small scale on the already-blurred copy,
    exploiting the fact that blurring by a then by b equals blurring by
    sqrt(a^2 + b^2). So to reach scale s it blurs by sqrt(s^2 - 1) and then
    filters at 1.0.

    It does this for speed -- one big blur shared by six filters beats six big
    filters. But it does NOT produce identical numbers to filtering directly at
    s, because the two stages use different kernel widths (3.5 sigmas vs 2.0)
    and so truncate differently. The trained forest learned the values this
    two-stage path produces. Skip it and every feature is slightly off, the
    forest is being fed something it never saw, and the predictions drift.

    Which is the moral of this file: a trained model is a contract with the
    exact code that produced its features.
    """
    if scale > 1.0:
        return math.sqrt(scale**2 - 1.0), 1.0
    else:
        return scale, scale


def compute_feature_stack(channels, selected):
    """
    Run every selected (filter, scale) pair and stack the outputs into one
    array of shape (height, width, n_features).

    Column order is not free. The forest's trees refer to features by column
    number, so the columns have to come out in exactly the order ilastik built
    them in during training: filter-major, then scale, then input channel,
    with a multi-output filter's own channels innermost.

    Returns the stack and a human-readable name per column.
    """
    planes = []
    names = []

    for filter_name, scale in selected:
        blur_sigma, filter_scale = presmoothing_split(scale)

        for channel_index, channel in enumerate(channels):
            # Stage one: the shared blur. (The optimised version computes this
            # once per scale and reuses it across filters; here it is redone
            # every time, which is wasteful and easier to follow.)
            blurred = gaussian_derivative(channel, blur_sigma, PRESMOOTHING_WINDOW, 0, 0)

            # Stage two: the filter itself, at the reduced scale.
            outputs = FILTERS[filter_name](blurred, filter_scale)

            for output_index, plane in enumerate(outputs):
                planes.append(plane)

                name = f"{filter_name} (sigma={scale})"
                if len(channels) > 1:
                    name += f" ch{channel_index}"
                if len(outputs) > 1:
                    name += f" [eigenvalue {output_index}]"
                names.append(name)

    return np.stack(planes, axis=-1), names


###############################################################################
#
#  PART 5 -- READING THE PROJECT FILE
#
#  A .ilp is an HDF5 file. Everything needed for inference sits in three
#  places, and h5py reads them directly.
#
###############################################################################


def read_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def read_selected_features(ilp):
    """
    The feature-selection grid, as a list of (filter_name, scale) pairs in
    ilastik's own column order.

    It is stored as a boolean matrix: one row per filter, one column per scale.
    A ticked box in the GUI is a True here.
    """
    group = ilp["FeatureSelections"]

    filter_names = [read_text(v) for v in group["FeatureIds"][()]]
    scales = [float(s) for s in group["Scales"][()]]
    ticked = np.asarray(group["SelectionMatrix"][()], dtype=bool)

    selected = []
    for row, filter_name in enumerate(filter_names):
        for column, scale in enumerate(scales):
            if ticked[row, column]:
                selected.append((filter_name, scale))

    return selected


def read_forest(ilp):
    """
    The trained forest, as a list of trees.

    vigra stores each tree as two flat arrays, which is compact and completely
    opaque until you know the layout -- see walk_tree() for how they are read.
    Several sub-forests are stored side by side (ilastik trains them in
    parallel); for prediction they are simply one big pool of trees.
    """
    group = ilp["PixelClassification"]["ClassifierForests"]

    trees = []
    for name in sorted(group.keys()):
        if not name.startswith("Forest"):
            continue
        forest = group[name]
        for tree_name in sorted(k for k in forest.keys() if k.startswith("Tree_")):
            tree = forest[tree_name]
            trees.append((tree["topology"][:], tree["parameters"][:]))

    labels = [int(v) for v in group["known_labels"][:]] if "known_labels" in group else None
    return trees, labels


def read_project(path):
    with h5py.File(path, "r") as ilp:
        workflow = read_text(ilp["workflowName"][()])
        if workflow != "Pixel Classification":
            raise ValueError(f"this reader only handles Pixel Classification projects, not {workflow!r}")

        selected = read_selected_features(ilp)
        label_names = [read_text(v) for v in ilp["PixelClassification"]["LabelNames"][()]]
        trees, known_labels = read_forest(ilp)

    if known_labels is None:
        known_labels = list(range(1, len(label_names) + 1))

    return {
        "selected": selected,
        "label_names": label_names,
        "trees": trees,
        "known_labels": known_labels,
    }


###############################################################################
#
#  PART 6 -- THE RANDOM FOREST
#
#  One pixel, one tree, one node at a time.
#
###############################################################################


LEAF_TAG = 0x40000000  # a node's type field has this bit set iff it is a leaf


def walk_tree(topology, parameters, features, trace=None):
    """
    Drop one feature vector down one tree and return the leaf's probabilities.

    vigra's layout, which is the only reason this looks cryptic. `topology` is
    a flat int array; a node at index i occupies the slots

        topology[i]     node type (LEAF_TAG bit set => leaf)
        topology[i + 1] where this node's numbers live in `parameters`
        topology[i + 2] index of the left child   (feature <  threshold)
        topology[i + 3] index of the right child  (feature >= threshold)
        topology[i + 4] which feature column to test

    and the root is always at index 2 (slots 0 and 1 hold the feature count
    and class count). A leaf uses only the first two slots; its parameters hold
    a weight followed by one probability per class.

    Pass a list as `trace` to collect a readable record of the descent.
    """
    class_count = int(topology[1])
    node = 2

    while True:
        if topology[node] & LEAF_TAG:
            address = topology[node + 1]
            # Skip the leading weight, then take one probability per class.
            probabilities = parameters[address + 1 : address + 1 + class_count]

            if trace is not None:
                trace.append(("leaf", [float(p) for p in probabilities]))

            return np.asarray(probabilities)

        address = topology[node + 1]
        left_child = topology[node + 2]
        right_child = topology[node + 3]
        column = topology[node + 4]
        threshold = parameters[address + 1]  # parameters[address] is the node weight

        value = features[column]
        go_left = value < threshold

        if trace is not None:
            trace.append(("node", int(column), float(value), float(threshold), "left" if go_left else "right"))

        node = left_child if go_left else right_child


def classify_pixel(trees, features):
    """
    Average one pixel's leaf probabilities over every tree.

    That average IS the forest's answer -- there is no weighting, no voting
    rule, nothing else. Each tree is a weak opinion; the mean of a hundred of
    them is the prediction.
    """
    total = None

    for topology, parameters in trees:
        probabilities = walk_tree(topology, parameters, features)
        total = probabilities if total is None else total + probabilities

    return total / len(trees)


def to_label_space(probabilities, known_labels, label_count):
    """
    Spread the forest's columns back out over all the project's labels.

    A label the user defined but never actually painted has no training samples
    and so gets no column in the forest. Without this step its channel would be
    missing and every later label would be shifted down into the wrong slot.
    """
    if known_labels == list(range(1, label_count + 1)):
        return probabilities

    full = np.zeros(label_count, dtype=np.float64)
    for column, label in enumerate(known_labels):
        full[label - 1] = probabilities[column]
    return full


###############################################################################
#
#  PART 7 -- THE WHOLE THING
#
###############################################################################


def predict(project, image):
    """
    image: 2D (height, width) or 3D (height, width, channels).

    The entire pipeline, in the order it happens.
    """
    channels = [image] if image.ndim == 2 else [image[..., c] for c in range(image.shape[-1])]

    stack, names = compute_feature_stack(channels, project["selected"])
    height, width, n_features = stack.shape

    label_count = len(project["label_names"])
    output = np.zeros((height, width, label_count), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            features = stack[y, x]
            probabilities = classify_pixel(project["trees"], features)
            output[y, x] = to_label_space(probabilities, project["known_labels"], label_count)

    return output, names


###############################################################################
#
#  PART 8 -- COMMAND LINE
#
###############################################################################


def explain_pixel(project, image, y, x, max_trees):
    """Print the full derivation of one pixel's prediction."""
    channels = [image] if image.ndim == 2 else [image[..., c] for c in range(image.shape[-1])]
    stack, names = compute_feature_stack(channels, project["selected"])
    features = stack[y, x]

    print(f"\n{'=' * 72}\nPIXEL ({y}, {x}) -- raw value {image[y, x]}\n{'=' * 72}")

    width = max(len(n) for n in names)
    print(f"\nFEATURE VECTOR ({len(features)} values)")
    print("-" * 72)
    for column, (name, value) in enumerate(zip(names, features)):
        print(f"  [{column:2d}] {name:{width}s}  {value:12.5f}")

    print(f"\nFOREST ({len(project['trees'])} trees, showing {min(max_trees, len(project['trees']))})")
    print("-" * 72)

    total = None
    for index, (topology, parameters) in enumerate(project["trees"]):
        trace = [] if index < max_trees else None
        probabilities = walk_tree(topology, parameters, features, trace=trace)
        total = probabilities if total is None else total + probabilities

        if trace is None:
            continue

        print(f"\n  Tree {index}:")
        for step in trace:
            if step[0] == "node":
                _, column, value, threshold, direction = step
                comparison = "<" if direction == "left" else ">="
                print(
                    f"    feature [{column:2d}] {value:10.5f} {comparison:2s} {threshold:10.5f}"
                    f"  -> {direction:5s}   ({names[column]})"
                )
            else:
                print(f"    leaf: {[round(p, 4) for p in step[1]]}")

    averaged = total / len(project["trees"])
    final = to_label_space(averaged, project["known_labels"], len(project["label_names"]))

    print(f"\n{'-' * 72}\nAVERAGED OVER ALL {len(project['trees'])} TREES")
    print("-" * 72)
    for label_name, probability in zip(project["label_names"], final):
        bar = "#" * int(round(probability * 40))
        print(f"  {label_name:20s} {probability:6.4f}  {bar}")
    print()


def load_image(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)

    try:
        import imageio.v3 as iio
    except ImportError:
        raise SystemExit("need imageio to read image files: pip install imageio (or pass a .npy)")

    return np.asarray(iio.imread(path)).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="ilastik Pixel Classification, unoptimised and readable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("SCOPE")[0].strip(),
    )
    parser.add_argument("project", help="a trained .ilp file")
    parser.add_argument("image", help="image to classify (anything imageio reads, or .npy)")
    parser.add_argument("--crop", type=int, metavar="N", help="use only the top-left NxN corner (do use this)")
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="use image[Y0:Y1, X0:X1] instead; --explain coordinates are then relative to it",
    )
    parser.add_argument("--explain", nargs=2, type=int, metavar=("Y", "X"), help="trace one pixel instead of predicting")
    parser.add_argument("--max-trees", type=int, default=3, help="how many trees --explain prints in full (default 3)")
    parser.add_argument("-o", "--output", help="save probabilities to a .npy file")
    args = parser.parse_args()

    project = read_project(args.project)
    image = load_image(args.image)

    if args.region:
        y0, y1, x0, x1 = args.region
        image = image[y0:y1, x0:x1]
    elif args.crop:
        image = image[: args.crop, : args.crop]

    print(f"project:  {args.project}")
    print(f"labels:   {project['label_names']}")
    print(f"features: {len(project['selected'])} filter/scale pairs, {len(project['trees'])} trees")
    print(f"image:    {image.shape}")

    if image.size == 0:
        raise SystemExit("that crop/region is empty")

    if args.explain:
        y, x = args.explain
        if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
            raise SystemExit(f"pixel ({y}, {x}) is outside the {image.shape[0]}x{image.shape[1]} image")
        explain_pixel(project, image, y, x, args.max_trees)
        return

    pixels = image.shape[0] * image.shape[1]
    if pixels > 10000:
        print(f"\nWARNING: {pixels} pixels, one at a time. This will take a while -- try --crop 32.\n")

    probabilities, _ = predict(project, image)

    print(f"\nprobabilities: {probabilities.shape}")
    for index, label_name in enumerate(project["label_names"]):
        channel = probabilities[..., index]
        print(f"  {label_name:20s} mean {channel.mean():.4f}   range {channel.min():.4f} - {channel.max():.4f}")

    if args.output:
        np.save(args.output, probabilities)
        print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
