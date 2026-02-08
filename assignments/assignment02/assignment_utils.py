import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def generate_checkerboard(n_samples=2000, grid_size=4, noise=0.05, seed=42):
    """
    Generate a checkerboard pattern dataset.

    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate
    grid_size : int
        Number of squares per side (e.g., 4 = 4x4 = 16 squares)
    noise : float
        Standard deviation of Gaussian noise added to coordinates
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Feature matrix with x, y coordinates
    y : ndarray of shape (n_samples,)
        Binary class labels (0 or 1)
    """
    np.random.seed(seed)

    # Generate random points in [0, grid_size] x [0, grid_size]
    X = np.random.uniform(0, grid_size, size=(n_samples, 2))

    # Add noise
    X += np.random.normal(0, noise, size=X.shape)

    # Determine class based on checkerboard pattern
    # Class = (floor(x) + floor(y)) mod 2
    x_bin = np.floor(X[:, 0]).astype(int)
    y_bin = np.floor(X[:, 1]).astype(int)
    y = (x_bin + y_bin) % 2

    return X, y


def load_dry_bean():
    """
    Load the Dry Bean dataset from UCI Machine Learning Repository.

    Returns:
    --------
    X : ndarray
        Feature matrix (n_samples, 16)
    y : ndarray
        Class labels (n_samples,)
    feature_names : list
        Names of the 16 features
    class_names : list
        Names of the 7 bean classes
    """
    import os

    # Create .cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "dry_bean_cache.csv")

    # Try to load from local cache first
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        # Try multiple sources
        loaded = False

        # Method 1: Try UCI ML Repository direct CSV link
        try:
            print("Downloading Dry Bean dataset from UCI...")
            url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
            import zipfile
            import io
            import urllib.request

            with urllib.request.urlopen(url, timeout=30) as response:
                zip_data = io.BytesIO(response.read())

            with zipfile.ZipFile(zip_data, "r") as zip_ref:
                for name in zip_ref.namelist():
                    if name.endswith(".csv"):
                        with zip_ref.open(name) as f:
                            df = pd.read_csv(f)
                            loaded = True
                            break
                    elif name.endswith(".arff"):
                        # Parse ARFF file
                        with zip_ref.open(name) as f:
                            content = f.read().decode("utf-8")
                            df = _parse_arff(content)
                            loaded = True
                            break
        except Exception as e:
            print(f"UCI download failed: {e}")

        # Method 2: Try OpenML with different dataset ID
        if not loaded:
            try:
                print("Trying OpenML...")
                from sklearn.datasets import fetch_openml

                # Dry Bean Dataset on OpenML has data_id=42980
                data = fetch_openml(data_id=42980, as_frame=True, parser="auto")
                df = data.frame
                loaded = True
            except Exception as e:
                print(f"OpenML failed: {e}")

        # Method 3: Generate a similar synthetic dataset as fallback
        if not loaded:
            print(
                "All download methods failed. Generating synthetic bean-like dataset..."
            )
            df = _generate_synthetic_bean_data()

        # Cache for future use
        df.to_csv(cache_path, index=False)
        print("Dataset cached locally.")

    # Extract features and target
    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].values.astype(float)
    y_raw = df[target_col].values

    # Encode class labels as integers
    class_names = sorted(list(set(y_raw)))
    class_to_int = {name: i for i, name in enumerate(class_names)}
    y = np.array([class_to_int[label] for label in y_raw])

    feature_names = feature_cols

    return X, y, feature_names, class_names


def _parse_arff(content):
    """Parse ARFF file content into a DataFrame."""
    lines = content.split("\n")
    data_started = False
    data_rows = []
    attributes = []

    for line in lines:
        line = line.strip()
        if line.lower().startswith("@attribute"):
            parts = line.split()
            attr_name = parts[1].strip("'\"")
            attributes.append(attr_name)
        elif line.lower() == "@data":
            data_started = True
        elif data_started and line and not line.startswith("%"):
            values = line.split(",")
            data_rows.append(values)

    df = pd.DataFrame(data_rows, columns=attributes)
    return df


def _generate_synthetic_bean_data(n_samples=13611, seed=42):
    """
    Generate synthetic data similar to Dry Bean dataset structure.
    Used as fallback if download fails.
    """
    np.random.seed(seed)

    class_names = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
    n_classes = len(class_names)
    samples_per_class = n_samples // n_classes

    feature_names = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRatio",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "Roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
    ]

    all_data = []
    all_labels = []

    # Generate different distributions for each class
    for i, class_name in enumerate(class_names):
        # Base parameters vary by class
        base_area = 30000 + i * 10000 + np.random.randn(samples_per_class) * 5000
        base_perimeter = 500 + i * 50 + np.random.randn(samples_per_class) * 50

        data = np.column_stack(
            [
                base_area,  # Area
                base_perimeter,  # Perimeter
                np.sqrt(base_area) * (1.5 + i * 0.1)
                + np.random.randn(samples_per_class) * 10,  # MajorAxisLength
                np.sqrt(base_area) * (0.8 + i * 0.05)
                + np.random.randn(samples_per_class) * 5,  # MinorAxisLength
                1.5 + i * 0.1 + np.random.randn(samples_per_class) * 0.1,  # AspectRatio
                0.6
                + i * 0.03
                + np.random.randn(samples_per_class) * 0.05,  # Eccentricity
                base_area * 1.05
                + np.random.randn(samples_per_class) * 1000,  # ConvexArea
                np.sqrt(base_area / np.pi) * 2
                + np.random.randn(samples_per_class) * 5,  # EquivDiameter
                0.7 + np.random.randn(samples_per_class) * 0.05,  # Extent
                0.98 + np.random.randn(samples_per_class) * 0.01,  # Solidity
                0.8 + i * 0.02 + np.random.randn(samples_per_class) * 0.05,  # Roundness
                0.7
                + i * 0.02
                + np.random.randn(samples_per_class) * 0.05,  # Compactness
                0.005 + np.random.randn(samples_per_class) * 0.001,  # ShapeFactor1
                0.001 + np.random.randn(samples_per_class) * 0.0002,  # ShapeFactor2
                0.5
                + i * 0.05
                + np.random.randn(samples_per_class) * 0.05,  # ShapeFactor3
                0.95 + np.random.randn(samples_per_class) * 0.02,  # ShapeFactor4
            ]
        )

        all_data.append(data)
        all_labels.extend([class_name] * samples_per_class)

    X = np.vstack(all_data)
    df = pd.DataFrame(X, columns=feature_names)
    df["Class"] = all_labels[: len(df)]

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def plot_decision_boundary(clf, X, y, ax=None, title="Decision Boundary", cmap="RdBu"):
    """
    Plot the decision boundary of a classifier for 2D data.

    Parameters:
    -----------
    clf : fitted classifier (can be a Pipeline)
        Must have a predict method
    X : ndarray of shape (n_samples, 2)
        Feature matrix (2D only)
    y : ndarray
        True class labels
    ax : matplotlib axis (optional)
        Axis to plot on
    title : str
        Plot title
    cmap : str
        Colormap for decision regions

    Returns:
    --------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict on mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, levels=1, cmap=cmap, alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)

    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=30, alpha=0.7)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)

    return ax


def print_evaluation_metrics(y_true, y_pred, model_name="Model"):
    """
    Print classification metrics in a standardized format.

    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    model_name : str
        Name to display in the output

    Returns:
    --------
    report : dict
        Dictionary containing the classification report
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"{model_name} - Test Set Results")
    print(f"{'=' * 50}")
    print(f"Test Accuracy:      {report['accuracy']:.4f}")
    print(f"Macro Precision:    {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall:       {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-Score:     {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score:  {report['weighted avg']['f1-score']:.4f}")
    print(f"{'=' * 50}\n")

    return report


def print_grid_search_results(grid_search, model_name="Model"):
    """
    Print grid search results in a standardized format.

    Parameters:
    -----------
    grid_search : GridSearchCV object
        Fitted grid search object
    model_name : str
        Name to display in the output
    """
    print(f"\n{'=' * 50}")
    print(f"{model_name} - Grid Search Results")
    print(f"{'=' * 50}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print("Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"{'=' * 50}\n")


def plot_grid_search_heatmap(grid_search, param_x, param_y, title=None, ax=None):
    """
    Plot a heatmap of cross-validation scores from a grid search.

    Parameters:
    -----------
    grid_search : GridSearchCV object
        Fitted grid search object
    param_x : str
        Name of the parameter for the x-axis (columns), e.g., "svc__C"
    param_y : str
        Name of the parameter for the y-axis (rows), e.g., "svc__gamma"
    title : str, optional
        Title for the plot. If None, a default title is generated.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure is created.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object with the heatmap
    """
    import pandas as pd
    import seaborn as sns

    results = pd.DataFrame(grid_search.cv_results_)

    heatmap_data = results.pivot_table(
        index=f"param_{param_y}",
        columns=f"param_{param_x}",
        values="mean_test_score",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax)

    if title is None:
        title = f"CV Accuracy — {param_y.split('__')[-1]} vs {param_x.split('__')[-1]}"
    ax.set_title(title)
    ax.set_xlabel(param_x.split("__")[-1])
    ax.set_ylabel(param_y.split("__")[-1])

    plt.tight_layout()
    return ax


def plot_roc_curves(y_true, y_proba, class_names=None, title=None, ax=None):
    """
    Plot ROC curves and compute ROC-AUC score.

    Handles both binary and multi-class classification (One-vs-Rest).

    Parameters:
    -----------
    y_true : array-like
        True class labels (integer encoded)
    y_proba : array-like
        Predicted probabilities. For binary: shape (n_samples,) or (n_samples, 2).
        For multi-class: shape (n_samples, n_classes).
    class_names : list, optional
        Names of the classes (for multi-class legend). If None, uses class indices.
    title : str, optional
        Title for the plot. If None, uses "ROC Curve".
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure is created.

    Returns:
    --------
    roc_auc : float
        ROC-AUC score (macro average for multi-class)
    ax : matplotlib.axes.Axes
        The axes object with the ROC curve
    """
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import RocCurveDisplay, roc_auc_score

    # Determine if binary or multi-class
    y_proba = np.atleast_2d(y_proba)
    if y_proba.shape[0] == 1:
        y_proba = y_proba.T
    n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 1

    # Handle binary case where y_proba might be (n_samples, 2)
    is_binary = len(np.unique(y_true)) == 2

    if ax is None:
        figsize = (6, 5) if is_binary else (8, 6)
        fig, ax = plt.subplots(figsize=figsize)

    if is_binary:
        # Binary classification
        if y_proba.ndim > 1 and y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba.ravel()

        RocCurveDisplay.from_predictions(y_true, y_proba_pos, ax=ax)
        roc_auc = roc_auc_score(y_true, y_proba_pos)
    else:
        # Multi-class (One-vs-Rest)
        y_true_bin = LabelBinarizer().fit_transform(y_true)

        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        for i, name in enumerate(class_names):
            RocCurveDisplay.from_predictions(
                y_true_bin[:, i], y_proba[:, i], name=name, ax=ax
            )

        roc_auc = roc_auc_score(y_true_bin, y_proba, average="macro")
        ax.legend(loc="lower right", fontsize=8)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    if title is None:
        title = "ROC Curve" if is_binary else "ROC Curves (One-vs-Rest)"
    ax.set_title(title)

    plt.tight_layout()

    return roc_auc, ax
