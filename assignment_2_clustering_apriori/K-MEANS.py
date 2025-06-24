import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt

# ===============================
# Data Setup
# ===============================

points = [
    ("N1", (0, 0)), ("N2", (0, 7.5)), ("N3", (0, 15)), ("N4", (5, 7.5)),
    ("N5", (10, 0)), ("N6", (10, 7.5)), ("N7", (10, 15)), ("A1", (20, 0)),
    ("A2", (22.5, 7.5)), ("A3", (25, 15)), ("A4", (24, 7.5)), ("A5", (26, 7.5)),
    ("A6", (27.5, 7.5)), ("A7", (30, 0)), ("V1", (40, 15)), ("V2", (45, 0)),
    ("V3", (42, 9)), ("V4", (44, 3)), ("V5", (48, 9)), ("V6", (50, 15)),
    ("V7", (46, 3))
]

# Extract just the coordinates into a NumPy array
point_coords = np.array([coord for _, coord in points])
point_labels = [label for label, _ in points]

# ===============================
# Helper Functions
# ===============================

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def plot_clusters(clusters, centroids, initial_centroids, title):
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    plt.figure(figsize=(8, 6))

    # Plot clusters
    for idx, cluster in enumerate(clusters):
        if cluster:
            cluster_points = np.array([point[1] for point in cluster])
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[idx], label=f'Cluster {idx + 1}')
            # Annotate points with labels
            for label, (x, y) in cluster:
                plt.text(x + 0.5, y + 0.5, label, fontsize=9)

    # Plot initial centroids (in a different style)
    initial_centroids = np.array(initial_centroids)
    plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='purple', marker='o', s=100, label='Initial Centroids')

    # Plot final centroids
    final_centroids = np.array(centroids)
    plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='black', marker='X', s=200, label='Final Centroids')

    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_centroids(centroids):
    for idx, centroid in enumerate(centroids):
        print(f"Centroid {idx}: {np.round(centroid, 2)}")

# ===============================
# K-Means Algorithm
# ===============================

def kmeans(points, initial_centroids, run_name):
    print(f"\n=== {run_name}: Initial Centroids ===")
    centroids = initial_centroids.copy()
    print_centroids(centroids)

    iterations = 0
    clusters_history = []  # To track clusters across iterations

    while True:
        print(f"\n--- Iteration {iterations + 1} ---")

        # Assignment step
        clusters = [[] for _ in range(len(centroids))]
        for label, point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append((label, point))

            # Detailed distance calculations and assignments
            distance_str = ', '.join([f"{round(d, 2)}" for d in distances])
            print(f"Point {label} at {point} distances to centroids: [{distance_str}], assigned to Cluster {closest_centroid}")

        # Update step
        new_centroids = []
        for idx, cluster in enumerate(clusters):
            if cluster:
                cluster_points = np.array([point for _, point in cluster])
                new_centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(new_centroid)
            else:
                # Handle empty cluster by random repositioning (if needed)
                new_centroids.append(np.random.rand(2) * 50)

        new_centroids = np.array(new_centroids)
        print("\nUpdated Centroids:")
        print_centroids(new_centroids)

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            print("\nCentroids stabilized. K-Means converged!")
            break

        centroids = new_centroids
        iterations += 1
        clusters_history.append(clusters)

    # Final cluster assignments
    print("\n=== Final Cluster Assignments ===")
    for cluster_idx, cluster in enumerate(clusters):
        labels_in_cluster = [label for label, _ in cluster]
        print(f"Cluster {cluster_idx}: {', '.join(labels_in_cluster)}")

    # Plotting the results
    plot_clusters(clusters, centroids, initial_centroids, f"K-Means Clustering Result: {run_name}")
    return clusters, centroids

# ===============================
# Main Execution
# ===============================

# Good initial centroids (near true centers of N, A, V)
good_initial_centroids = np.array([
    [5, 7.5],     # N cluster
    [25, 7.5],    # A cluster
    [45, 7.5]     # V cluster
])

# Bad initial centroids (intentionally suboptimal)
bad_initial_centroids = np.array([
    [0, 0],
    [5, 5],
    [45, 5]
])

# Run K-Means with good initial centroids
kmeans(points, good_initial_centroids, "K-Means with Good Initial Centroids")

# Run K-Means with bad initial centroids
kmeans(points, bad_initial_centroids, "K-Means with Bad Initial Centroids")
