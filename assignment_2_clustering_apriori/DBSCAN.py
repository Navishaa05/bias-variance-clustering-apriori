import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Define the points as provided
points = [
    ("N1", (0, 0)), ("N2", (0, 7.5)), ("N3", (0, 15)), ("N4", (5, 7.5)),
    ("N5", (10, 0)), ("N6", (10, 7.5)), ("N7", (10, 15)), ("A1", (20, 0)),
    ("A2", (22.5, 7.5)), ("A3", (25, 15)), ("A4", (24, 7.5)), ("A5", (26, 7.5)),
    ("A6", (27.5, 7.5)), ("A7", (30, 0)), ("V1", (40, 15)), ("V2", (45, 0)),
    ("V3", (42, 9)), ("V4", (44, 3)), ("V5", (48, 9)), ("V6", (50, 15)),
    ("V7", (46, 3))
]

# Extract labels and coordinates
labels = [p[0] for p in points]
coords = np.array([p[1] for p in points])

# Calculate pairwise distances
distances = squareform(pdist(coords, metric='euclidean'))

# Create a DataFrame for better visualization of the distance matrix
distance_df = pd.DataFrame(distances, index=labels, columns=labels)

# Display the distance matrix
print("Distance Matrix:")
print(distance_df.round(2))

# DBSCAN parameters
epsilon = 10  # Changed from 10 to 9 as requested
min_points = 3  # Minimum points to form a cluster

print(f"\nDBSCAN Parameters:")
print(f"Epsilon: {epsilon}")
print(f"MinPoints: {min_points}")

# Step 1: Find all core points
print("\nStep 1: Finding core points...")
core_points = []
for i, point in enumerate(labels):
    # Count how many points are within epsilon distance
    neighbors = [labels[j] for j in range(len(labels)) if distances[i, j] <= epsilon]
    if len(neighbors) >= min_points:
        core_points.append(point)
        print(f"  {point} is a core point with neighbors: {', '.join(neighbors)}")

print(f"\nCore points: {', '.join(core_points)}")

# Step 2: Find initial clusters based on each core point
print("\nStep 2: Finding initial clusters for each core point...")
initial_clusters = {}
cluster_id = 0

for core in core_points:
    core_idx = labels.index(core)
    # Find all points within epsilon distance of this core point
    cluster_members = [labels[j] for j in range(len(labels)) if distances[core_idx, j] <= epsilon]
    
    print(f"  Initial cluster for core {core}: {', '.join(cluster_members)}")
    initial_clusters[cluster_id] = set(cluster_members)
    cluster_id += 1

# Step 3: Merge overlapping clusters
print("\nStep 3: Merging overlapping clusters...")
merged = True
iteration = 1

# We'll keep merged cluster IDs in this dictionary
merged_into = {}
active_clusters = list(initial_clusters.keys())

while merged:
    merged = False
    print(f"  Iteration {iteration}:")
    
    # Check each pair of active clusters
    for i in range(len(active_clusters)):
        for j in range(i+1, len(active_clusters)):
            ci = active_clusters[i]
            cj = active_clusters[j]
            
            # Skip if either cluster has already been merged
            if ci in merged_into or cj in merged_into:
                continue
            
            # Check if clusters share any points
            if initial_clusters[ci].intersection(initial_clusters[cj]):
                # Merge cj into ci
                print(f"    Merging cluster {cj} into cluster {ci} as they share points")
                initial_clusters[ci] = initial_clusters[ci].union(initial_clusters[cj])
                merged_into[cj] = ci
                merged = True
    
    # Update active clusters
    active_clusters = [c for c in active_clusters if c not in merged_into]
    iteration += 1
    
    if not merged:
        print("    No more clusters to merge")

# Step 4: Assign each point to its final cluster
print("\nStep 4: Assigning points to final clusters...")
cluster_assignments = {}
point_to_cluster = {}

# Initialize with -1 (noise)
for point in labels:
    point_to_cluster[point] = -1

# Assign points to final clusters
cluster_counter = 0
for cluster_id in initial_clusters:
    # Skip if this cluster was merged into another
    if cluster_id in merged_into:
        continue
    
    print(f"  Final cluster {cluster_counter}: {', '.join(initial_clusters[cluster_id])}")
    
    # Assign points to this cluster
    for point in initial_clusters[cluster_id]:
        point_to_cluster[point] = cluster_counter
    
    # Add to cluster assignments
    cluster_assignments[cluster_counter] = list(initial_clusters[cluster_id])
    cluster_counter += 1

# Find noise points (not assigned to any cluster)
noise_points = [point for point, cluster in point_to_cluster.items() if cluster == -1]
if noise_points:
    print(f"  Noise points: {', '.join(noise_points)}")
    cluster_assignments[-1] = noise_points

# Step 5: Summary of final clusters
print("\nFinal DBSCAN Clustering Results:")
print(f"Number of clusters: {len(cluster_assignments) - (1 if -1 in cluster_assignments else 0)}")
print(f"Number of noise points: {len(noise_points) if noise_points else 0}")

print("\nCluster assignments:")
for cluster_id, members in cluster_assignments.items():
    if cluster_id == -1:
        print(f"Noise points: {', '.join(members)}")
    else:
        print(f"Cluster {cluster_id}: {', '.join(members)}")

# Visualize the clusters
plt.figure(figsize=(12, 8))

# Define colors for clusters (excluding noise)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']

# Plot the points
for i, label in enumerate(labels):
    cluster_id = point_to_cluster[label]
    color = colors[cluster_id % len(colors)] if cluster_id >= 0 else 'gray'
    marker = 'o' if cluster_id >= 0 else 'x'
    x, y = coords[i]
    plt.scatter(x, y, color=color, marker=marker, s=100)
    plt.text(x+0.5, y+0.5, label, fontsize=12)

# Highlight core points with a circle
for core in core_points:
    i = labels.index(core)
    x, y = coords[i]
    cluster_id = point_to_cluster[core]
    color = colors[cluster_id % len(colors)] if cluster_id >= 0 else 'gray'
    plt.scatter(x, y, color=color, marker='o', s=150, edgecolors='black', linewidths=2)

# Plot the epsilon radius for core points
for core in core_points:
    i = labels.index(core)
    plt.gca().add_patch(plt.Circle(coords[i], epsilon, fill=False, linestyle='--', alpha=0.3))

plt.grid(True)
plt.title(f'DBSCAN Clustering (ε={epsilon}, MinPts={min_points})')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.tight_layout()
plt.show()