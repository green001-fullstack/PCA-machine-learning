import numpy as np
import matplotlib.pyplot as plt


X = np.array([
    [2.0, 2.2],
    [3.0, 3.1],
    [4.0, 4.2],
    [5.0, 4.8],
    [6.0, 6.1],
    [7.0, 6.9],
    [8.0, 8.2],
    [9.0, 8.8],
])

def pca(X, n_components):
    # 1. Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 2. Calculate XᵀX
    XTX = X_centered.T @ X_centered

    # 3. Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(XTX)
    # print("Eigenvalues:", eigenvalues)

    # 4. Sort eigenvalues from largest to smallest
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # print(sorted_indices)

    eigenvalues = eigenvalues[sorted_indices]
    # print(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices]

    # 5. Select the principal components
    components = eigenvectors[:, :n_components]
    # print(components)

    # 6. Project the data onto the selected components
    X_reduced = X_centered @ components

    return X_reduced, components, eigenvalues

X_reduced, components, eigenvalues = pca(X, 1)
print("X_reduced:", X_reduced)
print(X_reduced.shape)
print(components.shape)
X_reconstructed = X_reduced @ components.T + np.mean(X, axis=0)


# Center the data for visualization
# Center the data for visualization
mean = np.mean(X, axis=0)
X_centered = X - mean

# Principal direction
v1 = components[:, 0]

# Projected points
X_projected = X_reduced * v1

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Original centered data points
ax.scatter(
    X_centered[:, 0],
    X_centered[:, 1],
    label="Original points"
)

# Principal component line
t = np.linspace(-6, 6, 100)
line = np.outer(t, v1)

ax.plot(
    line[:, 0],
    line[:, 1],
    label="Principal component"
)

# Projected points
ax.scatter(
    X_projected[:, 0],
    X_projected[:, 1],
    label="Projected points"
)

# Projection lines
for original, projected in zip(X_centered, X_projected):
    ax.plot(
        [original[0], projected[0]],
        [original[1], projected[1]],
        linestyle="--"
    )

# Axes
ax.axhline(0)
ax.axvline(0)

# Labels and title
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("PCA Projection")

ax.legend()
ax.set_aspect("equal")

plt.show()