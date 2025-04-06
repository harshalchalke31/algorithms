import numpy as np
import matplotlib.pyplot as plt

class DTW:
    def __init__(self, signal1: np.array, signal2: np.array, normalized: bool = True):
        self.signal1 = signal1
        self.signal2 = signal2
        self.normalized = normalized
        self.distance_matrix = None
        self.cost_matrix = None
        self.path = None
        self.dtw_distance = None


    def compute(self):
        signal1, signal2 = self.signal1, self.signal2
        N, M = len(signal1), len(signal2)

        # 1. Create the distance matrix as before
        self.distance_matrix = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                self.distance_matrix[i, j] = abs(signal1[i] - signal2[j])

        # 2. Allocate the cost matrix (+1 in each dimension) and traceback matrix
        self.cost_matrix = np.full((N + 1, M + 1), np.inf)
        traceback = np.zeros((N + 1, M + 1))

        # 3. **Change**: Set cost_matrix[0, 0] = 0 instead of distance_matrix[0,0]
        self.cost_matrix[0, 0] = 0.0

        # 4. **Change**: Initialize the first row and the first column properly
        #    so the path can start from (0,0) and then extend.
        for i in range(1, N + 1):
            self.cost_matrix[i, 0] = self.cost_matrix[i - 1, 0] + self.distance_matrix[i - 1, 0]
            traceback[i, 0] = 0  # means we came from (i-1, 0)

        for j in range(1, M + 1):
            self.cost_matrix[0, j] = self.cost_matrix[0, j - 1] + self.distance_matrix[0, j - 1]
            traceback[0, j] = 1  # means we came from (0, j-1)

        # 5. **Change**: Fill in the cost matrix using the standard min of neighbors
        #    cost[i,j] = distance_matrix[i-1,j-1] + min(...).
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                dist = self.distance_matrix[i - 1, j - 1]
                # Check the three possible previous steps
                costs = [
                    self.cost_matrix[i - 1, j],    # came from top
                    self.cost_matrix[i, j - 1],    # came from left
                    self.cost_matrix[i - 1, j - 1] # came from diagonal
                ]
                min_cost = min(costs)
                self.cost_matrix[i, j] = dist + min_cost

                # Record which direction gave the min cost
                if min_cost == costs[0]:
                    traceback[i, j] = 0  # from top
                elif min_cost == costs[1]:
                    traceback[i, j] = 1  # from left
                else:
                    traceback[i, j] = 2  # from diagonal

        # 6. The final DTW distance is at cost_matrix[N, M]
        self.dtw_distance = self.cost_matrix[N, M]

        # 7. **Change**: Traceback from (N, M) down to (0, 0)
        i, j = N, M
        self.path = []
        while i > 0 or j > 0:
            self.path.append((i - 1, j - 1))  # store the (signal1 index, signal2 index)

            direction = traceback[i, j]
            if direction == 0:
                i -= 1  # came from top
            elif direction == 1:
                j -= 1  # came from left
            else:
                i -= 1
                j -= 1

        self.path.reverse()

        # 8. Normalize by path length if requested
        if self.normalized:
            return self.dtw_distance / len(self.path)
        else:
            return self.dtw_distance

    def visualize(self):
        if self.distance_matrix is None or self.cost_matrix is None or self.path is None:
            raise ValueError("Run compute() before calling visualize().")

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.title("Distance matrix")
        plt.imshow(self.distance_matrix, cmap=plt.cm.binary, interpolation="nearest", origin="lower")

        plt.subplot(2, 2, 2)
        plt.title("Cost matrix")
        plt.imshow(self.cost_matrix, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
        x_path, y_path = zip(*self.path)
        plt.plot(y_path, x_path)

        plt.figure(figsize=(10, 6))
        plt.title("Signal Overlap (DTW Alignment)")
        for x_i, y_j in self.path:
            plt.plot([x_i, y_j], [self.signal1[x_i] + 1.5, self.signal2[y_j] - 1.5], c="C7")
        plt.plot(np.arange(len(self.signal1)), self.signal1 + 1.5, "-.", c="C3", label="Signal 1")
        plt.plot(np.arange(len(self.signal2)), self.signal2 - 1.5, "-.", c="C0", label="Signal 2")
        plt.axis("off")
        plt.legend()
        plt.show()
