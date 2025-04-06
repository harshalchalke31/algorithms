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

        self.distance_matrix = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                self.distance_matrix[i, j] = abs(signal1[i] - signal2[j])

        self.cost_matrix = np.full((N + 1, M + 1), np.inf)
        self.cost_matrix[0, 0] = self.distance_matrix[0, 0]

        traceback = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                choices = [
                    self.cost_matrix[i, j],      # match (0)
                    self.cost_matrix[i, j + 1],  # insertion (1)
                    self.cost_matrix[i + 1, j]   # deletion (2)
                ]
                min_choice = np.argmin(choices)
                self.cost_matrix[i + 1, j + 1] = self.distance_matrix[i, j] + choices[min_choice]
                traceback[i, j] = min_choice

        self.dtw_distance = self.cost_matrix[N, M]

        i, j = N - 1, M - 1
        self.path = [(i, j)]
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                tb_direction = traceback[i, j]
                if tb_direction == 0:
                    i -= 1
                    j -= 1
                elif tb_direction == 1:
                    i -= 1
                elif tb_direction == 2:
                    j -= 1
            self.path.append((i, j))
        self.path.reverse()

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
