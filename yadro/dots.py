import numpy as np
import matplotlib.pyplot as plt


def validate_points(points):
    if not (2 <= len(points) <= 8):
        raise ValueError("Количество точек должно быть от 2 до 8")

    for x, y in points:
        if not (0 <= x <= 1):
            raise ValueError("Координата X должна быть от 0 до 1")
        if not (-1 <= y <= 1):
            raise ValueError("Координата Y должна быть от -1 до 1")


def geometric_median(points, eps=1e-7, max_iter=10000):
    points = np.array(points, dtype=float)

    current = np.mean(points, axis=0)

    for _ in range(max_iter):
        distances = np.linalg.norm(points - current, axis=1)

        if np.any(distances < eps):
            return current

        weights = 1 / distances

        next_point = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)

        if np.linalg.norm(next_point - current) < eps:
            return next_point

        current = next_point

    return current


def plot_points(points, result_point, output_file="result.png"):
    points = np.array(points)

    plt.figure(figsize=(7, 6))

    plt.scatter(points[:, 0], points[:, 1], label="Исходные точки")

    plt.scatter(
        result_point[0],
        result_point[1],
        color="red",
        s=120,
        label="Точка с минимальной суммой расстояний"
    )

    for i, (x, y) in enumerate(points):
        plt.text(x + 0.01, y + 0.01, f"P{i + 1}")

    plt.text(
        result_point[0] + 0.01,
        result_point[1] + 0.01,
        "Result",
        color="red"
    )

    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Поиск точки с минимальной дистанцией")
    plt.grid(True)
    plt.legend()

    plt.savefig(output_file)
    plt.show()


def main():
    n = int(input("Введите количество точек от 2 до 8: "))

    points = []

    print("Введите координаты точек в формате: x y")

    for i in range(n):
        x, y = map(float, input(f"Точка {i + 1}: ").split())
        points.append((x, y))

    validate_points(points)

    result_point = geometric_median(points)

    print("Результирующая точка:")
    print(f"x = {result_point[0]:.6f}")
    print(f"y = {result_point[1]:.6f}")

    plot_points(points, result_point)


if __name__ == "__main__":
    main()