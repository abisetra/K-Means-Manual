import numpy as np

def input_data():
    while True:
        try:
            JumlahData = int(input("Masukkan Jumlah data: "))
            break
        except ValueError:
            print("Masukan data integer.")
    numbers = []
    for i in range(JumlahData):
        while True:
            try:
                x = float(input(f"Masukkan variabel X untuk Data Ke-{i+1}: "))
                y = float(input(f"Masukkan variabel Y untuk Data Ke-{i+1}: "))
                numbers.append([x, y])
                break
            except ValueError:
                print("Masukkan angka yang valid.")
    return np.array(numbers)

def select_centroids(data, n_clusters):
    # Inisialisasi centroid secara acak dari data
    indices = np.random.choice(data.shape[0], size=n_clusters, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    # Hitung jarak antara setiap titik data dan setiap centroid
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    # Assign setiap titik data ke cluster dengan centroid terdekat
    return np.argmin(distances, axis=0)

def update_centroids(data, assignments, n_clusters):
    # Hitung centroid baru sebagai rata-rata titik data dalam setiap cluster
    return np.array([data[assignments == i].mean(axis=0) for i in range(n_clusters)])

def compute_kmeans(data, n_clusters, n_iterations):
    centroids = select_centroids(data, n_clusters)
    for _ in range(n_iterations):
        assignments = assign_clusters(data, centroids)
        centroids = update_centroids(data, assignments, n_clusters)
    return assignments, centroids

def main():
    data = input_data()
    assignments, centroids = compute_kmeans(data, n_clusters=3, n_iterations=10)
    for i in range(len(data)):
        print(f"Data Ke-{i+1} merupakan Cluster {assignments[i]+1}")

if __name__ == "__main__":
    main()