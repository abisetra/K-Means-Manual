import sys, os
sys.path.append("C:/Users/abiyy/Downloads/Pemroraman_Python/")
os.environ['OMP_NUM_THREADS'] = '1'

import modul

def main():
    data = modul.input_data()
    while True:
        try:
            n_clusters = int(input("Masukkan jumlah centroid: ")) #
            break
        except ValueError:
            print("Masukan data integer.")
    while True:
        try:
            iterations = int(input("Masukkan jumlah pengulangan/iterasi: "))
            break
        except ValueError:
            print("Masukan data integer.")
    assignments, centroids = modul.compute_kmeans(data, n_clusters, iterations)
    for i in range(len(data)):
        print(f"Data Ke-{i+1} merupakan Cluster {assignments[i]+1}")

if __name__ == "__main__":
    main()