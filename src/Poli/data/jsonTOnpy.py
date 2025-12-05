import json
import numpy as np
import os
import argparse

def main(json_path, output_dir="npy"):
    # create output folder if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading JSON dataset: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    num_points = data["metadata"]["num_points"]
    k_neighbors = data["metadata"]["k_neighbors"]
    frames = data["frames"]

    print(f"Dataset contains {len(frames)} frames.")
    print(f"Points per frame: {num_points}")
    print(f"K neighbors: {k_neighbors}")

    for i, frame in enumerate(frames):
        pts = np.array(frame["points"], dtype=np.float32).reshape(num_points, 3)
        knn = np.array(frame["knn"], dtype=np.int32).reshape(num_points, k_neighbors)

        np.save(f"{output_dir}/frame_{i:04d}_points.npy", pts)
        np.save(f"{output_dir}/frame_{i:04d}_knn.npy", knn)

        if i % 20 == 0:
            print(f"Saved frame {i}/{len(frames)}")

    print("Done.")

    # OPZIONALE: salva tutto in un unico .npz compatto
    save_npz = True
    if save_npz:
        print("Saving unified dataset.npz ...")
        all_points = np.stack(
            [np.array(f["points"]).reshape(num_points, 3) for f in frames],
            axis=0
        )
        all_knn = np.stack(
            [np.array(f["knn"]).reshape(num_points, k_neighbors) for f in frames],
            axis=0
        )

        np.savez_compressed(
            f"{output_dir}/dataset.npz",
            points=all_points,
            knn=all_knn
        )
        print("Saved dataset.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to the JSON dataset file")
    parser.add_argument("--out", default="npy", help="Output directory")
    args = parser.parse_args()

    main(args.json_path, args.out)