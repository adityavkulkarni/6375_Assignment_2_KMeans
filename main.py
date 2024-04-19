import argparse
import re
import string

import matplotlib.pyplot as plt
import pandas as pd

from k_means_clustering import KMeansClustering


def get_preprocessed_data(file="data/bbchealth.txt"):
    lines = []
    with open(file, encoding="utf8", errors='ignore') as f:
        print(f"Reading {file}")
        for line in f.readlines():
            lines.append(line)

    for i in range(len(lines)):
        # Remove the tweet id and timestamp
        delim = lines[i].split("|")[2:]
        lines[i] = " | ".join(delim)

        # Remove any word that starts with the symbol @
        lines[i] = " ".join(filter(lambda x: x[0] != '@', lines[i].split()))

        # Removing any hash-tags symbols
        lines[i] = lines[i].replace('#', '')

        # Remove any URL
        lines[i] = re.sub(r"http\S+", "", lines[i])
        lines[i] = re.sub(r"www\S+", "", lines[i])
        lines[i] = lines[i].strip()

        # Convert every word to lowercase
        lines[i] = lines[i].lower()

        # Remove punctuations
        lines[i] = re.sub('[^A-Za-z0-9 ]+', '', lines[i])
        lines[i] = " ".join(lines[i].split())
    print(f"Total tweets: {len(lines)}\n")
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="gdnhealthcare.txt")
    parser.add_argument('--max_k', type=int, default=10)
    args = parser.parse_args()
    # Data preprocessing
    tweets = get_preprocessed_data(file=f"data/{args.file}")
    # Experiments
    results = []
    for k in range(3, args.max_k+1, 1):
        print(f"Running k-means clustering with {k} clusters")
        kmeans = KMeansClustering(k)
        results.append((k, kmeans.train(tweets),
                        "\n".join(kmeans.print_clusters())))
        print()
    # Create the plot
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.style.use('ggplot')
    plt.plot([t[0] for t in results], [t[1] for t in results], marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.savefig(f"results/k_means_elbow_{args.file.split('.')[0]}.png")
    # Save results
    pd.DataFrame(results, columns=["k", "SSE", "Clusters"]).to_csv(
        f"results/k_means_{args.file.split('.')[0]}.csv", index=False)
