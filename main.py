import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd

from k_means_clustering import KMeansClustering


def get_preprocessed_data(file="data/bbchealth.txt"):
    """
    Reads data from file and returns preprocessed data
    :param file: path to the txt file (default: data/bbchealth.txt).
    :return:
    """
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

        # Remove punctuations and symbols
        lines[i] = re.sub('[^A-Za-z0-9 ]+', '', lines[i])
        lines[i] = " ".join(lines[i].split())
    print(f"Total tweets: {len(lines)}\n")
    return lines


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description="Script to run k-means clustering on tweets and generate results")
    parser.add_argument('--file', type=str, default="gdnhealthcare.txt",
                        help="path to the txt file (default: data/gdnhealthcare.txt)")
    parser.add_argument('--max_k', type=int, default=10,
                        help="maximum number of clusters to use: results will be generated "
                             "from k=3 to max_k (default: 10)")
    args = parser.parse_args()
    # Data preprocessing
    tweets = get_preprocessed_data(file=f"data/{args.file}")
    # Experiments
    results = []
    for k in list(range(3, args.max_k+1, 1)) + [15, 20, 25, 40, 50,  75, 100]:
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
