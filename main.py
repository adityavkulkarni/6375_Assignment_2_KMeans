import argparse
import os
import re
import string

import matplotlib.pyplot as plt
import pandas as pd

from k_means_clustering import KMeansClustering


def get_preprocessed_data(tweet_cnt, file="data/bbchealth.txt"):
    lines = []
    if tweet_cnt != 0:
        for file in os.listdir("data/"):
            print(f"Reading {file}")
            with open(f"data/{file}", encoding="utf8", errors='ignore') as f:
                for line in f.readlines():
                    lines.append(line)
    else:
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

        # Remove any URL using the regular expression
        lines[i] = re.sub(r"http\S+", "", lines[i])
        lines[i] = re.sub(r"www\S+", "", lines[i])
        lines[i] = lines[i].strip()
        tweet_len = len(lines[i])
        if tweet_len > 0:
            if lines[i][len(lines[i]) - 1] == ':':
                lines[i] = lines[i][:len(lines[i]) - 1]

        # Convert every word to lowercase
        lines[i] = lines[i].lower()

        lines[i] = lines[i].translate(str.maketrans('', '', string.punctuation))
        lines[i] = " ".join(lines[i].split())
    lines = lines[:tweet_cnt] if tweet_cnt else lines
    print(f"Total tweets: {len(lines)}\n")
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--max_k', type=int, default=100)
    args = parser.parse_args()
    # Data preprocessing
    tweets = get_preprocessed_data(tweet_cnt=args.data_size)
    # Experiments
    results = []
    for k in range(3, args.max_k, 3):
        print(f"Running k-means clustering with {k} clusters")
        kmeans = KMeansClustering(k)
        results.append((k, kmeans.train(tweets)))
        kmeans.print_clusters()
        print()
    # Create the plot
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.style.use('ggplot')
    plt.plot([t[0] for t in results], [t[1] for t in results], marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.savefig("results/k_means_elbow.png")
    # Save results
    pd.DataFrame(results, columns=["k", "SSE"]).to_csv("results/k_means.csv", index=False)
