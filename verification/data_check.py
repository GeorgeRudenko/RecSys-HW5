#!/usr/bin/env python
"""Independent verification of week5/data.js: split rule + HW5 metrics.

Re-implements the per-user chronological 80/10/10 split and every number
the HW5 report needs, from scratch with numpy only. It does not import or
run any week5/*.js code: if a number here disagrees with the page, one of
the two implementations has a bug.

Split rule mirrored from data.js:splitByTimestamp:
  - group ratings per user;
  - sort each user's ratings by (timestamp, movieId) ascending;
  - skip users with fewer than 5 ratings;
  - nTrain = floor(0.8 * n), nVal = floor(0.1 * n), nTest = n - nTrain - nVal.
Only membership matters: the JS Map iterates in first-appearance order, so the
concatenation order of the three arrays can differ from this script's.
"""

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK5 = os.path.join(os.path.dirname(HERE), "week5")
U_DATA = os.path.join(WEEK5, "u.data")
U_ITEM = os.path.join(WEEK5, "u.item")

MIN_RATINGS = 5
RELEVANT = 4.0


def load_movies(path):
    movies = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) < 24:
                continue
            movies.append((int(fields[0]), fields[1]))
    return movies


def load_ratings(path):
    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            rows.append(
                (int(fields[0]), int(fields[1]), float(fields[2]), int(fields[3]))
            )
    return rows


def split_by_timestamp(ratings):
    by_user = {}
    for row in ratings:
        by_user.setdefault(row[0], []).append(row)

    train, val, test = [], [], []
    skipped = []
    for user_id in sorted(by_user):
        user_rows = sorted(by_user[user_id], key=lambda r: (r[3], r[1]))
        n = len(user_rows)
        if n < MIN_RATINGS:
            skipped.append((user_id, n))
            continue
        n_train = math.floor(0.8 * n)
        n_val = math.floor(0.1 * n)
        train.extend(user_rows[:n_train])
        val.extend(user_rows[n_train : n_train + n_val])
        test.extend(user_rows[n_train + n_val :])
    return train, val, test, skipped


def rmse_against(mean_value, rows):
    if not rows:
        return 0.0
    actual = np.array([r[2] for r in rows], dtype=np.float64)
    return float(np.sqrt(np.mean((mean_value - actual) ** 2)))


def main():
    movies = load_movies(U_ITEM)
    ratings = load_ratings(U_DATA)
    train, val, test, skipped = split_by_timestamp(ratings)

    user_ids = np.array([r[0] for r in ratings])
    movie_ids = np.array([r[1] for r in ratings])

    n_users = int(user_ids.max())  # data.js: numUsers = highest user id
    distinct_users = int(np.unique(user_ids).size)
    n_movies = len(movies)
    max_movie_id = max(m[0] for m in movies)
    n_ratings = len(ratings)

    unique_pairs = set(zip(user_ids.tolist(), movie_ids.tolist()))
    duplicates = n_ratings - len(unique_pairs)
    cells = n_users * n_movies
    sparsity = 1.0 - len(unique_pairs) / cells

    train_mean = float(np.mean([r[2] for r in train])) if train else 0.0
    train_variance = float(np.var([r[2] for r in train])) if train else 0.0
    val_baseline = rmse_against(train_mean, val)
    test_baseline = rmse_against(train_mean, test)

    all_movie_ids = {m[0] for m in movies}
    train_movie_ids = {r[1] for r in train}
    absent_from_train = sorted(all_movie_ids - train_movie_ids)

    relevant_test = [r for r in test if r[2] >= RELEVANT]
    relevant_test_ids = {r[1] for r in relevant_test}
    unseen_ids = sorted(relevant_test_ids - train_movie_ids)
    relevant_rows_unseen = sum(1 for r in relevant_test if r[1] not in train_movie_ids)

    per_user_relevant = {}
    for r in relevant_test:
        per_user_relevant.setdefault(r[0], set()).add(r[1])
    test_users = {r[0] for r in test}
    evaluable_users = {
        u for u, ids in per_user_relevant.items() if ids & train_movie_ids
    }
    users_without_relevant = sorted(test_users - evaluable_users)

    print("=== week5/data.js independent check (numpy only) ===")
    print(f"users (max id)                 : {n_users}  (distinct in u.data: {distinct_users})")
    print(f"movies (u.item rows)           : {n_movies}  (max movie id: {max_movie_id})")
    print(f"ratings (u.data rows)          : {n_ratings}  (duplicate user-movie pairs: {duplicates})")
    print(
        f"sparsity of {n_users}x{n_movies} matrix: {sparsity:.4%} "
        f"(observed pairs {len(unique_pairs)}, density {len(unique_pairs) / cells:.4%})"
    )
    print(f"users skipped (<{MIN_RATINGS} ratings)    : {len(skipped)} {skipped if skipped else ''}")
    print(f"train / val / test sizes       : {len(train)} / {len(val)} / {len(test)}")
    print(f"training mean (baseline mu)    : {train_mean:.4f}")
    print(f"training rating variance       : {train_variance:.4f}  (MSE of predicting mu at init)")
    print(f"baseline RMSE (mu) on val      : {val_baseline:.4f}")
    print(f"baseline RMSE (mu) on test     : {test_baseline:.4f}")
    print(f"movies absent from train       : {len(absent_from_train)}")
    print(f"relevant test rows (>= {RELEVANT}): {len(relevant_test)}")
    print(
        f"relevant test movie ids absent from train: {len(unseen_ids)} "
        f"({relevant_rows_unseen} relevant test rows)"
    )
    print(
        f"users with >=1 relevant candidate test movie: {len(evaluable_users)} "
        f"(of {len(test_users)} test users)"
    )
    print(
        f"test users without one         : {len(users_without_relevant)}"
        + (f" {users_without_relevant[:20]}" if users_without_relevant else "")
    )


if __name__ == "__main__":
    main()
