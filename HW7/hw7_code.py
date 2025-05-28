import numpy as np


def part_a(movies, ratings):
    seen = ratings != -1
    recs = ratings == 1
    n_seen = seen.sum(axis=0)
    n_recs = recs.sum(axis=0)
    popularity = n_recs / n_seen

    idx = np.argsort(popularity)
    print("\nPart (a): Movie popularity from least to most:\n")
    for i in idx:
        print(f"{movies[i]:40s}  {popularity[i]:.4f} ({n_recs[i]} / {n_seen[i]})")


def run_em(ratings, p_init, u_init, iterations=256, eps=1e-6):
    T, n_movies = ratings.shape
    p = p_init.copy()
    u = np.clip(u_init.copy(), eps, 1 - eps)
    seen = (ratings != -1)
    pos = (ratings == 1)

    ll = []
    for it in range(iterations + 1):
        log_p = np.log(p)
        log_u = np.log(u)
        log_1mu = np.log(1 - u)
        log_joint = np.zeros((T, p.size))
        for i in range(p.size):
            lj = np.full(T, log_p[i])
            arr = np.where(seen, pos * log_u[:, i] + (~pos) * log_1mu[:, i], 0)
            lj += arr.sum(axis=1)
            log_joint[:, i] = lj
        max_lj = np.max(log_joint, axis=1, keepdims=True)
        log_sum = max_lj + np.log(np.exp(log_joint - max_lj).sum(axis=1, keepdims=True) + eps)
        ll.append(log_sum.mean())
        if it == iterations:
            break
        log_post = log_joint - log_sum
        q = np.exp(log_post)
        p = q.sum(axis=0) / T
        den = q.sum(axis=0)
        new_u = np.zeros_like(u)
        for j in range(n_movies):
            seen_t = seen[:, j]
            sum_pos = (q[seen_t] * (ratings[seen_t, j][:, None] == 1)).sum(axis=0)
            sum_unseen = den - q[seen_t].sum(axis=0)
            new_u[j] = (sum_pos + u[j] * sum_unseen) / den
        u = np.clip(new_u, eps, 1 - eps)
    return ll, p, u


def part_e(ratings):
    p_init = np.loadtxt("hw7_probZ_init.txt")
    u_init = np.loadtxt("hw7_probR_init.txt")
    ll_vals, p_final, u_final = run_em(ratings, p_init, u_init)
    print("\nPart (e): Log-likelihood during EM:\n")
    print(f"{'iter':>4s}   {'log-likelihood':>15s}")
    for it in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        print(f"{it:4d}   {ll_vals[it]:15.4f}")
    return p_final, u_final


def part_f(movies, ratings, p, u, pids_file="hw7_pids.txt"):
    pids = [line.strip() for line in open(pids_file)]
    pid = input("\nEnter your PID: ").strip()
    if pid not in pids:
        print("PID not found; check your PID and try again.")
        return
    t = pids.index(pid)
    seen = (ratings != -1)
    log_p = np.log(p)
    log_u = np.log(u)
    log_1mu = np.log(1 - u)
    log_joint = np.zeros(p.size)
    for i in range(p.size):
        lj = log_p[i]
        idx = seen[t]
        r = ratings[t, idx]
        lj += (r * log_u[idx, i] + (1 - r) * log_1mu[idx, i]).sum()
        log_joint[i] = lj
    max_lj = np.max(log_joint)
    post = np.exp(log_joint - (max_lj + np.log(np.exp(log_joint - max_lj).sum())))
    expected = u.dot(post)
    unseen = np.where(~seen[t])[0]
    ranked = sorted(unseen, key=lambda j: expected[j], reverse=True)
    print(f"\nPart (f): Recommended unseen movies for PID {pid}:\n")
    for j in ranked:
        print(f"{movies[j]:40s} {expected[j]:.4f}")


if __name__ == "__main__":
    movies = [line.strip() for line in open("hw7_movies.txt")]
    ratings = []
    with open("hw7_ratings.txt") as f:
        for line in f:
            ratings.append([(-1 if v == "?" else int(v)) for v in line.strip().split()])
    ratings = np.array(ratings)
    part_a(movies, ratings)
    p_final, u_final = part_e(ratings)
    part_f(movies, ratings, p_final, u_final)
