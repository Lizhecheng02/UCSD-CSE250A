import numpy as np

# Movie titles
with open('hw7_movies.txt', 'r') as fid:
    movie_titles = [line.strip() for line in fid.readlines()]

# read rating
with open("hw7_ratings.txt", 'r') as f:
    data = []
    for line in f:
        # Split line and convert values
        row = []
        for val in line.strip().split():
            if val == '?':
                row.append(-1)
            else:
                row.append(int(val))
        data.append(row)
    ratings = np.array(data)

print("rating shape: ", ratings.shape)

# Model initialization
n_movie = len(movie_titles)
print("number of movies: ", n_movie)
np.random.seed(1)
k = 4
u = np.loadtxt('hw7_probR_init.txt').T
p = np.loadtxt("hw7_probZ_init.txt")

print("p shape", p.shape)
n_student = len(ratings)

# EM
for iterate in range(257):  # 0 to 256 inclusive
    # init value
    log_l = 0
    q = np.zeros((n_student, k))

    ### TODO: ####
    pass
    ##############

    # E-STEP
    for t in range(n_student):
        ### TODO: ####
        pass
        ##############

    # M-STEP

    ### TODO: ####
    pass
    ##############

    for j in range(n_movie):
        ### TODO: ####
        pass
        ##############

    if iterate in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        print(f'iterate: {iterate}\tlogL: {log_l/n_student:.4f}')
