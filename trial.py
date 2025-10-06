import pstats

p = pstats.Stats("profile_more_days.stats")
p.sort_stats("tottime").print_stats(100)