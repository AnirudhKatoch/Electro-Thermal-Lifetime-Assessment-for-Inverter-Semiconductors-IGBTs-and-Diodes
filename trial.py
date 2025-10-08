
import pstats

stats = pstats.Stats("main_1.stats")
stats.strip_dirs().sort_stats("tottime").print_stats(50)