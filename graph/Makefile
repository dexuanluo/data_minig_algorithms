task2:
	spark-submit bin/task2.py 2 data/sample_user_state.csv output/betweenness.out output/community.out
task1:
	spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 bin/task1.py 7 data/ub_sample_data.csv output/res.out

clean:
	rm -f output/*
	rm -rf spark-warehouse