task11:
	spark-submit bin/task1.py 1 4 data/small2.csv output/task1.out
task2:
	spark-submit --executor-memory 4G --driver-memory 4G bin/task2.py 200 data/user_state.csv output/task2.out2

clean:
	rm -f output/*
	