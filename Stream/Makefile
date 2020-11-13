task3:
	spark-submit bin/task3.py 8087 output/task3.out
task2:
	spark-submit bin/task2.py 8087 output/task2.out
task1:
	spark-submit bin/task1.py data/business_first.json data/business_second.json output/task1.out
dgen:
	java -cp data/generate_stream.jar StreamSimulation data/business.json 8087 100
clean:
	rm -f output/*
	rm -rf bin/__pycache__