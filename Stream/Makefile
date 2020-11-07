task1:
	spark-submit bin/task1.py data/business_first.json data/business_second.json output/task1.out
dgen:
	java -cp data/generate_stream.jar StreamSimulation data/business.json 9999 100
clean:
	rm -f output/*
	rm -rf bin/__pycache__