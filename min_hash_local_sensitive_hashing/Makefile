task3: 
	spark-submit bin/task3train.py data/train_review.json output/task3.model item_based
task3p: output/task3.model
	spark-submit bin/task3predict.py data/train_review.json data/test_review.json output/task3.model output/task3item.predict item_based
task2p: output/task2.model
	spark-submit bin/task2predict.py data/test_review.json output/task2.model output/task2.predict
task2:
	spark-submit bin/task2train.py data/train_review.json output/task2.model data/stopwords
task1:
	spark-submit bin/task1.py data/train_review.json outpu/task1.res
clean:
	rm -f output/*