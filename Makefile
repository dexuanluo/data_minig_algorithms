clean:
	rm -f */output/*
	rm -rf */bin/__pycache__
unzipdata:
	tar -xzvf Apriori/data.tar.gz
	tar -xzvf CF_LSH/data.tar.gz
	tar -xzvf Clustering/data.tar.gz
	tar -xzvf Graph/data.tar.gz
	rm -rf */data.tar.gz
zipdata:
	tar -czvf Apriori/data.tar.gz Apriori/data/*
	tar -czvf CF_LSH/data.tar.gz CF_LSH/data/*
	tar -czvf Clustering/data.tar.gz Clustering/data/*
	tar -czvf Graph/data.tar.gz Graph/data/*
	rm -rf */data