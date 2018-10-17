if [ $# -eq 0 ]
then
	echo "Usage:\n\t sh import_webpage_datasets_to_mongodb.sh HOST:PORT(HOST and PORT of mongodb)"
	echo "Example:\n\t sh import_webpage_datasets_to_mongodb.sh 127.0.0.1:27017"
	exit
fi

if [ ! -d "CQA_datasets" ]; then
	unzip CQA_webpage_datasets.zip
fi

HOST=$1
mongorestore --host $HOST --db quora --collection qa CQA_webpage_datasets/quora/qa.bson
mongorestore --host $HOST --db qa_zhihu --collection qa CQA_webpage_datasets/zhihu/qa.bson
