SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
BASE_DIR=$SCRIPT_DIR
echo $BASE_DIR
docker build -t flask:1.0 docker/flask
docker run --rm -it -v $BASE_DIR:$HOME -p 8000:8000 -w $HOME flask:1.0 python main.py