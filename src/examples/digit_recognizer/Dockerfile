FROM yuxio/flask-python351

MAINTAINER Kirill Panarin <kirill.panarin@gmail.com>

RUN apt-get update
RUN apt-get -y install git
RUN pip install --upgrade pip
RUN git clone https://github.com/stormy-ua/DeepLearningToy.git

WORKDIR /usr/src/app/DeepLearningToy/src/examples/digit_recognizer

EXPOSE 8081

FLASK_APP=app.py

CMD flask run --host=0.0.0.0 --port=8081

