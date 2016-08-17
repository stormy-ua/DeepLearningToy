FROM yuxio/flask-python351

MAINTAINER Kirill Panarin <kirill.panarin@gmail.com>

RUN apt-get update -y
RUN apt-get -y install git
RUN pip install --upgrade pip
RUN git clone https://github.com/stormy-ua/DeepLearningToy.git

WORKDIR /usr/src/app/DeepLearningToy/src/examples/digit_recognizer

EXPOSE 5000

RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]

