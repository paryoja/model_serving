FROM ubuntu:19.10

RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y dist-upgrade


WORKDIR /work

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install tensorflow

ADD ./requirements.txt /work/
RUN python3.7 -m pip install -r requirements.txt


ADD ./src ./src
ENV FLASK_APP=/work/src/app.py
CMD ["flask", "run"]
