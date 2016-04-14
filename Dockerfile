# A container with scientific python libraries installed
# There are installed Theano, keras and xgboost

FROM eabdullin/keras-everware

MAINTAINER Yelaman Abdullin <a.elaman.b@gmail.com>

USER jupyter
WORKDIR /home/jupyter/

EXPOSE 8888