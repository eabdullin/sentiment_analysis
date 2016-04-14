FROM continuumio/anaconda
MAINTAINER Yelaman Abdullin <a.elaman.b@gmail.com>

COPY scripts/_start_jupyter.sh /root/start.sh
RUN chmod +x /root/start.sh
COPY environment.yml /root/
RUN conda env update -n=py27 -f=/root/environment.yml #  -q QUIET
