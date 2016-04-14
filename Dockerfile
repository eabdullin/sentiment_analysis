FROM everware/base

MAINTAINER Project Everware

USER root
COPY environment.yml /root/
RUN conda env update -n=py27 -f=/root/environment.yml #  -q QUIET

USER jupyter
WORKDIR /home/jupyter/

EXPOSE 8888