FROM continuumio/anaconda
MAINTAINER Kamil Kwiek <kamil.kwiek@continuum.io>

RUN conda env update -n=py27 -f=anaconda-base/environment.yml #  -q QUIET

RUN mkdir -p /srv/
WORKDIR /srv/

# fetch juptyerhub-singleuser entrypoint
ADD https://raw.githubusercontent.com/jupyter/jupyterhub/master/jupyterhub/singleuser.py /usr/local/bin/jupyterhub-singleuser
RUN chmod 755 /usr/local/bin/jupyterhub-singleuser

# jupyter is our user
RUN useradd -m -s /bin/bash jupyter

USER jupyter
ENV HOME /home/jupyter
ENV SHELL /bin/bash
ENV USER jupyter

WORKDIR /home/jupyter/

EXPOSE 8888

ADD anaconda-base/singleuser.sh /srv/singleuser/singleuser.sh
CMD ["sh", "/srv/singleuser/singleuser.sh"]