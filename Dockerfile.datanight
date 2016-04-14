FROM yandex/rep:0.6.5
MAINTAINER Andrey Ustyuzhanin <andrey.u@gmail.com>

COPY scripts/_start_jupyter.sh /root/start.sh
RUN chmod +x /root/start.sh
COPY environment.yml /root/
RUN /root/miniconda/bin/conda env update -n=py27 -f=/root/environment.yml #  -q QUIET