FROM python:3.9.12

RUN apt-get update &&  \
	apt install -y 

ENV HOME /home/model_mavericks
RUN mkdir -p $HOME/project1
WORKDIR $HOME/project1


COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 model_mavericks
RUN echo 'model_mavericks:password' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

ENTRYPOINT python entrypoint.py 