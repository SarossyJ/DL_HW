FROM python:3.9.12

RUN apt-get update &&  \
	apt install -y  \
	openssh-server \
	git \
	libgl1 \
	vim \
	tmux \
	byobu

ENV HOME /home/model_mavericks
RUN mkdir -p $HOME/
WORKDIR $HOME/


COPY . .
COPY ssh/sshd_config /etc/ssh/sshd_config

EXPOSE 22
EXPOSE 8888


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=10000

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 model_mavericks
RUN echo 'model_mavericks:password' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

SHELL ["/bin/bash", "-l", "-c"]

# Start Jupyter lab with custom password
ENTRYPOINT service ssh start && jupyter-lab \
	--ip 0.0.0.0 \
	--port 8888 \
	--no-browser \
	--NotebookApp.notebook_dir='$home' \
	--ServerApp.terminado_settings="shell_command=['/bin/bash']" \
	--allow-root