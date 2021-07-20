FROM python:latest

RUN apt-get install git

RUN pip install --upgrade pip
RUN pip install jupyter pandas nilearn tslearn psutil matplotlib \
    black jupyter_contrib_nbextensions

# Enable Black extension for formatting
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
RUN jupyter nbextension enable jupyter-black-master/jupyter-black

# Install pgmpy from source
RUN git clone https://github.com/pgmpy/pgmpy && \
    cd pgmpy/ && \
    pip install -r requirements.txt && \
    python setup.py install

# During development, the src folder will be overriden by a volume
COPY ./src /app/src

WORKDIR /app/src
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]
