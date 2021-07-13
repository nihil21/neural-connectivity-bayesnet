FROM python:latest

RUN pip install --upgrade pip
RUN pip install jupyter pandas sklearn nilearn pgmpy matplotlib ipympl tqdm

# During development, the src folder will be overriden by a volume
COPY ./src /app/src

WORKDIR /app/src
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]
