FROM python:3.10.6-buster

COPY . .

WORKDIR ./

RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:$PWD/model"

RUN export PYTHONPATH="${PYTHONPATH}:$PWD/model"

ENV PYTHONPATH="${PYTHONPATH}:$PWD/model"

RUN echo $PYTHONPATH

EXPOSE 5000

CMD ["python3", "main.py"]
