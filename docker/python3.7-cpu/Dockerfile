FROM python:3.7-slim

ADD requirements.txt /tmp
RUN pip install pip setuptools -U && pip install --upgrade pip && pip install -r /tmp/requirements.txt