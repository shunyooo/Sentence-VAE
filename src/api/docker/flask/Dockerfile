FROM python:3.7-slim

ADD requirements.txt /tmp
RUN pip install pip setuptools -U && pip install --upgrade pip && pip install -r /tmp/requirements.txt

RUN apt-get update && apt-get -y install curl \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz  \
    && mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh
