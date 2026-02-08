# base image  
FROM python:3.13-slim

# setup environment variables
ENV TZ="Europe/Berlin"
ENV DockerHOME=/home/app/spt

# set work directories 
RUN mkdir -p $DockerHOME \
    mkdir -pv /var/log/gunicorn/ \
    mkdir -pv /var/run/gunicorn/
WORKDIR $DockerHOME  

# disable saving python bytecode 
ENV PYTHONDONTWRITEBYTECODE=1
# enable unbuffered output for realtime logging
ENV PYTHONUNBUFFERED=1
# suppress warning about running pip as root, as its ok in docker
ENV PIP_ROOT_USER_ACTION=ignore
# use standardized build interface for building wheels
ENV PIP_USE_PEP517=true

# install dependencies 
COPY ./requirements.txt ${DockerHOME} 
RUN pip install --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt

# copy whole project
COPY . .

EXPOSE 8000

# start server
CMD ["gunicorn", "main:create_app()", "--bind", "0.0.0.0:8000", "--workers", "2"]