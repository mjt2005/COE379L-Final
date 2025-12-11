FROM python:3.11

RUN pip install tensorflow==2.19.0
RUN pip install Flask==3.0
#RUN pip install numpy==1.26
RUN pip install scikit-image==0.21
RUN pip install Pillow==12.0.0
RUN pip install diffusers==0.36.0
RUN pip install transformers==4.51.3
RUN pip install accelerate==1.6.0

RUN mkdir /images
RUN mkdir /app
WORKDIR /app


COPY models/ /app/models
COPY api.py /app/api.py

CMD ["python", "api.py"]