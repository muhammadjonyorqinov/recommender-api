# base image
FROM continuumio/miniconda3

# project dependencies
RUN conda install pandas
RUN conda install seaborn
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge matplotlib

RUN conda install -c conda-forge flask
RUN conda install -c conda-forge keras
RUN conda install https://anaconda.org/conda-forge/scikit-surprise/1.1.1/download/linux-64/scikit-surprise-1.1.1-py39h1dff97c_1.tar.bz2
RUN echo $(conda --version)
RUN echo $(python --version)
RUN echo $(conda list)

# create working directory
WORKDIR /app

# copy all the source files from current directory to docker
COPY . .

# expose port
#EXPOSE 80

# start the application
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0", "--port=80"]

