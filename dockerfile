FROM tensorflow/tensorflow
RUN 
RUN apt update
RUN apt install -y git wget zip
RUN pip install pandas scikit-learn matplotlib
COPY $pwd /dl-4-tsc
ENTRYPOINT ["/bin/bash"]