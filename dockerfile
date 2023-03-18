FROM tensorflow/tensorflow
RUN pip install pandas scikit-learn matplotlib
RUN apt install -y git 
COPY $pwd /dl-4-tsc
ENTRYPOINT ["/bin/bash"]