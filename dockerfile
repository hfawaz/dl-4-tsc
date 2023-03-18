FROM tensorflow/tensorflow

COPY $pwd /dl-4-tsc
RUN pip install pandas scikit-learn matplotlib

RUN apt install git

ENTRYPOINT ["/bin/bash"]