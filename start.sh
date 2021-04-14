RUN echo "#!/bin/bash
cd /opt
PhoXiControl &
cd ~
cd qt_calibration/dkqt_calibration/build
./main" > start.sh
RUN chmod 777 start.sh
RUN ./start.sh