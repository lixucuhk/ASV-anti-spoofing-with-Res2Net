#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
baseline_CM
EOF
