#!/bin/bash

repo_dir=$(dirname $BASH_SOURCE)/..
python $repo_dir/src/XFinity_Scrape.py &&
  echo "Raw Done" &&
  python $repo_dir/src/XFinity_Engineering.py &&
  echo 'Engineered' &&
  python $repo_dir/src/XFinity_predict.py &&
  echo 'Predicted' &&
  streamlit run $repo_dir/src/streamlit_appX.py