#!/bin/bash

repo_dir=$(dirname $BASH_SOURCE)/..
python $repo_dir/src/NASCAR_Scrape.py &&
  echo "Raw Done" &&
  python $repo_dir/src/NASCAR_Engineering.py &&
  echo 'Engineered' &&
  python $repo_dir/src/NASCAR_predict.py &&
  echo 'Predicted' &&
  streamlit run $repo_dir/src/streamlit_app.py