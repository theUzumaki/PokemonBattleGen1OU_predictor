#!/bin/bash

if [[ "$1" == "logistic" ]]; then
    bash run.sh predict_logistic
elif [[ "$1" == "xgboost" ]]; then
    bash run.sh predict_xgboost
elif [[ "$1" == "ensemble" ]]; then
    bash ensemble_three/predict_xgboost_ensemble_3.sh "$3" "$4"
else
    echo "Usage: $0 <logistic|xgboost|ensemble> run [input.jsonl] [output.csv]"
    echo "  logistic run               - Emit predictions using logistic regression model"
    echo "  xgboost run                - Emit predictions using xgboost model"
    echo "  ensemble run <in> <out>    - Emit predictions using xgboost ensemble model"
    exit 1
fi