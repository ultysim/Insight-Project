#!/bin/bash
python3 test.py

aws s3 cp "./test.py" "s3://simas-project-bucket/Insight_Project/test_py"
